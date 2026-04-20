"""Evaluator for the protein GO-term prediction task (Molecular Function sub-ontology).

Two scoring paths:

1. Per-protein score (used during evolution): For a single protein, sweep
   confidence thresholds from 0.01 to 0.99 in steps of 0.01; at each threshold,
   propagate the predicted set to include all MFO ancestors and compute F1
   against the (propagated) ground-truth set; return the max. This provides
   a smooth per-example signal in [0, 1] for Elo/GEPA/Autoresearch to use.

2. Batch CAFA Fmax (used for test reporting): Write all test-set predictions
   to a CAFA-format TSV file and invoke `cafaeval` (Piovesan et al., 2024) to
   compute the canonical Fmax reported by CAFA papers. This is the headline
   number directly comparable to published methods. Exposed via
   `compute_cafa_fmax_batch()` below.

The two are correlated but not identical — CAFA Fmax averages per-threshold
metrics across the whole test set before picking the max threshold, while the
per-protein score picks the best threshold per protein individually. The
per-protein score is a valid signal for optimization but should not be reported
as Fmax.
"""

from __future__ import annotations

import json
import logging
import re
import tempfile
import threading
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import litellm

from RoboPhD.eval_utils import retry_on_rate_limit, exec_with_stdout_capture
from tools import (
    make_blast, make_uniprot, make_go_ancestors, make_sequence_features,
    propagate_to_mfo_ancestors,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

# CAFA-evaluator config
GO_OBO_PATH = DATA_DIR / "go-basic.obo"


# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------

@dataclass
class CostTracker:
    budget: float
    llm_cost: float = 0.0
    embed_cost: float = 0.0
    llm_calls: int = 0
    embed_calls: int = 0
    llm_prompts: List[str] = field(default_factory=list)
    llm_responses: List[str] = field(default_factory=list)

    @property
    def total(self) -> float:
        return self.llm_cost + self.embed_cost


def make_tracked_llm(model: str, tracker: CostTracker) -> Callable[..., str]:
    """Return an llm(prompt) -> str callable with cost tracking and rate limit retry."""

    def llm(prompt: str, temperature: float = 0.0) -> str:
        resp = retry_on_rate_limit(
            lambda: litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                timeout=300,
                num_retries=0,
            )
        )
        try:
            cost = litellm.completion_cost(completion_response=resp)
        except Exception:
            cost = 0.0
        text = resp.choices[0].message.content or ""
        tracker.llm_cost += cost
        tracker.llm_calls += 1
        tracker.llm_prompts.append(prompt)
        tracker.llm_responses.append(text)
        return text

    return llm


def make_tracked_embed(model: str, tracker: CostTracker) -> Callable[[str], List[float]]:
    """Return an embed(text) -> list[float] callable with cost tracking and rate limit retry."""

    def embed(text: str) -> List[float]:
        resp = retry_on_rate_limit(
            lambda: litellm.embedding(model=model, input=[text], timeout=300, num_retries=0)
        )
        try:
            cost = litellm.completion_cost(completion_response=resp)
        except Exception:
            cost = 0.0
        tracker.embed_cost += cost
        tracker.embed_calls += 1
        return resp.data[0]["embedding"]

    return embed


# ---------------------------------------------------------------------------
# Agent execution
# ---------------------------------------------------------------------------

def _run_agent(agent_code: str, sequence: str, tools: Dict[str, Callable]) -> Tuple[Any, str]:
    """Execute agent_code, invoke predict(), capture stdout. Returns (result, stdout)."""

    def _call_predict(ns: Dict[str, Any]) -> Any:
        if "predict" not in ns:
            raise RuntimeError("Agent code does not define a `predict` function")
        return ns["predict"](
            sequence,
            tools["blast"],
            tools["uniprot"],
            tools["go_ancestors"],
            tools["sequence_features"],
            tools["llm"],
            tools["embed"],
        )

    namespace, stdout = exec_with_stdout_capture(agent_code, then=_call_predict)
    return namespace["_result"], stdout


def _clean_predictions(raw: Any) -> Dict[str, float]:
    """Coerce the agent's return value into {go_id: float} with valid GO IDs in [0,1]."""
    if not isinstance(raw, dict):
        return {}
    cleaned: Dict[str, float] = {}
    for k, v in raw.items():
        k_str = str(k).strip()
        if not re.fullmatch(r"GO:\d{7}", k_str):
            continue
        try:
            v_float = float(v)
        except (TypeError, ValueError):
            continue
        if 0.0 <= v_float <= 1.0:
            cleaned[k_str] = v_float
    return cleaned


# ---------------------------------------------------------------------------
# Per-protein scoring (used during evolution)
# ---------------------------------------------------------------------------

def _per_protein_fmax(
    predictions: Dict[str, float],
    ground_truth: List[str],
) -> Tuple[float, float, float, float]:
    """Compute max F1 over confidence thresholds for a single protein.

    Returns (fmax, best_precision, best_recall, best_threshold).
    Both predictions and ground truth are propagated to MFO ancestors.
    Thresholds are integer-stepped (tau = i/100 for i in 1..99) to match
    cafaeval's internal behavior and avoid floating-point drift.
    """
    if not predictions:
        return 0.0, 0.0, 0.0, 0.0
    gt_set = propagate_to_mfo_ancestors(set(ground_truth))
    if not gt_set:
        return 0.0, 0.0, 0.0, 0.0

    best = (0.0, 0.0, 0.0, 0.0)
    for i in range(1, 100):
        tau = i / 100
        above = {g for g, s in predictions.items() if s >= tau}
        if not above:
            continue
        pred_set = propagate_to_mfo_ancestors(above)
        tp = len(pred_set & gt_set)
        if tp == 0:
            continue
        precision = tp / len(pred_set)
        recall = tp / len(gt_set)
        f1 = 2 * precision * recall / (precision + recall)
        if f1 > best[0]:
            best = (f1, precision, recall, tau)
    return best


# ---------------------------------------------------------------------------
# Evaluator class — called per-example by optimize_anything
# ---------------------------------------------------------------------------

class ProteinGOEvaluator:
    """Evaluator for protein GO-term prediction (Molecular Function sub-ontology).

    Per-example, returns (score, diagnostics). The score is the per-protein Fmax.
    For the canonical batch CAFA Fmax, use compute_cafa_fmax_batch() separately
    after collecting predictions on the full test set.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        embed_model: str = DEFAULT_EMBED_MODEL,
        cost_budget: float = 0.10,
        over_budget_penalty: float = 0.9,
    ):
        self.model = model
        self.embed_model = embed_model
        self.cost_budget = cost_budget
        self.over_budget_penalty = over_budget_penalty

        self._eval_count = 0
        self._total_cost = 0.0
        self._last_logged = 0
        self._lock = threading.Lock()

        # Pre-warm caches once per evaluator instance so the first iteration
        # doesn't pay the load-time cost in a user-visible way.
        _ = self._warmup()

    def _warmup(self) -> bool:
        """Touch the SwissProt and GO DAG caches. Returns True if both loaded."""
        try:
            # Both are module-level singletons; calling the factories triggers
            # lazy loads inside tools.py.
            make_uniprot()("")  # dummy lookup triggers load
            make_go_ancestors()("GO:0003674")  # MFO root
            return True
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")
            return False

    def __call__(
        self,
        candidate: Dict[str, str],
        example: Dict[str, Any],
        *,
        problem_dir: Optional[Path] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        agent_code = candidate.get("agent.py", "")
        sequence = example["sequence"]
        ground_truth = example["go_terms_mfo"]  # list of GO IDs

        tracker = CostTracker(budget=self.cost_budget)
        tools = {
            "blast": make_blast(),
            "uniprot": make_uniprot(),
            "go_ancestors": make_go_ancestors(),
            "sequence_features": make_sequence_features(),
            "llm": make_tracked_llm(self.model, tracker),
            "embed": make_tracked_embed(self.embed_model, tracker),
        }

        # Run the agent
        try:
            raw_predictions, agent_stdout = _run_agent(agent_code, sequence, tools)
        except Exception as e:
            tb = traceback.format_exc()
            with self._lock:
                self._eval_count += 1
            return 0.0, {
                "error.md": f"Agent crashed: {e}\n\n```\n{tb}\n```",
                "agent_stdout": "",
                "sequence_length": len(sequence),
                "ground_truth": ground_truth,
                "cost_usd": tracker.total,
                "predictions": {},
            }

        predictions = _clean_predictions(raw_predictions)
        n_raw = len(raw_predictions) if isinstance(raw_predictions, dict) else 0
        n_valid = len(predictions)

        fmax, prec, rec, best_tau = _per_protein_fmax(predictions, ground_truth)
        over_budget = tracker.total > self.cost_budget
        score = fmax * (self.over_budget_penalty if over_budget else 1.0)

        # Stats
        with self._lock:
            self._eval_count += 1
            self._total_cost += tracker.total
            n = self._eval_count
            total_cost = self._total_cost
            milestone = n // 50 * 50
            should_log = milestone > 0 and milestone > self._last_logged
            if should_log:
                self._last_logged = milestone
        if should_log:
            logger.info(
                f"GO-MFO evaluator: {milestone} evaluations completed "
                f"(${total_cost:.2f} spent)"
            )

        # Diagnostics (string values with .md suffix get written as files for the
        # evolution AI; non-string values are returned in-memory only).
        diagnostics: Dict[str, Any] = {
            "score": score,
            "fmax": fmax,
            "best_precision": prec,
            "best_recall": rec,
            "best_threshold": best_tau,
            "over_budget": over_budget,
            "cost_usd": tracker.total,
            "cost_llm": f"${tracker.llm_cost:.4f}",
            "cost_embed": f"${tracker.embed_cost:.4f}",
            "num_llm_calls": tracker.llm_calls,
            "num_embed_calls": tracker.embed_calls,
            "sequence_length": len(sequence),
            "num_predictions_raw": n_raw,
            "num_predictions_valid": n_valid,
            "num_ground_truth_terms": len(ground_truth),
            # Structured field for downstream batch CAFA Fmax aggregation (callers
            # pair this with the source dataset by list index to recover the
            # accession — the eval_result's per_example_diagnostics preserves order).
            "predictions": predictions,
            # Human-readable forms for the evolution AI:
            "ground_truth.md": _format_go_list("Ground-truth GO-MFO terms", ground_truth),
            "predictions.md": _format_predictions(predictions),
        }

        if n_raw > n_valid:
            diagnostics["warning.md"] = (
                f"{n_raw - n_valid} of {n_raw} returned predictions were filtered as "
                f"invalid (malformed GO ID or score outside [0,1])."
            )

        if agent_stdout.strip():
            diagnostics["agent_stdout"] = agent_stdout

        # Preview LLM traffic
        for i, (p, r) in enumerate(zip(tracker.llm_prompts, tracker.llm_responses)):
            diagnostics[f"llm_prompt_{i+1}.md"] = _preview(p)
            diagnostics[f"llm_response_{i+1}.md"] = r

        return score, diagnostics

    @property
    def total_evaluations(self) -> int:
        return self._eval_count

    @property
    def total_cost(self) -> float:
        return self._total_cost


def _format_go_list(title: str, terms: List[str]) -> str:
    if not terms:
        return f"# {title}\n\n(none)"
    lines = [f"# {title}", "", *(f"- {t}" for t in sorted(terms))]
    return "\n".join(lines)


def _format_predictions(predictions: Dict[str, float]) -> str:
    if not predictions:
        return "# Predictions\n\n(none)"
    lines = ["# Predictions", "", "| GO term | Score |", "|---------|-------|"]
    for go_id, score in sorted(predictions.items(), key=lambda x: -x[1]):
        lines.append(f"| {go_id} | {score:.3f} |")
    return "\n".join(lines)


def _preview(text: str, head: int = 500, tail: int = 500) -> str:
    if len(text) <= head + tail + 50:
        return text
    return text[:head] + f"\n\n... [{len(text) - head - tail} chars omitted] ...\n\n" + text[-tail:]


# ---------------------------------------------------------------------------
# Batch CAFA Fmax computation (used for test reporting)
# ---------------------------------------------------------------------------

def compute_cafa_fmax_batch(
    predictions_by_protein: Dict[str, Dict[str, float]],
    ground_truth_by_protein: Dict[str, List[str]],
    out_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Compute canonical CAFA Fmax using CAFA-evaluator on the full batch.

    This is the number directly comparable to published CAFA/ProteInfer/CLEAN
    results. It differs from the per-protein score used during evolution: CAFA
    Fmax averages precision/recall across all proteins at each threshold, then
    picks the threshold that maximizes the averaged F1.

    Args:
        predictions_by_protein: {protein_id: {go_id: confidence}}
        ground_truth_by_protein: {protein_id: [go_ids]}
        out_dir: Where to write intermediate files. If None, uses a tempdir.

    Returns:
        Dict with keys: fmax, best_threshold, precision_at_best, recall_at_best,
        coverage, num_proteins, num_predictions, cafaeval_df_path.
    """
    try:
        from cafaeval.evaluation import cafa_eval
    except ImportError:
        raise ImportError(
            "cafaeval is not installed. Run: pip install cafaeval"
        )

    if not GO_OBO_PATH.exists():
        raise FileNotFoundError(
            f"GO OBO file not found at {GO_OBO_PATH}. Run setup.sh first."
        )

    workdir = out_dir or Path(tempfile.mkdtemp(prefix="cafa_eval_"))
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    pred_dir = workdir / "predictions"
    pred_dir.mkdir(exist_ok=True)

    # Write predictions in CAFA format: <protein_id>\t<go_id>\t<score>
    pred_file = pred_dir / "evolved_agent.tsv"
    with open(pred_file, "w") as f:
        for pid, preds in predictions_by_protein.items():
            for go_id, score in preds.items():
                f.write(f"{pid}\t{go_id}\t{score:.4f}\n")

    # Write ground truth: <protein_id>\t<go_id>
    gt_file = workdir / "ground_truth.tsv"
    with open(gt_file, "w") as f:
        for pid, terms in ground_truth_by_protein.items():
            for go_id in terms:
                f.write(f"{pid}\t{go_id}\n")

    # Run CAFA-evaluator
    logger.info(f"Running cafaeval on {len(predictions_by_protein)} proteins...")
    df_all, best = cafa_eval(
        str(GO_OBO_PATH),
        str(pred_dir),
        str(gt_file),
        norm="cafa",  # standard CAFA normalization
        prop="max",   # standard propagation
        th_step=0.01,
    )

    # Extract the best-F1 row
    if "f" not in best or len(best["f"]) == 0:
        raise RuntimeError("cafa_eval did not return an F-measure best row")
    best_row = best["f"].iloc[0]

    # Save the full dataframe for plotting / inspection
    df_path = workdir / "evaluation_all.tsv"
    df_all.to_csv(df_path, sep="\t")

    return {
        "fmax": float(best_row["f"]),
        # best_row.name is cafaeval's MultiIndex (namespace, threshold, ...).
        # Depends on the cafaeval DataFrame schema — keep the version pin in
        # examples/protein_go/requirements.txt aligned if you upgrade.
        "best_threshold": float(best_row.name[-1]) if hasattr(best_row, "name") else None,
        "precision_at_best": float(best_row["pr"]),
        "recall_at_best": float(best_row["rc"]),
        "coverage": float(best_row["cov_max"]) if "cov_max" in best_row else None,
        "num_proteins": len(predictions_by_protein),
        "num_predictions": sum(len(p) for p in predictions_by_protein.values()),
        "cafaeval_df_path": str(df_path),
    }


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

# ProteInfer train is the BLAST reference corpus (populated into the DIAMOND DB
# by setup.sh), not an evaluation set — no train.jsonl is written.
_SPLITS = ("validation", "test", "price149")


def load_protein_go(split: str = "validation") -> List[Dict[str, Any]]:
    """Load the protein-GO benchmark split built by setup.sh.

    Each example: {id, sequence, go_terms_mfo (list[str])}
    """
    if split not in _SPLITS:
        raise ValueError(f"Unknown split: {split!r}. Use one of {_SPLITS}.")

    path = DATA_DIR / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found: {path}. Run setup.sh first."
        )

    examples = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            examples.append({
                "id": row["accession"],
                "sequence": row["sequence"],
                "go_terms_mfo": row["go_terms_mfo"],
            })
    logger.info(f"GO-MFO {split} set: {len(examples)} examples")
    return examples
