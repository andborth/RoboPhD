"""Loader for DiscoveryBench's synth split.

AstaBench's `astabench/evals/discoverybench/task.py:load_discoverybench_hf`
covers only the `real/` split. The `synth/` split lives in the public
GitHub repo `allenai/discoverybench` and is not mirrored on HuggingFace
in the same flat-table form. This module fills the gap:

  - On first call: shallow-clone the public repo into a local cache.
  - For each `synth/{split}/<folder>/metadata_*.json`, build an Inspect
    `Sample` with `files` pointing at the cached `data.csv`.
  - Normalize the column-metadata path so synth's flat
    `columns: [{name, description}, ...]` shape becomes the real-shaped
    `columns: {raw: [{name, description}, ...]}` that the AstaBench
    scorer's `prepare_dataset_metadata_json` expects.

Each metadata file holds one query → one Sample (not nested-and-flattened
the way real is; synth has 1 question per metadata file).
"""

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Literal

from inspect_ai.dataset import Sample

logger = logging.getLogger(__name__)

DISCOVERYBENCH_REPO_URL = "https://github.com/allenai/discoverybench.git"

CACHE_ROOT = Path(os.environ.get(
    "ROBOPHD_DISCOVERYBENCH_CACHE",
    Path.home() / ".cache" / "robophd" / "discoverybench_synth",
))

TASK_INSTRUCTIONS_SUFFIX = (
    "\n\nMake sure you load the dataset to analyze it "
    "(or defer to an agent that can)."
)


def _ensure_repo_cached() -> Path:
    """Shallow-clone the discoverybench repo into the cache on first use.

    Returns the path to the local `discoverybench/synth/` directory.
    """
    repo_dir = CACHE_ROOT / "repo"
    synth_dir = repo_dir / "discoverybench" / "synth"

    if synth_dir.is_dir():
        return synth_dir

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    if repo_dir.exists():
        shutil.rmtree(repo_dir)

    logger.info(f"Cloning {DISCOVERYBENCH_REPO_URL} into {repo_dir}")
    subprocess.run(
        ["git", "clone", "--depth", "1", DISCOVERYBENCH_REPO_URL, str(repo_dir)],
        check=True,
    )
    if not synth_dir.is_dir():
        raise RuntimeError(
            f"Cloned discoverybench repo but {synth_dir} is missing — "
            f"upstream layout may have changed"
        )
    return synth_dir


def _normalize_columns(cols):
    """Synth has flat `columns: [...]`; real (and the scorer) want `{raw: [...]}`."""
    if isinstance(cols, dict) and "raw" in cols:
        return cols
    return {"raw": [{"name": c["name"], "description": c.get("description", "")} for c in cols]}


def _format_query(question: str, datasets: list) -> str:
    """Build the input string in the same shape AstaBench uses for real."""
    lines = [f"Query: {question}", "", "Datasets:"]
    for d in datasets:
        cols = d["columns"]["raw"] if isinstance(d["columns"], dict) else d["columns"]
        lines.append(f"- {d['name']}: {d['description']}")
        lines.append("  columns:")
        for c in cols:
            lines.append(f"    - {c['name']}: {c.get('description', '')}")
    return "\n".join(lines)


def _build_sample(folder: Path, metadata_idx: int) -> Sample | None:
    """One metadata_*.json → one Sample. Returns None on parse failure."""
    metadata_path = folder / f"metadata_{metadata_idx}.json"
    csv_path = folder / "data.csv"
    if not metadata_path.is_file() or not csv_path.is_file():
        return None

    # Some synth metadata files contain non-UTF-8 bytes (smart quotes from
    # an LLM-driven generation step). Decode permissively rather than fail.
    raw_bytes = metadata_path.read_bytes()
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = raw_bytes.decode("utf-8", errors="replace")
    raw = json.loads(text)
    queries = raw.get("queries", [])
    if not queries:
        return None
    q = queries[0]  # synth packs one question per metadata file

    # synth/test is the upstream held-out competition split — it omits
    # `true_hypothesis`, so we can't score against it. Skip those.
    if "true_hypothesis" not in q:
        return None

    datasets_normalized = [
        {
            "name": d["name"],
            "description": d["description"],
            "columns": _normalize_columns(d["columns"]),
        }
        for d in raw["datasets"]
    ]

    sample_id = f"{folder.name}__m{metadata_idx}__q{q.get('qid', 0)}"
    sample_path_in_sandbox = f"{folder.name}/data.csv"

    return Sample(
        id=sample_id,
        input=_format_query(q["question"], datasets_normalized) + TASK_INSTRUCTIONS_SUFFIX,
        target=[str(q["true_hypothesis"]), ""],  # synth has no gold workflow
        metadata={
            "query": q["question"],
            "metadata": {"datasets": datasets_normalized},
            "split": "synth",
            "domain": raw.get("domain"),
            "question_type": q.get("question_type"),
            "difficulty": q.get("difficulty"),
        },
        files={sample_path_in_sandbox: str(csv_path.resolve())},
    )


def load_synth(split: Literal["train", "dev", "test"]) -> list[Sample]:
    """Return all *scoreable* DiscoveryBench synth samples in the named split.

    Counts (verified against the public repo):
      train: 550 scoreable samples
      dev:   153 scoreable samples
      test:    0 scoreable samples — synth/test is the upstream held-out
               competition set with `true_hypothesis` removed; passing
               "test" here raises ValueError. Use synth/dev for held-out
               evaluation locally, or real/test for the leaderboard.
    """
    if split not in ("train", "dev", "test"):
        raise ValueError(f"split must be train/dev/test, got {split!r}")
    if split == "test":
        raise ValueError(
            "synth/test has no scoreable samples — upstream withholds "
            "`true_hypothesis` so it can't be judged locally. Use "
            "synth/dev (153 scoreable) for held-out synth evaluation, or "
            "real/test (239) via load_real('test') for the leaderboard "
            "metric."
        )

    synth_root = _ensure_repo_cached()
    split_dir = synth_root / split
    samples: list[Sample] = []
    for folder in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        for meta_file in sorted(folder.glob("metadata_*.json")):
            try:
                idx = int(meta_file.stem.removeprefix("metadata_"))
            except ValueError:
                continue
            s = _build_sample(folder, idx)
            if s is not None:
                samples.append(s)
    return samples
