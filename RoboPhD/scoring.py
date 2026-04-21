"""Scoring helpers shared across evaluators and strategy tools.

Currently exposes ``fmax_with_ancestor_closure``: the canonical
max-F1-over-thresholds computation with hierarchical ancestor propagation
applied to both prediction and ground-truth sets. Used by the protein_go
evaluator at evaluation time, and by the protein_go ``score()`` agent tool
so evolution instances can simulate scoring locally against hypothesized
ground truths.

This module is intentionally task-agnostic: the ancestor closure is supplied
by the caller (as a dict or callable), so any benchmark with a DAG-structured
label space can reuse it.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Set, Tuple, Union


AncestorSource = Union[
    Mapping[str, Iterable[str]],
    Callable[[str], Iterable[str]],
]


def _resolve_ancestors(term: str, ancestors: AncestorSource) -> Iterable[str]:
    if callable(ancestors):
        return ancestors(term) or ()
    return ancestors.get(term, ())


def _close(terms: Iterable[str], ancestors: AncestorSource) -> Set[str]:
    """Return ``terms ∪ {ancestors of each term}``."""
    result: Set[str] = set()
    for t in terms:
        result.add(t)
        result.update(_resolve_ancestors(t, ancestors))
    return result


def fmax_with_ancestor_closure(
    predictions: Mapping[str, float],
    ground_truth: Iterable[str],
    ancestors: AncestorSource,
    thresholds: Sequence[float] | None = None,
) -> Dict[str, Any]:
    """Compute max-F1-over-thresholds with ancestor closure on both sides.

    Matches the scoring semantics of the protein_go evaluator: for each
    threshold tau, the prediction set is the terms with confidence >= tau
    expanded with their ancestors, and the ground-truth set is similarly
    expanded. F1 is computed against these closed sets; the max across
    thresholds is reported.

    Args:
        predictions: ``{term_id: confidence_in_[0,1]}``.
        ground_truth: iterable of ground-truth term IDs (pre-closure).
        ancestors: either a dict ``{term_id: [ancestor_ids]}`` or a callable
            ``term_id -> iterable[ancestor_ids]``. Ancestors should *not*
            include the term itself; this function adds the term before
            expanding.
        thresholds: ordered thresholds to sweep. Defaults to
            ``i / 100 for i in range(1, 100)`` (i.e. 0.01..0.99).

    Returns:
        A dict with the following keys:
          - ``fmax``: maximum F1 across thresholds (0.0 if no positive set).
          - ``best_tau``: threshold at which fmax was achieved.
          - ``precision_at_best``, ``recall_at_best``: P/R at ``best_tau``.
          - ``TP``, ``FP``, ``FN``: confusion counts at ``best_tau``.
          - ``pred_set_at_best``: sorted list of closed prediction terms at best tau.
          - ``gt_set``: sorted list of closed ground-truth terms.

        If ``predictions`` is empty or ``ground_truth`` has zero closure, the
        return is a zero-filled dict with empty sets and best_tau=0.0.
    """
    gt_set = _close(ground_truth, ancestors)

    base: Dict[str, Any] = {
        "fmax": 0.0,
        "best_tau": 0.0,
        "precision_at_best": 0.0,
        "recall_at_best": 0.0,
        "TP": 0,
        "FP": 0,
        "FN": len(gt_set),
        "pred_set_at_best": [],
        "gt_set": sorted(gt_set),
    }
    if not predictions or not gt_set:
        return base

    if thresholds is None:
        thresholds = [i / 100 for i in range(1, 100)]

    best_f1 = 0.0
    best: Tuple[float, float, float, int, int, int, List[str]] = (
        0.0, 0.0, 0.0, 0, 0, len(gt_set), []
    )

    for tau in thresholds:
        above = {term for term, score in predictions.items() if score >= tau}
        if not above:
            continue
        pred_closed = _close(above, ancestors)
        tp = len(pred_closed & gt_set)
        if tp == 0:
            continue
        fp = len(pred_closed) - tp
        fn = len(gt_set) - tp
        precision = tp / len(pred_closed)
        recall = tp / len(gt_set)
        f1 = 2 * precision * recall / (precision + recall)
        if f1 > best_f1:
            best_f1 = f1
            best = (
                float(tau), precision, recall, tp, fp, fn,
                sorted(pred_closed),
            )

    tau, precision, recall, tp, fp, fn, pred_closed_sorted = best
    return {
        "fmax": best_f1,
        "best_tau": tau,
        "precision_at_best": precision,
        "recall_at_best": recall,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "pred_set_at_best": pred_closed_sorted,
        "gt_set": sorted(gt_set),
    }


__all__ = ["fmax_with_ancestor_closure"]
