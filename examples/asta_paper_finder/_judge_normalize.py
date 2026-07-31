"""Judge-output normalization shared by the training-judge path and the
calibration study.

Astabench's scorer parses each relevance verdict with a strict JSON
extractor + pydantic model; ANY format deviance drops the whole document
as Not Relevant — silent score noise attributed to the agent. GPT-4o
deviates essentially never, so the stock path is safe as-is; alternate
training judges deviate rarely but nonzero (gpt-5.6-luna: 2/300 in the
2026-07-20 calibration study; gpt-5.4-mini: ~5% in the 2026-07-17 one).
This module repairs the near-JSON those models emit and normalizes
rescuable shape deviations, isolating *format* non-compliance from
genuine judgement differences.

Two consumers, one implementation (so measurement and production cannot
drift):
  - `_check_judge_calibration.py` installs it for BOTH judges when
    measuring agreement (format noise must not masquerade as verdict
    disagreement);
  - `evaluator._apply_training_grader` installs it ONLY when the
    training-judge override is active — stock GPT-4o evals (training
    default, all test evals, official submissions) keep astabench's
    strict parser untouched for official parity.

Repair counters are process-global; call `reset()` at eval start and
`last_repairs()` after scoring (mirrors `_grounding`'s record pattern)
to surface per-eval counts in diagnostics.
"""

from __future__ import annotations

import json
import re

# Counts since the last reset(). "strict_ok" = parsed with no help;
# "recovered" = JSON repaired; "shape_fixed" = parsed but pydantic-shape
# normalized; "unrecoverable" = dropped (astabench then scores the doc
# Not Relevant, same as stock).
_REPAIR = {"strict_ok": 0, "recovered": 0, "shape_fixed": 0, "unrecoverable": 0}


def reset() -> None:
    """Zero the repair counters. Call at the start of each evaluation."""
    for k in _REPAIR:
        _REPAIR[k] = 0


def last_repairs() -> dict:
    """Counters accumulated since the last reset()."""
    return dict(_REPAIR)


def _normalize_judgement_shape(obj):
    """Rescue structurally-sound judgements that astabench's pydantic model
    would reject on shape alone. Measured on gpt-5.4-mini (150-doc study,
    2026-07-17): every single dropped doc was one of these mechanical
    quirks — the verdicts themselves were present and well-formed.

      - `relevance_summary` nested INSIDE the `criteria` dict (9/15) →
        relocate to top level (pydantic tries to validate the string as a
        criterion judgement and rejects the whole doc);
      - `criteria` as a list of {name, ...} objects instead of a dict (2/15);
      - a criterion missing the `relevant_snippet` key (required-but-
        nullable in the schema) → fill None.

    Returns the (possibly modified) dict and whether anything changed."""
    if not isinstance(obj, dict):
        return obj, False
    changed = False
    crit = obj.get("criteria")
    if isinstance(crit, list):
        obj["criteria"] = {
            c.get("name", f"criterion_{i}"): {k: v for k, v in c.items() if k != "name"}
            for i, c in enumerate(crit) if isinstance(c, dict)
        }
        crit = obj["criteria"]
        changed = True
    if isinstance(crit, dict):
        for key in [k for k, v in crit.items() if not isinstance(v, dict)]:
            stray = crit.pop(key)
            if key not in obj:
                obj[key] = stray
            changed = True
        for v in crit.values():
            if isinstance(v, dict) and "relevant_snippet" not in v:
                v["relevant_snippet"] = None
                changed = True
    if "relevance_summary" not in obj:
        obj["relevance_summary"] = None
        changed = True
    return obj, changed


def _lenient_extract_json(response: str):
    """Drop-in for astabench's extract_json_from_response that repairs the
    near-JSON some models emit, then normalizes rescuable shape deviations
    (see _normalize_judgement_shape). Strict output is untouched.

    Decode ladder: strict → comma/dangling-quote regexes → dangling-key
    null-fill → closing-brace balancing. All observed gpt-5.4-mini decode
    failures were a missing final `}` (usually alongside the misplaced
    relevance_summary) or a `"key"}` with no value."""
    s = response[response.find("{"): response.rfind("}") + 1] if "{" in response else ""
    if not s:
        _REPAIR["unrecoverable"] += 1
        return None
    decoded = None
    try:
        decoded = json.loads(s)
        _REPAIR["strict_ok"] += 1
    except json.JSONDecodeError:
        base = re.sub(r",(\s*[}\]])", r"\1", s)              # trailing commas
        base = re.sub(r',\s*"\s*([}\]])', r"\1", base)       # dangling ,"  before } ]
        # "key"} with no value — anchor on a preceding { or , so string
        # VALUES ending in "} (the normal case) never match.
        keyfill = re.sub(r'([{,]\s*"[^"]+")\s*}', r"\1: null}", base)
        # `},{"` mid-dict: the model closed and reopened the object between
        # entries. Merging is only ever TRIED — variants parse in order and
        # the first success wins, so valid list-of-objects JSON (which the
        # earlier variants already parse) is never touched by this one.
        merged = re.sub(r'}\s*,\s*{\s*(")', r",\1", keyfill)
        variants = [base, keyfill, merged]
        for variant in variants[:]:
            variants.extend(variant + "}" * n for n in (1, 2))  # missing closer(s)
        for candidate in variants:
            try:
                decoded = json.loads(candidate)
                _REPAIR["recovered"] += 1
                break
            except json.JSONDecodeError:
                continue
    if decoded is None:
        _REPAIR["unrecoverable"] += 1
        return None
    decoded, shape_changed = _normalize_judgement_shape(decoded)
    if shape_changed:
        _REPAIR["shape_fixed"] += 1
    return decoded


def install() -> None:
    """Patch the extractor the judge actually calls.

    relevance.py binds extract_json_from_response by from-import, so the
    module attribute on `relevance` itself is the binding that must be
    replaced. Idempotent."""
    from astabench.evals.paper_finder import relevance as rel
    rel.extract_json_from_response = _lenient_extract_json


# Labels-only judge prompt: identical judging instructions, but drops the
# mandated per-criterion `relevant_snippet` and the `relevance_summary` —
# prose the scorer never reads (only the relevance labels enter the
# metric). Validated for gpt-5.6-luna on 2026-07-23 (3-arm study, 148
# stored docs, same-prompt rerun as noise floor): labels-only sat AT the
# floor on every measure (agreement 0.811 vs floor 0.838; Perfect drift
# +9% vs the floor's own +6%) while cutting output tokens 65% —
# $0.0022/verdict, ~5.7x cheaper than the stock GPT-4o basis -- at the
# then-current $1.00/$6.00 luna rates; the 2026-07-31 repricing to
# $0.20/$1.20 makes it ~$0.00044/verdict, ~28x, leaving the 65% token
# reduction and the agreement findings unchanged. UNSAFE for
# gpt-4o (+18.5% Perfect inflation vs a stable 54->54 floor: the
# snippet-writing is chain-of-thought for the older model) — mispairing is
# impossible by construction in main.py, where the profile is DERIVED from
# the judge (_prompt_for_judge) rather than chosen, and refused outright in
# _apply_training_grader (a prompt profile without a judge override is a
# hard error). rejudge_test.py keeps an explicit --judge-prompt because it
# re-judges stored submissions on arbitrary bases; it validates the same
# pairing itself. The shape normalizer backfills the pydantic-required
# snippet/summary fields as nulls, so astabench's parsing is untouched.
NO_PROSE_JUDGE_TEMPLATE = """
Judge how relevant the following paper is to each of the provided criteria. For each criterion, consider its entire description when making your judgement.

For each criterion, provide one output:
- `relevance` (str): one of "Perfectly Relevant", "Somewhat Relevant", "Not Relevant".

Output a JSON:
- top-level key `criteria`. Under it, for every criterion name (exactly as given in the provided criteria), there should be an object containing one field: `relevance`.

Criteria:
```
{criteria}
```"""


def install_no_prose_prompt() -> None:
    """Patch the judge prompt template to the labels-only variant.

    relevance.py reads the template as a module global at call time, so
    reassigning the attribute is enough. Idempotent. Callers must pair
    this with a judge-basis-distinct verdict-cache namespace (a prompt
    variant is a different verdict basis; verdicts from different prompts
    must never share a cache file)."""
    from astabench.evals.paper_finder import relevance as rel
    rel.relevance_criteria_judgement_prompt_with_relevant_snippets_after = (
        NO_PROSE_JUDGE_TEMPLATE
    )


def install_prompt_reorder() -> None:
    """EXPERIMENTAL — measured and NOT adopted (2026-07-23); kept as the
    reproducible artifact behind upstream/judge-scoring-cost-report.md.

    Reorders the judge-prompt payload so the stable part leads — intended
    to enable OpenAI prompt caching across a query's per-paper judge
    calls. Measurement killed it on both axes: cache reads only improved
    1.0% -> 1.8% (astabench fires a query's judge calls concurrently, so
    no request finishes early enough to write the prefix cache for its
    siblings — a warm-first-call serialization would also be needed), and
    the reorder is NOT metric-neutral (same 113 docs, same judge: 76.1%
    exact agreement, Perfect count 35 -> 42). Do not wire into
    _apply_training_grader without a fresh calibration pass.

    Astabench's judge prompt embeds the RelevanceJudgementInput dict repr
    with `document` constructed FIRST: the per-paper text starts diverging
    right after ~450 tokens of shared instructions, under OpenAI's
    1024-token cache-eligibility floor — measured 0.007% cache hits across
    46.2M judge input tokens on the v0_0_7 official run. Building the dict
    criteria-first makes [instructions + criteria] the shared prefix for
    all ~250 calls of a query. Content is identical; only dict-repr key
    order changes. Alternate-judge (training/internal) paths only — stock
    evals stay byte-identical to official scoring. Verdict stability under
    the reorder is gated by the same calibration discipline as the judge
    switch itself."""
    from astabench.evals.paper_finder import relevance as rel

    def _format_criteria_first(documents, criteria_to_use):
        import json as _json
        formatted_criteria = _json.dumps(
            [{"name": r.name, "description": r.description} for r in criteria_to_use],
            indent=2,
        )
        return [
            rel.RelevanceJudgementInput(
                criteria=formatted_criteria,
                doc_id=doc.corpus_id,
                document=doc.markdown,
            )
            for doc in documents
        ]

    rel._format_documents_for_relevance_judgement = _format_criteria_first
