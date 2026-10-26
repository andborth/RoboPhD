"""Invalidate the OpenAI-outage samples in the v0_1_0_cap_0_063_fable51 PFB eval
so that resuming the eval-set re-runs them.

Background: on 2026-10-22 at about 10:10:52Z the OpenAI account ran out of quota,
and every call returned 429 insufficient_quota. Both the primary tier and the
seed tier then ran out their wait_for. The wrapper (agent.py) submitted an empty
result and recorded __wrapper_fallback_caught in the sample metadata. Those
samples were judged 0.0 with sample.error=None, and inspect's eval-set retry
reuses every sample that has no error and no invalidation. They have to be
invalidated, or the resume keeps the zeros.

This uses inspect's own invalidation mechanism (inspect_ai.log.invalidate_samples):
  - Retry skips samples with invalidation set (task/run.py eval_log_sample_source).
  - The reason and author are kept in the log as ProvenanceData.
  - Nothing is deleted from the record.

Selection rule: no error, and __wrapper_fallback_caught in the metadata, which
means the wrapper submitted an empty result. Sample ids are not hard-coded. The
script refuses to write unless that rule picks exactly the set of samples that
completed after the outage onset, and every sample that finished before the
onset is clean (no wrapper metadata at all).

Dry run by default. --apply first copies the original .eval to BACKUP_DIR, which
is outside the eval-set log dir so eval-set never sees two logs for the task.
It then writes the invalidated log in place and verifies it by reading it back.
"""
import argparse
import datetime as dt
import shutil
import sys
from pathlib import Path

from inspect_ai.log import (ProvenanceData, invalidate_samples, read_eval_log,
                            write_eval_log)

SUB = Path("/Users/andrew/Desktop/cc/RoboPhD/submissions/asta_paper_finder/"
           "v0_1_0_cap_0_063_fable51")
LOG_DIR = SUB / "logs" / "full_test"
BACKUP_DIR = SUB / "logs" / "backup_pre_invalidate"
ONSET = dt.datetime.fromisoformat("2026-10-22T10:10:52+00:00")
REASON = ("OpenAI account quota exhausted (429 insufficient_quota) from about "
          "2026-10-22T10:10:52Z. Primary and seed tiers both timed out and the "
          "wrapper submitted an empty result, which scored 0.0 with no sample "
          "error. Invalidated so that the eval-set resume re-runs it.")


def _ts(x):
    return dt.datetime.fromisoformat(x) if isinstance(x, str) else x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write (default: dry run)")
    args = ap.parse_args()

    logs = sorted(LOG_DIR.glob("*.eval"))
    if len(logs) != 1:
        print(f"expected exactly one .eval in {LOG_DIR}, found {len(logs)}")
        return 1
    path = logs[0]
    log = read_eval_log(str(path))
    print(f"{path.name}: status={log.status} samples={len(log.samples or [])} "
          f"revision={log.eval.revision.commit if log.eval.revision else None}")
    if log.status not in ("cancelled", "error", "started"):
        print("log is not an incomplete run; refusing")
        return 1

    target, after_onset, dirty_before = [], [], []
    for s in log.samples:
        wrapped = any(k.startswith("__wrapper") for k in (s.metadata or {}))
        if s.error is None and "__wrapper_fallback_caught" in (s.metadata or {}):
            target.append(s)
        if s.error is None and _ts(s.completed_at) > ONSET:
            after_onset.append(s.id)
        if s.error is None and _ts(s.completed_at) <= ONSET and wrapped:
            dirty_before.append(s.id)

    errored = sorted(s.id for s in log.samples if s.error is not None)
    ids = sorted(s.id for s in target)
    print(f"invalidate ({len(ids)}): {ids}")
    for s in target:
        sc = s.scores.get("score_paper_finder").value if s.scores else None
        print(f"  {s.id}: total_time={s.total_time:.0f}s score={sc}")
    print(f"errored, already re-run by retry ({len(errored)}): {errored}")

    if sorted(after_onset) != ids:
        print(f"REFUSING: completed-after-onset set {sorted(after_onset)} != rule set")
        return 1
    if dirty_before:
        print(f"REFUSING: pre-onset samples carry wrapper metadata: {dirty_before}")
        return 1
    if any(s.invalidation is not None for s in log.samples):
        print("REFUSING: log already has invalidated samples")
        return 1
    if not args.apply:
        print("dry run: nothing written (pass --apply)")
        return 0

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup = BACKUP_DIR / path.name
    if backup.exists():
        print(f"REFUSING: backup already exists at {backup}")
        return 1
    shutil.copy2(path, backup)
    print(f"backup: {backup}")

    new = invalidate_samples(
        log, [s.uuid for s in target],
        ProvenanceData(author="Andrew Borthwick", reason=REASON,
                       metadata={"onset_utc": ONSET.isoformat(),
                                 "sample_ids": ids}))
    write_eval_log(new, str(path))

    check = read_eval_log(str(path))
    inv = sorted(s.id for s in check.samples if s.invalidation is not None)
    ok = (inv == ids and len(check.samples) == len(log.samples)
          and check.status == log.status and check.invalidated)
    print(f"verify: status={check.status} samples={len(check.samples)} "
          f"invalidated={inv} log.invalidated={check.invalidated} -> "
          f"{'OK' if ok else 'MISMATCH'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
