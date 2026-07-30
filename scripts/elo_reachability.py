#!/usr/bin/env python3
"""Replay the Elo-reachability guard against a finished run.

The guard ships off by default, so the question before enabling it is
empirical: on runs you have already paid for, when would it have fired, and
was the agent it would have suppressed actually worthless?

This reads a checkpoint and reports, for every iteration, whether an agent
evolved at that point could still have become the Elo leader — plus what the
run's winner actually turned out to be, so a firing can be checked against
the outcome.

Usage:
    python scripts/elo_reachability.py <run_dir>
    python scripts/elo_reachability.py <run_dir> --at-iteration 12
    python scripts/elo_reachability.py <run_dir> --min-rounds 2

Caveat on the retrospective view: ratings are replayed from `test_history`,
so the figures at iteration N are exactly what the guard would have seen
live. The horizon, however, uses the budget recorded in the checkpoint; a
run that was resumed with a different budget will show that later value.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from RoboPhD.elo_reachability import (  # noqa: E402
    CHALLENGER_ID,
    assess_reachability,
    calculate_elo_updates,
    clone_penalty_totals,
    remaining_rounds,
    strip_clone_penalties,
)


def replay_elos_through(test_history, upto):
    """Ratings after the first `upto` iterations, pre-clone-penalty.

    Mirrors researcher._recalculate_all_elo_scores, which replays from 1500
    rather than carrying rating forward, so a partial replay is the honest
    way to recover what the ladder looked like mid-run.
    """
    elos = {}
    for iteration_data in test_history[:upto]:
        for agent in iteration_data:
            elos.setdefault(agent, 1500.0)
        results = {
            agent: {"average_score": data["average_score"]}
            for agent, data in iteration_data.items()
        }
        sub = {agent: elos[agent] for agent in results}
        elos.update(calculate_elo_updates(sub, results))
    return elos


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("run_dir", help="Run directory containing checkpoint.json")
    ap.add_argument("--at-iteration", type=int, default=None,
                    help="Report a single iteration instead of every one")
    ap.add_argument("--min-rounds", type=int, default=3,
                    help="Rounds remaining above which the guard never fires "
                         "(default: %(default)s)")
    ap.add_argument("--agents-per-iteration", type=int, default=None,
                    help="Round-robin size (default: read from the checkpoint)")
    ap.add_argument("--evaluation-budget", type=int, default=None,
                    help="Override the budget recorded in the checkpoint")
    args = ap.parse_args()

    checkpoint_path = Path(args.run_dir) / "checkpoint.json"
    if not checkpoint_path.exists():
        raise SystemExit(f"No checkpoint.json in {args.run_dir}")
    ckpt = json.loads(checkpoint_path.read_text())

    test_history = ckpt.get("test_history") or []
    if not test_history:
        raise SystemExit("Checkpoint has no test_history to replay")

    fresh_evals = ckpt.get("iteration_fresh_evals") or []
    penalties = clone_penalty_totals(ckpt.get("clone_detections") or [])

    cfg = (ckpt.get("config_manager") or {}).get("resolved_configs") or {}
    def _cfg(key, fallback):
        if cfg:
            latest = cfg[max(cfg, key=lambda k: int(k))]
            return latest.get(key, fallback)
        return fallback

    budget = args.evaluation_budget or _cfg("evaluation_budget", None)
    per_iter = args.agents_per_iteration or _cfg("agents_per_iteration", 3)

    perf = ckpt.get("performance_records") or {}
    winner = max(perf, key=lambda a: perf[a].get("elo", 0)) if perf else "?"

    print(f"Run: {args.run_dir}")
    print(f"Iterations: {len(test_history)}   agents/iteration: {per_iter}   "
          f"budget: {budget}")
    print(f"Final Elo leader: {winner}"
          + (f" ({perf[winner]['elo']:.0f})" if winner in perf else ""))
    if budget is None:
        print("\nNOTE: no evaluation_budget recorded, so the horizon is "
              "unbounded and the guard would never have fired on this run.")

    iterations = ([args.at_iteration] if args.at_iteration
                  else range(2, len(test_history) + 1))

    print(f"\n{'iter':>5}  {'verdict':<12}  detail")
    print("-" * 100)
    fired = []
    for iteration in iterations:
        if not 2 <= iteration <= len(test_history):
            raise SystemExit(
                f"--at-iteration {iteration} outside 2..{len(test_history)}"
            )
        # State as of the START of `iteration`: everything before it.
        elos = replay_elos_through(test_history, iteration - 1)
        if not elos:
            continue
        verdict = assess_reachability(
            strip_clone_penalties(elos, penalties),
            rounds_remaining=remaining_rounds(fresh_evals[:iteration - 1], budget),
            agents_per_iteration=per_iter,
            min_rounds=args.min_rounds,
            clone_penalties=penalties,
        )
        tag = "reachable" if verdict.reachable else "WOULD FIRE"
        print(f"{iteration:>5}  {tag:<12}  {verdict.summary()}")
        if not verdict.reachable:
            fired.append(iteration)

    if not args.at_iteration:
        print("-" * 100)
        if fired:
            # Every unreachable iteration becomes greedy: the guard is sticky
            # once fired, because the verdict only deteriorates from there.
            # This retrospective view double-counts slightly in one respect —
            # replacing evolution with a greedy round changes what the later
            # iterations would have contained, so iterations after the first
            # firing are counterfactual rather than observed.
            print(f"Would have fired at: {fired}  "
                  f"({len(fired)} of {len(test_history) - 1} evolving iterations)")
            print(f"First firing: iteration {fired[0]}. From there the run "
                  f"stays greedy unless the horizon grows (--extend).")
            print(f"\nCheck against the outcome: the winner was {winner}. If it "
                  f"was created at or after iteration {fired[0]}, the guard "
                  f"would have suppressed the run's best agent.")
            suppressed = sorted(
                a for a in perf
                if any(a in test_history[i - 1]
                       for i in fired if i <= len(test_history))
                and a.startswith(tuple(f"iter{i}_" for i in fired))
            )
            if suppressed:
                print(f"Agents that would not have been created: {suppressed}")
        else:
            print("Would never have fired on this run.")


if __name__ == "__main__":
    main()
