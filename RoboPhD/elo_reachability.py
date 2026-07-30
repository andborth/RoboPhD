"""Can an agent evolved in the upcoming iteration still become the Elo leader?

`find_best_agent` selects a run's output by maximum Elo, so an agent that
cannot reach the top of the ladder cannot be the run's result. Late in a
run — when few evaluations remain — a newly evolved agent can be
arithmetically incapable of climbing from its starting 1500 to the
incumbent's rating no matter how well it performs. Evolving it spends an
evolution session and a share of the remaining evaluation budget on an
agent that is dead weight by construction, and it displaces re-testing of
the agents that *can* still win.

This module answers the reachability question so the caller can spend the
tail of a run on `greedy` rounds (deterministic top-k by Elo, no evolution)
instead.

## Single source for the Elo formula

`calculate_elo_updates` lives here and `researcher.py` imports it, rather
than the reverse. A projection that re-implemented the update would drift
from the one that scores real iterations, and the whole verdict rests on
the two agreeing. The dependency points this way because this module has
no framework imports at all, which also keeps it unit-testable without
constructing a researcher.

## What "best case" means, precisely

The projection is optimistic on every axis the caller does not control:

  - the challenger wins every game it plays;
  - it is matched against the highest-rated agents available each round
    (beating a stronger opponent yields more rating and costs that
    opponent more, so this dominates facing weaker agents);
  - the remaining agents' results are chosen to maximize the challenger's
    final lead, searched exhaustively over weak orderings (ties allowed).

Two honest limits on the verdict:

  - **Matchmaking is assumed, not guaranteed.** Real selection is
    priority-based (pending winners, the new agent, untested agents, then
    top-k), so the challenger may never face the leader at all. Assuming
    it does is the optimistic choice, which is the right direction for a
    guard that should only fire when there is clearly no path.
  - **The search is greedy across rounds.** Each round is optimized in
    isolation, and the per-round objective genuinely trades off against
    itself over a horizon: keeping opponents highly rated grows the
    challenger's wins, while pushing them down is what clears its path.
    So `reachable=False` means "no path found under optimistic play", not
    a proof of impossibility. `min_rounds` exists to keep the guard away
    from the regime where that distinction could matter.

## Clone penalties

`researcher._recalculate_all_elo_scores` replays history from 1500 and
*then* subtracts 200 per clone detection, so `performance_records['elo']`
is post-penalty while the values the Elo formula consumes are pre-penalty.
A projection must not mix the two: it runs on pre-penalty ratings (use
`strip_clone_penalties`) and re-applies them before the final comparison,
because leadership is judged on the post-penalty number. Helpers for both
directions are provided so callers cannot get the basis silently wrong.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set

# Elo constants. These are the values researcher.py has always used; they
# live here now so the projection and the real ladder cannot disagree.
K_FACTOR = 32
INITIAL_ELO = 1500.0
CLONE_PENALTY = 200.0

# Stand-in id for the agent that does not exist yet. Deliberately not a
# legal agent name so it can never collide with a real pool entry.
CHALLENGER_ID = "<challenger>"

# Weak-ordering enumeration is a Fubini number in the opponent count
# (2->3, 3->13, 4->75, 5->541, 6->4683). Past this we fall back to a
# single heuristic ordering and say so in the verdict, rather than
# silently spending minutes or silently searching less.
MAX_EXHAUSTIVE_OPPONENTS = 6


def calculate_elo_updates(
    current_elos: Dict[str, float],
    iteration_results: Dict[str, Dict[str, Any]],
    k: int = K_FACTOR,
) -> Dict[str, float]:
    """Round-robin Elo update over one iteration's scores, ties included.

    Every participating agent plays every other exactly once; the winner of
    a pair is whichever scored higher on that iteration, and equal scores
    (to 6dp, collapsing float noise) are draws worth 0.5 each.

    Args:
        current_elos: agent_id -> rating going into the iteration.
        iteration_results: agent_id -> {'average_score': float}.
        k: Elo K-factor.

    Returns:
        agent_id -> rating after the iteration. Input is not mutated.
    """
    updated_elos = current_elos.copy()
    agents = list(iteration_results.keys())

    # Group by score so equal scorers can be paired as draws.
    score_groups: Dict[float, List[str]] = {}
    for agent in agents:
        score = round(iteration_results[agent]['average_score'], 6)
        score_groups.setdefault(score, []).append(agent)

    # Draws within a tied group.
    for group in score_groups.values():
        if len(group) > 1:
            for i, agent1 in enumerate(group):
                for agent2 in group[i + 1:]:
                    elo1 = updated_elos[agent1]
                    elo2 = updated_elos[agent2]
                    expected1 = 1 / (1 + 10 ** ((elo2 - elo1) / 400))
                    expected2 = 1 / (1 + 10 ** ((elo1 - elo2) / 400))
                    updated_elos[agent1] += k * (0.5 - expected1)
                    updated_elos[agent2] += k * (0.5 - expected2)

    # Wins/losses across groups.
    sorted_groups = sorted(score_groups.keys(), reverse=True)
    for i, higher_score in enumerate(sorted_groups[:-1]):
        for lower_score in sorted_groups[i + 1:]:
            for winner in score_groups[higher_score]:
                for loser in score_groups[lower_score]:
                    winner_elo = updated_elos[winner]
                    loser_elo = updated_elos[loser]
                    expected_winner = 1 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
                    expected_loser = 1 / (1 + 10 ** ((winner_elo - loser_elo) / 400))
                    updated_elos[winner] += k * (1 - expected_winner)
                    updated_elos[loser] += k * (0 - expected_loser)

    return updated_elos


# --- rating basis helpers ----------------------------------------------------


def clone_penalty_totals(
    clone_detections: Sequence[Any], penalty: float = CLONE_PENALTY
) -> Dict[str, float]:
    """agent_id -> total penalty currently subtracted from its rating.

    `clone_detections` entries are (clone_id, matched_id, iteration) as
    stored on the researcher; only the first element is read. An agent
    detected twice carries twice the penalty, matching the replay, which
    subtracts once per detection.
    """
    totals: Dict[str, float] = {}
    for detection in clone_detections:
        clone_id = detection[0] if isinstance(detection, (tuple, list)) else detection
        totals[clone_id] = totals.get(clone_id, 0.0) + penalty
    return totals


def strip_clone_penalties(
    elos: Dict[str, float], penalties: Dict[str, float]
) -> Dict[str, float]:
    """Recover pre-penalty ratings — the basis the Elo formula operates on."""
    return {a: elo + penalties.get(a, 0.0) for a, elo in elos.items()}


def apply_clone_penalties(
    elos: Dict[str, float], penalties: Dict[str, float]
) -> Dict[str, float]:
    """Re-impose penalties — the basis leadership is judged on."""
    return {a: elo - penalties.get(a, 0.0) for a, elo in elos.items()}


# --- remaining horizon ------------------------------------------------------


def remaining_rounds(
    iteration_fresh_evals: Sequence[int],
    evaluation_budget: Optional[int],
    window: int = 5,
) -> float:
    """How many more iterations the evaluation budget affords.

    Uses a trailing mean of per-iteration fresh-eval counts rather than a
    nominal figure, because the (agent, example) cache makes later
    iterations progressively cheaper — dividing the budget by a nominal
    per-iteration cost badly underestimates the horizon.

    Returns `inf` when the answer is unknown or unbounded (no budget set,
    no history yet, or a trailing window that spent nothing). `inf` reads
    as "plenty of room" downstream, so an absent signal never triggers an
    intervention.
    """
    if evaluation_budget is None:
        return math.inf
    if not iteration_fresh_evals:
        return math.inf
    recent = list(iteration_fresh_evals[-window:])
    avg = sum(recent) / len(recent)
    if avg <= 0:
        return math.inf
    spent = sum(iteration_fresh_evals)
    return max(0.0, (evaluation_budget - spent) / avg)


def iterations_remaining(
    current_iteration: int, num_iterations: Optional[int]
) -> float:
    """Iterations the loop will still run, counting the current one.

    The second terminator, independent of the budget: researcher's loop is
    `while iteration <= self.num_iterations`. Exact rather than estimated —
    no trailing mean needed — because it is a plain iteration count.

    `inf` when no cap is configured. Note that every shipped example defaults
    `--num-iterations` to 999 precisely so the budget is the real limit, so
    in normal use this returns a large number and the budget binds; it is the
    binding constraint only when a run is deliberately capped by iteration.
    """
    if num_iterations is None:
        return math.inf
    return max(0, num_iterations - current_iteration + 1)


def horizon(
    iteration_fresh_evals: Sequence[int],
    evaluation_budget: Optional[int],
    *,
    current_iteration: int,
    num_iterations: Optional[int],
    window: int = 5,
) -> tuple[float, str]:
    """Rounds remaining under whichever terminator binds first.

    A run can end two ways and either may come first, so the horizon is the
    minimum. Considering only the budget overestimates the horizon of an
    iteration-capped run (firing late or never); considering only the
    iteration cap overestimates a budget-bound one.

    Returns (rounds, which) where `which` names the binding terminator —
    "budget", "iterations", or "none". Reported in the verdict because the
    remedy differs entirely: raise --evaluation-budget, or --extend.
    """
    budget_rounds = remaining_rounds(iteration_fresh_evals, evaluation_budget, window)
    iteration_rounds = iterations_remaining(current_iteration, num_iterations)
    if budget_rounds <= iteration_rounds:
        which = "none" if math.isinf(budget_rounds) else "budget"
        return budget_rounds, which
    which = "none" if math.isinf(iteration_rounds) else "iterations"
    return iteration_rounds, which


def rounds_playable(rounds_remaining: float) -> int:
    """Whole iterations a challenger created next round will actually play.

    Ceiling, not floor, and the difference matters at exactly the horizon
    where the guard fires. The evaluation-budget check runs *after* an
    iteration completes (researcher.py, "Check evaluation budget"), so a
    partial round's worth of budget still buys a whole iteration — one that
    overshoots and then ends the run. With 0.7 rounds of budget left the
    upcoming iteration runs in full, and the agent it creates plays that
    round-robin; flooring would model it as never playing a game and call it
    unreachable on a technicality.

    Zero only when the budget is already spent, in which case the run ends
    before the upcoming iteration and there is no agent to reason about.
    """
    if math.isinf(rounds_remaining):
        return 0  # callers short-circuit on an unbounded horizon
    return max(0, math.ceil(rounds_remaining))


# --- best-case projection ---------------------------------------------------


def _weak_orderings(items: Sequence[str]) -> Iterator[List[Set[str]]]:
    """Every ranking of `items` that allows ties, as a list of tiers.

    Ties are enumerated rather than assumed away because a draw is not
    dominated: it moves the higher-rated of two opponents down and the
    lower one up, which can serve the challenger better than either strict
    ordering.
    """
    items = list(items)
    if not items:
        yield []
        return
    for size in range(1, len(items) + 1):
        for combo in itertools.combinations(items, size):
            rest = [i for i in items if i not in combo]
            for tail in _weak_orderings(rest):
                yield [set(combo)] + tail


def _orderings_for(opponents: Sequence[str]) -> tuple[Iterable[List[Set[str]]], bool]:
    """Orderings to search, plus whether the search is exhaustive."""
    if len(opponents) <= MAX_EXHAUSTIVE_OPPONENTS:
        return list(_weak_orderings(opponents)), True
    # Heuristic: weakest wins, so the highest-rated agents shed the most
    # rating and the challenger's path clears fastest.
    ascending = sorted(opponents)
    return [[{a} for a in ascending]], False


def best_case_projection(
    current_elos: Dict[str, float],
    *,
    rounds: int,
    agents_per_iteration: int,
    k: int = K_FACTOR,
    challenger_id: str = CHALLENGER_ID,
) -> tuple[Dict[str, float], bool]:
    """Project ratings forward under play maximally favorable to a new agent.

    The challenger enters at INITIAL_ELO and wins every game. Each round it
    faces the top `agents_per_iteration - 1` agents by current projected
    rating, and the others' results are chosen to maximize the challenger's
    lead. See the module docstring for what this does and does not prove.

    Appending projected iterations this way is exact, not an approximation:
    `_recalculate_all_elo_scores` replays history through this same update,
    so iterating it forward from current ratings yields what the replay
    would produce for those additional iterations.

    Returns:
        (projected ratings including the challenger, search_was_exhaustive)
    """
    elos = dict(current_elos)
    elos[challenger_id] = INITIAL_ELO
    n_opponents = max(1, agents_per_iteration - 1)
    exhaustive = True

    for _ in range(max(0, int(rounds))):   # rounds is pre-computed by rounds_playable
        opponents = sorted(
            (a for a in elos if a != challenger_id),
            key=lambda a: elos[a],
            reverse=True,
        )[:n_opponents]
        if not opponents:
            break

        orderings, round_exhaustive = _orderings_for(opponents)
        exhaustive = exhaustive and round_exhaustive

        best_lead = None
        best_update = None
        for ordering in orderings:
            # Encode the ranking as scores: the challenger strictly above
            # every tier, then one score per tier, descending.
            results = {challenger_id: {'average_score': float(len(ordering) + 1)}}
            for tier_idx, tier in enumerate(ordering):
                tier_score = float(len(ordering) - tier_idx)
                for agent in tier:
                    results[agent] = {'average_score': tier_score}

            sub_elos = {a: elos[a] for a in results}
            updated = calculate_elo_updates(sub_elos, results, k)
            lead = updated[challenger_id] - max(
                v for a, v in updated.items() if a != challenger_id
            )
            if best_lead is None or lead > best_lead:
                best_lead = lead
                best_update = updated

        elos.update(best_update or {})

    return elos, exhaustive


# --- verdict ----------------------------------------------------------------


@dataclass(frozen=True)
class ReachabilityVerdict:
    """Whether a not-yet-evolved agent could still top the ladder."""

    reachable: bool
    reason: str
    rounds_remaining: float
    leader_id: Optional[str] = None
    leader_elo: Optional[float] = None
    projected_challenger_elo: Optional[float] = None
    projected_best_rival_elo: Optional[float] = None
    projection_ran: bool = False
    search_exhaustive: bool = True
    # Which terminator set the horizon: "budget", "iterations", or "none".
    # Named in the summary because the remedy differs -- raise
    # --evaluation-budget, or --extend.
    binding_constraint: str = "none"

    def summary(self) -> str:
        """One-line, log-ready account of the decision."""
        head = "reachable" if self.reachable else "UNREACHABLE"
        rounds = ("unbounded" if math.isinf(self.rounds_remaining)
                  else f"{self.rounds_remaining:.1f}")
        bound = ("" if self.binding_constraint == "none"
                 else f", limited by {self.binding_constraint}")
        line = f"{head}: {self.reason} (rounds remaining: {rounds}{bound})"
        if self.projection_ran:
            line += (
                f" | best-case challenger {self.projected_challenger_elo:.0f}"
                f" vs best rival {self.projected_best_rival_elo:.0f}"
            )
            if not self.search_exhaustive:
                line += " | search truncated (heuristic ordering)"
        return line


def assess_reachability(
    current_elos: Dict[str, float],
    *,
    rounds_remaining: float,
    agents_per_iteration: int,
    min_rounds: int = 3,
    clone_penalties: Optional[Dict[str, float]] = None,
    binding_constraint: str = "none",
    k: int = K_FACTOR,
) -> ReachabilityVerdict:
    """Decide whether an agent evolved next iteration could lead on Elo.

    Args:
        current_elos: PRE-penalty ratings (see `strip_clone_penalties`).
        rounds_remaining: from `remaining_rounds`; `inf` means unbounded.
        agents_per_iteration: round-robin size, hence games per round.
        min_rounds: above this many remaining rounds, return reachable
            without projecting. A cheap early-out, and it keeps the guard
            out of the long-horizon regime where the greedy cross-round
            search is least trustworthy.
        clone_penalties: agent_id -> penalty, re-applied before comparing
            because leadership is judged post-penalty.

    Returns:
        A verdict carrying the numbers behind it, for logging.
    """
    penalties = clone_penalties or {}

    if not current_elos:
        return ReachabilityVerdict(
            reachable=True,
            reason="no rated agents yet, so nothing to overtake",
            rounds_remaining=rounds_remaining,
            binding_constraint=binding_constraint,
        )

    judged = apply_clone_penalties(current_elos, penalties)
    leader_id = max(judged, key=lambda a: judged[a])

    if rounds_remaining > min_rounds:
        return ReachabilityVerdict(
            reachable=True,
            reason=f"more than {min_rounds} rounds remain",
            rounds_remaining=rounds_remaining,
            leader_id=leader_id,
            leader_elo=judged[leader_id],
            binding_constraint=binding_constraint,
        )

    projected, exhaustive = best_case_projection(
        current_elos,
        rounds=rounds_playable(rounds_remaining),
        agents_per_iteration=agents_per_iteration,
        k=k,
    )
    judged_projection = apply_clone_penalties(projected, penalties)
    challenger = judged_projection[CHALLENGER_ID]
    rivals = {a: v for a, v in judged_projection.items() if a != CHALLENGER_ID}
    best_rival_id = max(rivals, key=lambda a: rivals[a])
    best_rival = rivals[best_rival_id]
    reachable = challenger > best_rival

    reason = (
        "best-case play overtakes the field"
        if reachable
        else f"even winning every game it cannot pass {best_rival_id}"
    )
    return ReachabilityVerdict(
        reachable=reachable,
        reason=reason,
        rounds_remaining=rounds_remaining,
        leader_id=leader_id,
        leader_elo=judged[leader_id],
        projected_challenger_elo=challenger,
        projected_best_rival_elo=best_rival,
        projection_ran=True,
        search_exhaustive=exhaustive,
        binding_constraint=binding_constraint,
    )
