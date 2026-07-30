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

## The question is existential, and the search reflects that

"Could a new agent end up leader" asks whether ANY line of play gets it
there. One winning path is enough and which path does not matter, so
`search_reachable` is a depth-first search that returns the moment the
challenger leads rather than computing a best case and testing it at the end.

That matters for trust. An earlier formulation optimised each round in
isolation, but the per-round objective fights itself over a horizon --
keeping opponents highly rated earns the challenger more, while pushing them
down is what clears its path -- so a locally-greedy line could miss a winning
one and report a false "unreachable". Searching for existence closes that gap:
within the assumptions below, `reachable=False` is a proof, and the verdict
says so (`search_exhaustive`) when the node budget cut the search short.

Cost is asymmetric in the useful direction: a reachable field exits on the
first winning path, and only the rare unreachable field pays for the tree.

Four gates run before any of that, in this order: the guard must be enabled,
the iteration must be past the first, the run must have `min_history`
completed iterations (see TRAILING_WINDOW), and the horizon must be finite.
The history check deliberately precedes the horizon: with a short history the
horizon is the least trustworthy number available, so it should not be what
decides anything -- not even which reason gets reported.

Two assumptions remain, both stated in the verdict's own terms:

  - **The challenger wins every game it plays.** By construction — this is
    the best case for it.
  - **Matchmaking is assumed, not guaranteed.** Real selection is
    priority-based (pending winners, the new agent, untested agents, then
    top-k), so the challenger may never face the leader at all. Assuming it
    does is the optimistic choice, which is the right direction for a guard
    that should only fire when there is no path. Facing the top-rated agents
    is also taken as dominant rather than searched: it both earns more and
    costs the agents that threaten the challenger.

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
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

# Elo constants. These are the values researcher.py has always used; they
# live here now so the projection and the real ladder cannot disagree.
K_FACTOR = 32
INITIAL_ELO = 1500.0
CLONE_PENALTY = 200.0

# Stand-in id for the agent that does not exist yet. Deliberately not a
# legal agent name so it can never collide with a real pool entry.
CHALLENGER_ID = "<challenger>"

# Trailing window for the per-iteration eval-cost mean, and — because the two
# want the same number for related reasons — the minimum history the guard
# requires before it will fire at all.
#
# Iteration 1 runs no evolution (no new agent, no Deep Focus evals), so it
# costs about half a steady-state iteration and drags the mean down with it.
# Measured on the archived runs, fresh evals per iteration:
#
#   v0_0_8:  [14, 26, 33, 38, 35, ...]  trailing mean N=1:14  N=3:24  N=5:29
#   v0_0_1:  [20, 34, 52, 48, 51, ...]  trailing mean N=1:20  N=3:35  N=5:41
#
# So a one- or two-iteration mean understates cost roughly 2x, which overstates
# the horizon roughly 2x. That error is optimistic — it delays firing — so it
# is unreliable rather than unsafe, but it is large, and it converges right
# around N=5. Requiring a full window before firing removes it for free.
#
# The same floor independently protects short runs. A smoke test
# (--num-iterations 3 or 5, often extended afterwards) has no tail worth
# saving: the guard would fire on its FINAL iteration, silently converting the
# last evolution round into a greedy one, so a run whose whole purpose was to
# exercise evolution would quietly exercise less of it. A floor of 5 blocks
# firing at iterations 2-5, covering any run capped at 5 or fewer; larger
# smoke tests still fire on their last iteration and rely on --extend
# restoring the displaced strategy.
TRAILING_WINDOW = 5

# Weak-ordering enumeration is a Fubini number in the opponent count
# (2->3, 3->13, 4->75, 5->541, 6->4683). Past this we fall back to a
# single heuristic ordering and say so in the verdict, rather than
# silently spending minutes or silently searching less.
MAX_EXHAUSTIVE_OPPONENTS = 6

# Nodes the existence search will expand before giving up and reporting an
# incomplete verdict. Only an UNREACHABLE field can approach this: a reachable
# one exits on the first winning path found. At the default
# agents_per_iteration=3 the whole tree is 3^rounds -- 243 at five rounds --
# so the budget is never approached and every verdict is a proof.
SEARCH_NODE_BUDGET = 200_000


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
    window: int = TRAILING_WINDOW,
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
    window: int = TRAILING_WINDOW,
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


# --- the existence search ----------------------------------------------------


def _weak_orderings(items: Sequence[str]) -> Iterator[List[Tuple[str, ...]]]:
    """Every ranking of `items` that allows ties, as a list of tiers.

    Ties are enumerated rather than assumed away because a draw is not
    dominated: it moves the higher-rated of two opponents down and the
    lower one up, which can serve the challenger better than either strict
    ordering.

    Tiers are ORDERED tuples, not sets. That is not cosmetic:
    calculate_elo_updates applies pair updates sequentially against a
    mutating dict, so its output depends on the order agents are enumerated
    in — and iterating a set of strings varies between processes under hash
    randomisation. With sets here the same field could be judged reachable
    on one run and unreachable on the next.
    """
    items = list(items)
    if not items:
        yield []
        return
    for size in range(1, len(items) + 1):
        for combo in itertools.combinations(items, size):
            rest = [i for i in items if i not in combo]
            for tail in _weak_orderings(rest):
                yield [tuple(combo)] + tail


def _orderings_for(opponents: Sequence[str]) -> tuple[Iterable[List[Tuple[str, ...]]], bool]:
    """Orderings to search, plus whether the search is exhaustive."""
    if len(opponents) <= MAX_EXHAUSTIVE_OPPONENTS:
        return list(_weak_orderings(opponents)), True
    # Heuristic: weakest wins, so the highest-rated agents shed the most
    # rating and the challenger's path clears fastest.
    ascending = sorted(opponents)
    return [[(a,) for a in ascending]], False


def _round_participants(
    elos: Dict[str, float],
    challenger_id: str,
    n_opponents: int,
    round_index: int,
    forced_opponent: Optional[str],
) -> Iterator[List[str]]:
    """Opponent sets the challenger could face in one round.

    Mirrors select_agents_for_iteration's priority order rather than assuming
    a top-k field, because the two differ in a way that matters:

      P1  pending winners      -- the previous round's winner is ALWAYS pending
                                  (it won at N-1 and was tested at N-1, so
                                  last_test <= last_win holds)
      P2  the newly evolved agent -- always gets a slot
      P3  untested agents
      P4  random from the top 2*slots with Elo > 1500

    So at agents_per_iteration=3 a round is exactly: pending winner + new agent
    + one filler. One of the challenger's two games is FORCED and one is free:

      round 1     forced slot = the winner of the last completed iteration
      round 2+    the challenger, having won, is itself the pending winner, so
                  the forced slot is that iteration's OWN newly evolved agent,
                  entering at INITIAL_ELO

    The free slot is searched over the top-2-by-Elo pool. Exploring the
    most favourable draw is legitimate rather than optimistic: P4's draw is
    random, so a draw that lands on the leader is a real path through the
    algorithm, and reachability asks whether any path exists. That is why the
    leader can be dragged down every round even though it is never guaranteed
    to play.
    """
    rivals = sorted(
        (a for a in elos if a != challenger_id),
        key=lambda a: elos[a],
        reverse=True,
    )
    if not rivals:
        return

    # P4 draws from the top 2*slots; with one free slot that is the top 2, so
    # the leader is always an available choice.
    free_pool = rivals[:2] or rivals[:1]

    if n_opponents <= 1:
        # Nothing free to search: the single slot is the forced one. This is
        # the KotH shape (agents_per_iteration=2).
        if forced_opponent and forced_opponent in elos:
            yield [forced_opponent]
        else:
            yield [rivals[0]]
        return

    forced = forced_opponent if (forced_opponent and forced_opponent in elos) else None
    yielded = False
    for chosen in free_pool:
        if forced is not None and chosen == forced:
            continue
        opponents = [forced] if forced is not None else []
        opponents.append(chosen)
        # Any remaining slots (agents_per_iteration > 3) fill from the pool.
        for extra in rivals:
            if len(opponents) >= n_opponents:
                break
            if extra not in opponents:
                opponents.append(extra)
        yield opponents
        yielded = True

    if not yielded:
        # The free pool offers nothing distinct from the forced slot, i.e. the
        # whole pool is one agent. Yield the short round rather than nothing:
        # the real selector fills what it can and runs a smaller round-robin,
        # it does not invent agents. Yielding nothing here left the search
        # unable to expand at all, so it reported "no line of play passes X"
        # about a lone 1500-rated agent the challenger would simply beat.
        yield [forced] if forced is not None else [rivals[0]]


def _round_states(
    elos: Dict[str, float],
    challenger_id: str,
    n_opponents: int,
    k: int,
    round_index: int,
    forced_opponent: Optional[str],
) -> Iterator[tuple[Dict[str, float], Optional[str]]]:
    """Every rating state reachable from `elos` by one round the challenger wins.

    Yields (next ratings, forced opponent for the following round). After the
    challenger wins it becomes the pending winner, so the next round's forced
    slot is a freshly evolved agent -- created here at INITIAL_ELO so the
    arithmetic is exact rather than an approximation of one.
    """
    working = dict(elos)
    if round_index > 0:
        # The challenger is the pending winner now; the forced slot is this
        # iteration's own new agent. Distinct id per round so it never
        # collides, and it can never become the leader (it only ever loses).
        forced_opponent = f"{CHALLENGER_ID}_rival_{round_index}"
        working[forced_opponent] = INITIAL_ELO

    for opponents in _round_participants(
        working, challenger_id, n_opponents, round_index, forced_opponent
    ):
        orderings, _ = _orderings_for(opponents)
        for ordering in orderings:
            # Encode the ranking as scores: the challenger strictly above
            # every tier, then one score per tier, descending.
            results = {challenger_id: {'average_score': float(len(ordering) + 1)}}
            for tier_idx, tier in enumerate(ordering):
                tier_score = float(len(ordering) - tier_idx)
                for agent in tier:
                    results[agent] = {'average_score': tier_score}
            sub_elos = {a: working[a] for a in results}
            nxt = dict(working)
            nxt.update(calculate_elo_updates(sub_elos, results, k))
            yield nxt, None


def _leads(elos: Dict[str, float], challenger_id: str,
           penalties: Dict[str, float]) -> float:
    """Challenger's margin over the best rival, on the judged (post-penalty)
    basis -- the one find_best_agent reads."""
    judged = apply_clone_penalties(elos, penalties)
    rivals = [v for a, v in judged.items() if a != challenger_id]
    return judged[challenger_id] - max(rivals) if rivals else math.inf


def search_reachable(
    current_elos: Dict[str, float],
    *,
    rounds: int,
    agents_per_iteration: int,
    k: int = K_FACTOR,
    challenger_id: str = CHALLENGER_ID,
    penalties: Optional[Dict[str, float]] = None,
    previous_winner: Optional[str] = None,
    node_budget: int = SEARCH_NODE_BUDGET,
) -> tuple[bool, Dict[str, float], bool]:
    """Does ANY line of play leave a new agent on top of the ladder?

    Reachability is existential, not an optimisation: one winning path is
    enough, and which path does not matter. So this is a depth-first search
    that RETURNS THE MOMENT it finds a state where the challenger leads,
    rather than computing a best-case projection and testing it at the end.

    That distinction is what makes an "unreachable" verdict trustworthy. The
    earlier formulation optimised each round in isolation, and the per-round
    objective genuinely fights itself over a horizon -- keeping opponents
    highly rated earns the challenger more, while pushing them down is what
    clears its path -- so a locally-greedy line could miss a winning one and
    report a false "unreachable". Searching for existence has no such gap.

    Cost is asymmetric in the useful direction. A reachable field usually
    exits on the first path tried, because the natural ordering is also the
    winning one; only the rare unreachable field pays for the full tree, and
    that is the answer worth being sure about. Two further bounds keep even
    that finite: an admissible prune (no line can close the gap faster than
    `k` per game) and `node_budget`.

    Returns:
        (reachable, ratings of the winning line — or the best line found if
        none, for diagnostics — whether an UNREACHABLE answer was proven
        rather than cut short by the budget; always True when reachable,
        since one winning line settles existence)
    """
    penalties = penalties or {}
    start = dict(current_elos)
    start[challenger_id] = INITIAL_ELO
    rounds = max(0, int(rounds))
    n_opponents = max(1, agents_per_iteration - 1)

    # Admissible bound: per round the challenger plays n_opponents games and
    # can gain at most k each, while the strongest rival can shed at most the
    # same. If that cannot close the gap, no line in this subtree can.
    max_swing_per_round = 2.0 * k * n_opponents

    best_state = start
    best_lead = _leads(start, challenger_id, penalties)
    complete = True
    exhausted = False
    nodes = 0

    # (ratings, rounds left, forced opponent for this round). The forced slot
    # starts as the last completed iteration's winner and is replaced each
    # subsequent round by a freshly evolved agent -- see _round_states.
    stack: List[tuple[Dict[str, float], int, Optional[str]]] = [
        (start, rounds, previous_winner)
    ]
    while stack:
        elos, left, forced = stack.pop()
        lead = _leads(elos, challenger_id, penalties)
        if lead > 0:
            # Existence proved. `complete` is reported True unconditionally
            # here: it exists to say whether an UNREACHABLE verdict was
            # proven, and a found winning line is definitive whether or not
            # the rest of the tree was ever explored.
            return True, elos, True
        if lead > best_lead:
            best_lead, best_state = lead, elos
        if left <= 0 or exhausted:
            continue
        if lead + max_swing_per_round * left <= 0:
            continue                              # unreachable subtree
        round_index = rounds - left
        for nxt, next_forced in _round_states(
            elos, challenger_id, n_opponents, k, round_index, forced
        ):
            nodes += 1
            if nodes > node_budget:
                # Stop EXPANDING, but keep draining the stack. Discarding it
                # here would throw away already-generated states, one of which
                # may be the winner -- turning a budget limit into a false
                # "unreachable".
                complete = False
                exhausted = True
                break
            stack.append((nxt, left - 1, next_forced))

    return False, best_state, complete


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
    # Did the search actually run, or did a gate return early? Distinguishes
    # "no line exists" from "we never looked".
    search_ran: bool = False
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
        if self.search_ran:
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
    clone_penalties: Optional[Dict[str, float]] = None,
    binding_constraint: str = "none",
    previous_winner: Optional[str] = None,
    history_depth: Optional[int] = None,
    min_history: int = TRAILING_WINDOW,
    k: int = K_FACTOR,
) -> ReachabilityVerdict:
    """Decide whether an agent evolved next iteration could lead on Elo.

    Args:
        current_elos: PRE-penalty ratings (see `strip_clone_penalties`).
        rounds_remaining: from `remaining_rounds`; `inf` means unbounded.
        agents_per_iteration: round-robin size, hence games per round.
        clone_penalties: agent_id -> penalty, re-applied before comparing
            because leadership is judged post-penalty.

    Returns:
        A verdict carrying the numbers behind it, for logging.
    """
    penalties = clone_penalties or {}

    # Too early to judge. Checked before anything else, and before the
    # horizon is even consulted, because with a short history the horizon is
    # the least trustworthy number available — see TRAILING_WINDOW.
    if history_depth is not None and history_depth < min_history:
        return ReachabilityVerdict(
            reachable=True,
            reason=(
                f"only {history_depth} completed iteration(s) of history; the "
                f"guard needs {min_history} before its horizon estimate means "
                f"anything"
            ),
            rounds_remaining=rounds_remaining,
            binding_constraint=binding_constraint,
        )

    if not current_elos:
        return ReachabilityVerdict(
            reachable=True,
            reason="no rated agents yet, so nothing to overtake",
            rounds_remaining=rounds_remaining,
            binding_constraint=binding_constraint,
        )

    judged = apply_clone_penalties(current_elos, penalties)
    leader_id = max(judged, key=lambda a: judged[a])

    if math.isinf(rounds_remaining):
        return ReachabilityVerdict(
            reachable=True,
            reason="the horizon is unbounded",
            rounds_remaining=rounds_remaining,
            leader_id=leader_id,
            leader_elo=judged[leader_id],
            binding_constraint=binding_constraint,
        )

    reachable, final, complete = search_reachable(
        current_elos,
        rounds=rounds_playable(rounds_remaining),
        agents_per_iteration=agents_per_iteration,
        k=k,
        penalties=penalties,
        previous_winner=previous_winner,
    )
    judged_projection = apply_clone_penalties(final, penalties)
    challenger = judged_projection[CHALLENGER_ID]
    rivals = {a: v for a, v in judged_projection.items() if a != CHALLENGER_ID}
    best_rival_id = max(rivals, key=lambda a: rivals[a])
    best_rival = rivals[best_rival_id]

    if reachable:
        reason = "a winning line exists"
    elif not complete:
        # The search ran out of budget without finding a winning line, which
        # is NOT the same as proving there isn't one. Firing here would undo
        # the whole point of searching for existence: an unreachable verdict
        # is supposed to be a proof. Fail safe -- decline to fire, and say
        # why, so an operator can see the guard gave up rather than cleared
        # the iteration.
        return ReachabilityVerdict(
            reachable=True,
            reason=(
                f"could not prove unreachability within the search budget "
                f"(best line found reaches "
                f"{judged_projection[CHALLENGER_ID]:.0f} vs "
                f"{best_rival_id}); declining to fire on an unproven verdict"
            ),
            rounds_remaining=rounds_remaining,
            leader_id=leader_id,
            leader_elo=judged[leader_id],
            projected_challenger_elo=challenger,
            projected_best_rival_elo=best_rival,
            search_ran=True,
            search_exhaustive=False,
            binding_constraint=binding_constraint,
        )
    else:
        reason = f"no line of play passes {best_rival_id}"
    return ReachabilityVerdict(
        reachable=reachable,
        reason=reason,
        rounds_remaining=rounds_remaining,
        leader_id=leader_id,
        leader_elo=judged[leader_id],
        projected_challenger_elo=challenger,
        projected_best_rival_elo=best_rival,
        search_ran=True,
        search_exhaustive=complete,
        binding_constraint=binding_constraint,
    )
