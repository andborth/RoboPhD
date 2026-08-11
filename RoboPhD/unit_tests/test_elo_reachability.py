"""Unit tests for RoboPhD/elo_reachability.py.

Two reasons this file is load-bearing beyond ordinary coverage:

  - The Elo update had NO tests at all before it moved here out of
    researcher.py, despite deciding which agent a run returns.
  - The reachability verdict *ends evolution* for an iteration. A false
    "unreachable" silently throws away an evolution round, and nothing
    downstream would look wrong.
"""
import math
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

from RoboPhD.elo_reachability import (
    CHALLENGER_ID,
    INITIAL_ELO,
    K_FACTOR,
    ReachabilityVerdict,
    apply_clone_penalties,
    assess_reachability,
    search_reachable,
    calculate_elo_updates,
    clone_penalty_totals,
    remaining_rounds,
    strip_clone_penalties,
    _weak_orderings,
)


# --- the Elo update itself ----------------------------------------------------


def test_equal_ratings_winner_gains_half_k():
    """Textbook anchor: at equal ratings the expected score is 0.5, so the
    winner gains K/2 and the loser sheds the same."""
    out = calculate_elo_updates(
        {"a": 1500.0, "b": 1500.0},
        {"a": {"average_score": 0.9}, "b": {"average_score": 0.1}},
    )
    assert out["a"] == pytest.approx(1500 + K_FACTOR / 2)
    assert out["b"] == pytest.approx(1500 - K_FACTOR / 2)


def test_a_draw_between_equals_moves_nothing():
    out = calculate_elo_updates(
        {"a": 1500.0, "b": 1500.0},
        {"a": {"average_score": 0.5}, "b": {"average_score": 0.5}},
    )
    assert out["a"] == pytest.approx(1500.0)
    assert out["b"] == pytest.approx(1500.0)


def test_beating_a_stronger_opponent_gains_more():
    """The property the projection's matchmaking assumption rests on."""
    vs_strong = calculate_elo_updates(
        {"new": 1500.0, "strong": 1800.0},
        {"new": {"average_score": 1.0}, "strong": {"average_score": 0.0}},
    )
    vs_weak = calculate_elo_updates(
        {"new": 1500.0, "weak": 1200.0},
        {"new": {"average_score": 1.0}, "weak": {"average_score": 0.0}},
    )
    assert vs_strong["new"] - 1500 > vs_weak["new"] - 1500


def test_float_noise_collapses_into_a_draw():
    """Scores are rounded to 6dp before grouping, so a difference below
    that is a tie rather than a win — otherwise identical agents would
    trade rating on numerical dust."""
    out = calculate_elo_updates(
        {"a": 1500.0, "b": 1500.0},
        {"a": {"average_score": 0.5}, "b": {"average_score": 0.5 + 1e-9}},
    )
    assert out["a"] == pytest.approx(1500.0)
    assert out["b"] == pytest.approx(1500.0)


def test_zero_sum_across_a_round_robin():
    out = calculate_elo_updates(
        {"a": 1500.0, "b": 1600.0, "c": 1400.0},
        {"a": {"average_score": 0.5},
         "b": {"average_score": 0.9},
         "c": {"average_score": 0.1}},
    )
    assert sum(out.values()) == pytest.approx(1500 + 1600 + 1400)


def test_input_is_not_mutated():
    current = {"a": 1500.0, "b": 1500.0}
    calculate_elo_updates(
        current, {"a": {"average_score": 1.0}, "b": {"average_score": 0.0}}
    )
    assert current == {"a": 1500.0, "b": 1500.0}


# --- clone-penalty basis ------------------------------------------------------


def test_penalties_round_trip():
    penalties = clone_penalty_totals([("dup", "orig", 4)])
    judged = {"dup": 1300.0, "orig": 1500.0}
    raw = strip_clone_penalties(judged, penalties)
    assert raw == {"dup": 1500.0, "orig": 1500.0}
    assert apply_clone_penalties(raw, penalties) == judged


def test_repeated_detection_stacks():
    """The replay subtracts once per detection, so the totals must too."""
    penalties = clone_penalty_totals([("dup", "a", 3), ("dup", "b", 7)])
    assert penalties["dup"] == 400.0


def test_penalty_decides_who_counts_as_leader():
    """The reason the basis is tracked at all. Pre-penalty, dup's 1750 tops
    orig's 1600; post-penalty it is 1550 and orig leads. Leadership is
    judged post-penalty, because that is the number find_best_agent reads.
    Asserted in both directions so a dropped `apply_clone_penalties` call
    cannot pass."""
    field = {"dup": 1750.0, "orig": 1600.0}   # pre-penalty ratings
    penalties = clone_penalty_totals([("dup", "orig", 2)])

    penalised = assess_reachability(
        field, rounds_remaining=1, agents_per_iteration=3,
        clone_penalties=penalties,
    )
    assert penalised.leader_id == "orig"
    assert penalised.leader_elo == pytest.approx(1600.0)

    unpenalised = assess_reachability(
        field, rounds_remaining=1, agents_per_iteration=3,
    )
    assert unpenalised.leader_id == "dup", (
        "without the penalty the same field must rank the other way, else "
        "the test is not exercising the penalty at all"
    )


# --- remaining horizon --------------------------------------------------------


@pytest.mark.parametrize("evals,budget,expected", [
    ([30, 30, 30], 150, 2.0),        # 90 spent, 60 left, 30/iter
    ([30, 30, 30], 90, 0.0),         # exactly exhausted
    ([30, 30, 30], 60, 0.0),         # overspent clamps at 0, never negative
    ([10, 10, 40, 40, 40], 240, 2.0),  # trailing mean 28 -> 100 left / 28
])
def test_remaining_rounds_arithmetic(evals, budget, expected):
    got = remaining_rounds(evals, budget)
    if expected == 2.0 and evals[0] == 10:
        # 100/28 = 3.57; recompute rather than hardcode the mean
        expected = (budget - sum(evals)) / (sum(evals[-5:]) / len(evals[-5:]))
    assert got == pytest.approx(expected)


def test_trailing_window_beats_a_lifetime_mean():
    """Why the spec asks for a 5-iteration window: the (agent, example)
    cache makes later iterations far cheaper, so a lifetime mean would
    understate the horizon and fire the guard early."""
    evals = [100, 100, 100, 10, 10, 10, 10, 10]
    windowed = remaining_rounds(evals, 400, window=5)
    lifetime = (400 - sum(evals)) / (sum(evals) / len(evals))
    assert windowed > lifetime


@pytest.mark.parametrize("evals,budget", [
    ([30, 30], None),   # no budget configured
    ([], 500),          # no history yet
    ([0, 0, 0], 500),   # a fully-cached window would divide by zero
])
def test_unknown_horizon_is_unbounded(evals, budget):
    """An absent signal must read as 'plenty of room' so a missing input
    can never be what ends evolution."""
    assert remaining_rounds(evals, budget) == math.inf


@pytest.mark.parametrize("remaining,playable", [
    (0.0, 0),    # budget already gone; the run ends before the next iteration
    (0.1, 1),    # a sliver of budget still buys a whole overshooting round
    (0.7, 1),
    (1.0, 1),
    (1.6, 2),
    (2.0, 2),
    (2.4, 3),
])
def test_rounds_playable_rounds_up(remaining, playable):
    """Ceiling, not floor. The evaluation-budget check runs AFTER an
    iteration, so a partial round's budget still buys a full iteration that
    overshoots and then ends the run. Flooring modelled an agent created with
    0.7 rounds left as never playing a game, and called it unreachable on
    that technicality — it fired the guard a whole iteration early."""
    from RoboPhD.elo_reachability import rounds_playable

    assert rounds_playable(remaining) == playable


def test_a_challenger_created_at_the_buzzer_still_plays_once():
    """The concrete consequence: with a fraction of a round left, the agent
    is created, evaluated, and can move on the ladder."""
    from RoboPhD.elo_reachability import rounds_playable

    _, final, _ = search_reachable(
        {"a": 1500.0, "b": 1500.0},
        rounds=rounds_playable(0.7),
        agents_per_iteration=3,
    )
    assert final[CHALLENGER_ID] > INITIAL_ELO, (
        "an agent that plays a round must be able to gain rating"
    )


# --- the three horizon scenarios ----------------------------------------------
#
# A run can end two ways and either may come first, so all three combinations
# have to work: budget only (the normal case, every example defaults
# --num-iterations to 999 so the budget binds), iteration cap only, and both.


def test_budget_only_is_the_budget_horizon():
    from RoboPhD.elo_reachability import horizon

    rounds, binding = horizon(
        [30, 30, 30], 150, current_iteration=4, num_iterations=None
    )
    assert (rounds, binding) == (2.0, "budget")


def test_iterations_only_has_a_real_horizon():
    """The bug this pair of terminators fixes: with no budget configured the
    horizon used to come back inf and the guard could never fire, even though
    `--num-iterations 10` at iteration 9 leaves an exactly known 2 rounds."""
    from RoboPhD.elo_reachability import horizon

    rounds, binding = horizon(
        [], None, current_iteration=9, num_iterations=10
    )
    assert (rounds, binding) == (2, "iterations")


def test_both_set_takes_whichever_binds_first():
    from RoboPhD.elo_reachability import horizon

    # Vast budget, tight iteration cap.
    assert horizon([30, 30, 30], 100_000,
                   current_iteration=9, num_iterations=10) == (2, "iterations")
    # Tight budget, loose iteration cap.
    rounds, binding = horizon([30, 30, 30], 120,
                              current_iteration=4, num_iterations=999)
    assert (rounds, binding) == (1.0, "budget")


def test_neither_set_is_unbounded():
    from RoboPhD.elo_reachability import horizon

    rounds, binding = horizon([], None, current_iteration=1, num_iterations=None)
    assert math.isinf(rounds)
    assert binding == "none"


@pytest.mark.parametrize("current,cap,expected", [
    (10, 10, 1),    # the last iteration still runs
    (9, 10, 2),
    (1, 10, 10),
    (11, 10, 0),    # past the cap; the loop has already exited
])
def test_iterations_remaining_counts_the_current_one(current, cap, expected):
    from RoboPhD.elo_reachability import iterations_remaining

    assert iterations_remaining(current, cap) == expected


def test_verdict_names_the_binding_terminator():
    """The remedy differs entirely -- raise --evaluation-budget, or --extend --
    so a user reading the log needs to know which one ran out."""
    from RoboPhD.elo_reachability import horizon

    rounds, binding = horizon([], None, current_iteration=10, num_iterations=10)
    verdict = assess_reachability(
        {"leader": 2200.0, "b": 1500.0},
        rounds_remaining=rounds, agents_per_iteration=3,
        binding_constraint=binding,
    )
    assert not verdict.reachable
    assert "limited by iterations" in verdict.summary()


def test_unbounded_horizon_names_no_terminator():
    verdict = assess_reachability(
        {"a": 1500.0}, rounds_remaining=math.inf, agents_per_iteration=3,
        binding_constraint="none",
    )
    assert "limited by" not in verdict.summary()


def test_guard_fires_on_an_iteration_capped_run_with_no_budget():
    """End-to-end through the guard for scenario 2, which previously could
    not fire at all."""
    stub = _guard_stub(num_iterations=5, iteration_fresh_evals=[30] * 5)
    changed, resolved = _run_guard(stub, iteration=5, evaluation_budget=None)
    assert changed is True
    assert resolved["evolution_strategy"] == "greedy"


def test_extend_reopens_an_iteration_capped_run():
    """--extend raises num_iterations on the researcher, so the sticky greedy
    has to notice and hand evolution back."""
    stub = _guard_stub(num_iterations=5, iteration_fresh_evals=[30] * 5)
    _run_guard(stub, iteration=5, evaluation_budget=None)
    assert stub.config_manager.get_config(5)["evolution_strategy"] == "greedy"

    stub.num_iterations = 25          # what --extend 20 does
    stub.config_manager.set_current_iteration(6)
    changed, resolved = _run_guard(stub, iteration=6, evaluation_budget=None)
    assert changed is True
    assert resolved["evolution_strategy"] == "use_your_judgment"


# --- the minimum-history floor ------------------------------------------------
#
# Two independent reasons, both wanting the same number, which is why the
# default is derived from TRAILING_WINDOW rather than being its own literal:
#
#   1. Iteration 1 runs no evolution, so it costs about half a steady-state
#      iteration. A one- or two-iteration mean understates cost ~2x and so
#      overstates the horizon ~2x, converging around N=5.
#   2. A smoke test (--num-iterations 3 or 5, extended afterwards) would
#      otherwise have the guard fire on its FINAL iteration, quietly turning
#      the last evolution round into a greedy one.


def test_floor_defaults_to_the_averaging_window():
    """Derived, not a coincidence of two literals. Tuning the window has to
    move the floor with it."""
    from RoboPhD.config_manager import ConfigManager
    from RoboPhD.elo_reachability import TRAILING_WINDOW

    defaults = ConfigManager().get_defaults()
    assert defaults["elo_reachability_min_history"] == TRAILING_WINDOW
    assert "elo_reachability_min_rounds" not in defaults, (
        "min_rounds was removed: its only justification was distrust of a "
        "greedy search that no longer exists, and an unproven verdict now "
        "fails safe instead of being capped away"
    )


@pytest.mark.parametrize("depth", [0, 1, 2, 3, 4])
def test_short_history_never_fires(depth):
    """A runaway leader and no horizon at all still must not fire while the
    history is too short to average."""
    verdict = assess_reachability(
        {"leader": 5000.0, "b": 1500.0},
        rounds_remaining=0,
        agents_per_iteration=3,
        history_depth=depth,
    )
    assert verdict.reachable
    assert not verdict.search_ran
    assert f"only {depth} completed iteration" in verdict.reason


def test_a_full_window_lets_the_guard_through():
    verdict = assess_reachability(
        {"leader": 5000.0, "b": 1500.0},
        rounds_remaining=1,
        agents_per_iteration=3,
        history_depth=5,
    )
    assert not verdict.reachable


def test_floor_is_skipped_when_depth_is_unstated():
    """Callers that do not track history (ad-hoc use of the module) should
    not silently get a floor they never asked for."""
    verdict = assess_reachability(
        {"leader": 5000.0, "b": 1500.0},
        rounds_remaining=1, agents_per_iteration=3,
    )
    assert not verdict.reachable


def test_floor_is_checked_before_the_horizon():
    """With a short history the horizon is the least trustworthy number
    available, so it must not be what decides anything — including via the
    unbounded-horizon early-out, which reports a different reason."""
    verdict = assess_reachability(
        {"leader": 5000.0}, rounds_remaining=math.inf,
        agents_per_iteration=3, history_depth=1,
    )
    assert "completed iteration" in verdict.reason
    assert "rounds remain" not in verdict.reason


def test_floor_applies_whichever_terminator_binds():
    """The averaging rationale is about history depth, not about which
    terminator is in play. `--evaluation-budget 60` at ~30 evals/iteration is
    a 2-iteration run — the CLAUDE.md smoke config — and firing there is wrong
    for the same reason as an iteration-capped smoke test."""
    for binding in ("budget", "iterations"):
        verdict = assess_reachability(
            {"leader": 5000.0, "b": 1500.0},
            rounds_remaining=0, agents_per_iteration=3,
            history_depth=2, binding_constraint=binding,
        )
        assert verdict.reachable, f"fired too early on a {binding}-bound run"


def test_a_five_iteration_smoke_test_is_fully_protected():
    """The case that motivated the floor. Simulated across the whole run:
    without it the guard fires at iteration 5, the smoke test's last
    evolution round."""
    from RoboPhD.elo_reachability import horizon

    def field_after(n):
        elos = {"a": 1500.0, "b": 1500.0, "c": 1500.0}
        for _ in range(n):
            elos = calculate_elo_updates(elos, {
                "a": {"average_score": 0.9},
                "b": {"average_score": 0.5},
                "c": {"average_score": 0.1}})
        return elos

    fired_without, fired_with = [], []
    for iteration in range(2, 6):
        rounds, binding = horizon(
            [20] * (iteration - 1), None,
            current_iteration=iteration, num_iterations=5,
        )
        common = dict(rounds_remaining=rounds, agents_per_iteration=3,
                      binding_constraint=binding)
        if not assess_reachability(field_after(iteration - 1), **common).reachable:
            fired_without.append(iteration)
        if not assess_reachability(
            field_after(iteration - 1), history_depth=iteration - 1, **common
        ).reachable:
            fired_with.append(iteration)

    assert fired_without == [5], (
        f"expected the unprotected guard to fire on the final iteration, "
        f"got {fired_without}"
    )
    assert fired_with == [], f"the floor should suppress all of it, got {fired_with}"


def test_the_floor_does_not_cover_larger_smoke_tests():
    """Stated as a known limit rather than left to be discovered. A run capped
    at 8 still fires on its last iteration (history depth 7 clears a floor of
    5); the backstop there is --extend restoring the displaced strategy."""
    from RoboPhD.elo_reachability import horizon

    elos = {"a": 1500.0, "b": 1500.0, "c": 1500.0}
    for _ in range(7):
        elos = calculate_elo_updates(elos, {
            "a": {"average_score": 0.9}, "b": {"average_score": 0.5},
            "c": {"average_score": 0.1}})
    rounds, binding = horizon([20] * 7, None, current_iteration=8,
                              num_iterations=8)
    verdict = assess_reachability(
        elos, rounds_remaining=rounds, agents_per_iteration=3,
        binding_constraint=binding, history_depth=7,
    )
    assert not verdict.reachable


def test_guard_respects_the_floor_end_to_end():
    stub = _guard_stub(iteration_fresh_evals=[30, 30])   # depth 2
    changed, resolved = _run_guard(stub, iteration=3)
    assert changed is False
    assert resolved["evolution_strategy"] == "use_your_judgment"


def test_guard_floor_is_configurable():
    """So an 8-iteration smoke habit can raise it past the default."""
    stub = _guard_stub(iteration_fresh_evals=[30] * 7)   # depth 7
    changed, _ = _run_guard(stub, iteration=8)
    assert changed is True, "depth 7 clears the default floor of 5"

    stub = _guard_stub(iteration_fresh_evals=[30] * 7)
    changed, _ = _run_guard(stub, iteration=8, elo_reachability_min_history=10)
    assert changed is False, "a raised floor must suppress the same case"


# --- ordering enumeration -----------------------------------------------------


@pytest.mark.parametrize("n,count", [(0, 1), (1, 1), (2, 3), (3, 13), (4, 75)])
def test_weak_ordering_counts_are_fubini(n, count):
    items = [f"a{i}" for i in range(n)]
    assert len(list(_weak_orderings(items))) == count


def test_weak_orderings_include_the_all_tied_case():
    orderings = list(_weak_orderings(["a", "b"]))
    assert [("a", "b")] in orderings, (
        f"the both-tied ranking is missing from {orderings}"
    )


# --- the existence search -----------------------------------------------------
#
# Reachability is existential, so the search returns the moment it finds a
# winning line rather than computing an optimum and testing it. That is what
# makes an "unreachable" verdict a proof instead of a failed heuristic: an
# earlier greedy-per-round formulation could miss a winning line because the
# per-round objective fights itself over a horizon (keeping opponents highly
# rated earns more, pushing them down clears the path).


def test_challenger_enters_at_base_rating():
    _, final, _ = search_reachable(
        {"a": 1600.0}, rounds=0, agents_per_iteration=3
    )
    assert final[CHALLENGER_ID] == INITIAL_ELO


def test_zero_rounds_leaves_the_field_untouched():
    field = {"a": 1700.0, "b": 1650.0}
    _, final, _ = search_reachable(field, rounds=0, agents_per_iteration=3)
    assert final["a"] == 1700.0 and final["b"] == 1650.0


def test_more_rounds_never_hurt():
    """Monotonicity. The search must not find a winning line at one horizon
    and lose it at a longer one."""
    field = {"leader": 1750.0, "b": 1600.0, "c": 1550.0}
    verdicts = [
        search_reachable(field, rounds=r, agents_per_iteration=3)[0]
        for r in range(1, 9)
    ]
    first = next((i for i, v in enumerate(verdicts) if v), None)
    if first is not None:
        assert all(verdicts[first:]), f"reachability flip-flopped: {verdicts}"


def test_search_arithmetic_is_the_real_elo_update():
    """Not a model of the ladder — the same function that scores it."""
    _, final, _ = search_reachable(
        {"a": 1500.0, "b": 1500.0}, rounds=1, agents_per_iteration=3,
    )
    manual = calculate_elo_updates(
        {CHALLENGER_ID: 1500.0, "a": 1500.0, "b": 1500.0},
        {CHALLENGER_ID: {"average_score": 3.0},
         "a": {"average_score": 2.0},
         "b": {"average_score": 1.0}},
    )
    assert final[CHALLENGER_ID] == pytest.approx(manual[CHALLENGER_ID])


def test_success_short_circuits():
    """The whole point: a reachable field must not pay for the tree. An
    obviously-winnable field should settle in a handful of nodes, which a
    tiny budget proves without needing to instrument the search."""
    field = {"a": 1500.0, "b": 1490.0}
    reachable, _, complete = search_reachable(
        field, rounds=5, agents_per_iteration=3, node_budget=4,
    )
    assert reachable, (
        "a winning line exists immediately, so a 4-node budget must suffice"
    )
    assert complete, (
        "a found winner settles existence; the flag only qualifies an "
        "UNREACHABLE answer"
    )


def test_unreachable_pays_for_the_tree_but_stays_bounded():
    field = {"leader": 5000.0, "b": 1500.0, "c": 1480.0}
    reachable, _, complete = search_reachable(
        field, rounds=5, agents_per_iteration=3,
    )
    assert not reachable
    assert complete, "at 3 agents/iteration the tree is small enough to prove"


def test_budget_exhaustion_is_reported_not_hidden():
    """An incomplete search must not masquerade as a proof.

    Needs a wide round-robin: at 3 agents/iteration the admissible prune
    disposes of a hopeless field in a handful of nodes, so the budget is
    never approached -- which is the desirable behaviour, just not what this
    test is about."""
    field = {f"a{i}": 1600.0 - i * 5 for i in range(8)}
    reachable, _, complete = search_reachable(
        field, rounds=6, agents_per_iteration=6, node_budget=5,
    )
    assert not reachable
    assert complete is False


def test_budget_exhaustion_still_reports_a_winner_it_already_found():
    """Regression: the budget path used to clear the stack, discarding
    already-generated states -- so a winning line one node past the limit
    became a false "unreachable"."""
    field = {"a": 1500.0, "b": 1490.0}
    reachable, _, _ = search_reachable(
        field, rounds=5, agents_per_iteration=3, node_budget=1,
    )
    assert reachable, (
        "a state generated before the budget ran out must still be checked"
    )


def test_the_forced_pending_winner_can_decide_the_verdict():
    """assess_reachability must thread previous_winner into the search.

    The forced slot spends one of the challenger's two games on whoever won
    last, whether or not that agent is the obstacle. When the previous winner
    is weak the cost is doubled: the forced game pays little, AND the game it
    displaces would have dragged a top rival down. Here that is the whole
    difference between a winning line existing and not.

    Without this the pass-through is unpinned — every other previous_winner
    test works below assess_reachability (_round_participants) or on a field
    lopsided enough that the forced slot cannot change the outcome.
    """
    field = {"leader": 1545.0, "second": 1541.0, "weak_winner": 1500.0}
    common = dict(rounds_remaining=1.0, agents_per_iteration=3,
                  binding_constraint="budget", history_depth=10)

    free = assess_reachability(field, previous_winner=None, **common)
    forced = assess_reachability(field, previous_winner="weak_winner", **common)

    assert free.reachable, (
        "free choice of the top two: the challenger climbs while dragging "
        "them down, so a line exists"
    )
    assert not forced.reachable, (
        "forced onto the weak previous winner, the challenger gains less and "
        "the leader is never played, so no line exists"
    )
    assert forced.projected_challenger_elo < free.projected_challenger_elo
    assert forced.projected_best_rival_elo > free.projected_best_rival_elo


def test_the_forced_slot_changes_which_rival_is_unpassable():
    """The verdict names the agent no line passes, and the forced slot can
    change which one that is: the leader can be attacked, but an agent the
    challenger is never free to play can only be passed on points. Observed
    on asta_paper_finder_20260809_222409 iteration 21, where the guard named
    the third-rated agent rather than the leader."""
    field = {"leader": 1560.0, "runner_up": 1556.0, "winner": 1555.0}
    common = dict(rounds_remaining=1.0, agents_per_iteration=3,
                  binding_constraint="budget", history_depth=10)

    forced = assess_reachability(field, previous_winner="winner", **common)

    assert not forced.reachable
    assert "runner_up" in forced.reason, (
        f"expected the un-attackable rival to be named, got: {forced.reason}"
    )


def test_forced_slot_is_the_previous_winner_in_round_one():
    """select_agents_for_iteration's P1: the previous round's winner is always
    pending, so it always occupies one of the challenger's two games."""
    from RoboPhD.elo_reachability import _round_participants

    elos = {"leader": 1700.0, "second": 1650.0, "mid": 1540.0}
    sets = list(_round_participants(elos, CHALLENGER_ID, 2, 0, "mid"))
    assert sets, "no participant sets produced"
    for opponents in sets:
        assert "mid" in opponents, (
            f"the forced pending winner is missing from {opponents}"
        )
    # ...and the leader is reachable in the free slot, because P4 draws from
    # the top 2 by Elo.
    assert any("leader" in o for o in sets)


def test_later_rounds_force_a_fresh_rival_not_a_strong_one():
    """Once the challenger is itself the pending winner, the forced slot is
    that iteration's OWN newly evolved agent at 1500 — worth far less to beat
    than a strong agent, which is why real climbing is slower than a
    top-k model implies."""
    from RoboPhD.elo_reachability import _round_states

    elos = {CHALLENGER_ID: 1560.0, "leader": 1700.0, "second": 1650.0}
    states = list(_round_states(elos, CHALLENGER_ID, 2, K_FACTOR, 1, None))
    assert states
    fresh = [a for a, _ in states[0][0].items() if a.startswith(f"{CHALLENGER_ID}_rival_")]
    assert fresh, "no freshly evolved rival was introduced for round 2"


def test_search_is_flagged_when_orderings_are_truncated():
    """Past MAX_EXHAUSTIVE_OPPONENTS the ordering enumeration falls back to a
    heuristic, and the verdict has to say so rather than imply a proof."""
    from RoboPhD.elo_reachability import _orderings_for

    _, exhaustive = _orderings_for([f"a{i}" for i in range(9)])
    assert exhaustive is False
    _, small = _orderings_for(["a", "b"])
    assert small is True


# --- the verdict --------------------------------------------------------------


def test_a_long_finite_horizon_is_searched_not_skipped():
    """min_rounds used to short-circuit here. With it gone a long horizon is
    searched like any other, and a runaway leader is still reachable given
    enough rounds — found by the search rather than assumed by a threshold."""
    verdict = assess_reachability(
        {"a": 1900.0, "b": 1500.0}, rounds_remaining=15, agents_per_iteration=3
    )
    assert verdict.reachable
    assert verdict.search_ran, "a finite horizon must actually be searched"


def test_unbounded_horizon_never_fires():
    verdict = assess_reachability(
        {"a": 9999.0}, rounds_remaining=math.inf, agents_per_iteration=3
    )
    assert verdict.reachable


def test_unreachable_against_a_runaway_leader_with_one_round_left():
    """The case the guard exists for: a huge gap and no time to close it."""
    verdict = assess_reachability(
        {"leader": 2200.0, "b": 1500.0, "c": 1500.0},
        rounds_remaining=1,
        agents_per_iteration=3,
    )
    assert not verdict.reachable
    assert verdict.search_ran
    assert verdict.leader_id == "leader"
    assert verdict.projected_challenger_elo < verdict.projected_best_rival_elo


def test_reachable_when_the_field_is_close():
    verdict = assess_reachability(
        {"a": 1505.0, "b": 1500.0, "c": 1495.0},
        rounds_remaining=2,
        agents_per_iteration=3,
    )
    assert verdict.reachable, verdict.summary()


def test_empty_field_is_reachable():
    verdict = assess_reachability({}, rounds_remaining=1, agents_per_iteration=3)
    assert verdict.reachable
    assert not verdict.search_ran


def test_summary_is_legible_in_both_directions():
    unreachable = assess_reachability(
        {"leader": 2200.0, "b": 1500.0},
        rounds_remaining=1, agents_per_iteration=3,
    )
    assert "UNREACHABLE" in unreachable.summary()
    assert "leader" in unreachable.summary()

    reachable = assess_reachability(
        {"a": 1500.0}, rounds_remaining=math.inf, agents_per_iteration=3
    )
    assert "unbounded" in reachable.summary()
    assert "UNREACHABLE" not in reachable.summary()


def test_verdict_is_immutable():
    """It is an audit record of a decision that skipped an evolution round."""
    verdict = assess_reachability(
        {"a": 1500.0}, rounds_remaining=1, agents_per_iteration=3
    )
    with pytest.raises(Exception):
        verdict.reachable = True  # type: ignore[misc]


# --- researcher wiring --------------------------------------------------------


def test_researcher_delegates_to_this_module():
    """One Elo formula, not two. If researcher grows its own copy, the
    guard's verdict starts describing a ladder the run is not climbing."""
    from RoboPhD.researcher import ParallelAgentResearcher

    assert ParallelAgentResearcher._calculate_elo_updates is calculate_elo_updates


def test_greedy_is_schedulable_from_meta_evolution():
    """Regression: the schedule validator used to whitelist only "none",
    so meta-evolution could never schedule greedy/challenger/random — it
    was told they did not exist."""
    from RoboPhD.meta_evolution_manager import MetaEvolutionManager
    from RoboPhD.researcher import ParallelAgentResearcher

    builtin = MetaEvolutionManager._builtin_strategy_names()
    assert builtin == set(ParallelAgentResearcher.NON_FILE_STRATEGIES)
    for name in ("greedy", "challenger", "random", "none"):
        assert name in builtin


def test_guard_is_on_by_default():
    """Flipped 2026-08-05 on the strength of the 121-run replay: fires in
    66% of eligible runs, never suppressing a run's own winning agent."""
    from RoboPhD.config_manager import ConfigManager

    defaults = ConfigManager().get_defaults()
    assert defaults["elo_reachability_guard"] is True


# --- the guard's own wiring ---------------------------------------------------
#
# Behavior tests against _apply_reachability_guard with a stub researcher: a
# real run spends minutes in evolution sessions, but the parts that can break
# silently are all here — reading config, writing the delta through
# ConfigManager, the alternation rule, and the state that carries it.


def _guard_stub(**overrides):
    """Minimal stand-in carrying only what the guard touches."""
    from types import SimpleNamespace

    from RoboPhD.config_manager import ConfigManager
    from RoboPhD.researcher import ParallelAgentResearcher

    cm = ConfigManager()
    cm.set_initial_config({"elo_reachability_guard": True,
                           "evaluation_budget": 160})
    cm.set_current_iteration(overrides.get("iteration", 5))
    stub = SimpleNamespace(
        # Pulled from the class rather than hardcoded, so a rename fails
        # here instead of silently diverging from what the guard reads.
        _NON_EVOLVING_STRATEGIES=ParallelAgentResearcher._NON_EVOLVING_STRATEGIES,
        # The verdict helper is a plain method on the same class; binding it
        # keeps the stub exercising the real computation rather than a mock.
        _reachability_verdict=lambda config, iteration: (
            ParallelAgentResearcher._reachability_verdict(stub, config, iteration)
        ),
        # Loose, like every shipped example's --num-iterations default, so
        # the budget is what binds unless a test says otherwise.
        num_iterations=999,
        # Bound from the class so the real history-reading logic runs.
        _last_iteration_winner=lambda: (
            ParallelAgentResearcher._last_iteration_winner(stub)
        ),
        test_history=[{"leader": {"average_score": 0.9},
                       "b": {"average_score": 0.4}}],
        config_manager=cm,
        reachability_guard_state={},
        # A runaway leader with a nearly-exhausted budget: unreachable.
        performance_records={
            "leader": {"elo": 2200.0, "test_count": 5},
            "b": {"elo": 1500.0, "test_count": 3},
            "c": {"elo": 1450.0, "test_count": 3},
        },
        clone_detections=[],
        # Five entries so the default min-history floor is cleared; these
        # tests are about the decision logic, not the floor. 150 of 160 spent
        # leaves a third of a round -- one playable iteration.
        iteration_fresh_evals=[30] * 5,
    )
    for key, value in overrides.items():
        setattr(stub, key, value)
    return stub


def _run_guard(stub, iteration=5, **config_overrides):
    from RoboPhD.researcher import ParallelAgentResearcher

    config = stub.config_manager.get_config(iteration)
    config.update(config_overrides)
    changed = ParallelAgentResearcher._apply_reachability_guard(
        stub, iteration, config
    )
    return changed, stub.config_manager.get_config(iteration)


def test_the_guard_feeds_the_real_pending_winner_into_the_verdict():
    """researcher passes _last_iteration_winner() as the forced opponent.

    The other guard tests run on a runaway 2200 leader, unreachable whichever
    agent is forced, so dropping that argument would not fail them. This field
    is reachable when the challenger may pick its two games and unreachable
    once the weak previous winner takes a slot, so the wiring — including
    reading the winner out of test_history by score — is load-bearing.
    """
    field = {
        "leader": {"elo": 1545.0, "test_count": 5},
        "second": {"elo": 1541.0, "test_count": 5},
        "weak_winner": {"elo": 1500.0, "test_count": 3},
    }
    history = [{"weak_winner": {"average_score": 0.9},
                "leader": {"average_score": 0.4}}]

    stub = _guard_stub(performance_records=field, test_history=history)
    changed, resolved = _run_guard(stub)

    assert changed is True, (
        "with the weak previous winner forced into the round there is no "
        "winning line, so the guard should switch to greedy"
    )
    assert resolved["evolution_strategy"] == "greedy"

    # Same field, but the strong leader won last: it can then be attacked in
    # the forced slot, and a line exists again.
    stub = _guard_stub(
        performance_records=field,
        test_history=[{"leader": {"average_score": 0.9},
                       "weak_winner": {"average_score": 0.4}}],
    )
    changed, resolved = _run_guard(stub)

    assert changed is False, (
        "forced onto the leader the challenger can drag it down, so evolution "
        "must be left alone"
    )
    assert resolved["evolution_strategy"] == "use_your_judgment"


def test_guard_rewrites_the_strategy_when_unreachable():
    stub = _guard_stub()
    changed, resolved = _run_guard(stub)
    assert changed is True
    assert resolved["evolution_strategy"] == "greedy"
    assert stub.reachability_guard_state["fired_at"] == 5
    assert stub.reachability_guard_state["displaced_strategy"] == "use_your_judgment"


def test_guard_records_its_decision_in_config_history():
    """The decision skips an evolution round, so it has to be auditable
    rather than a runtime-only mutation."""
    from RoboPhD.config_manager import ConfigSource

    stub = _guard_stub()
    _run_guard(stub)
    entries = [e for e in stub.config_manager.config_change_history
               if e["source"] == ConfigSource.ELO_REACHABILITY.value]
    assert len(entries) == 1
    assert entries[0]["delta"] == {"evolution_strategy": "greedy"}
    assert "could not become" in entries[0]["rationale"]


def test_guard_is_inert_when_disabled():
    stub = _guard_stub()
    changed, resolved = _run_guard(stub, elo_reachability_guard=False)
    assert changed is False
    assert resolved["evolution_strategy"] == "use_your_judgment"


def test_guard_never_fires_at_iteration_one():
    """Iteration 1 has no evolution step to displace."""
    stub = _guard_stub()
    changed, _ = _run_guard(stub, iteration=1)
    assert changed is False


def test_guard_skips_already_non_evolving_strategies():
    """Nothing to save: these rounds already spend no evolution session."""
    for strategy in ("none", "greedy", "challenger"):
        stub = _guard_stub()
        changed, _ = _run_guard(stub, evolution_strategy=strategy)
        assert changed is False, f"fired needlessly on {strategy!r}"


def test_guard_still_fires_on_a_random_round():
    """`random` is non-FILE but very much evolving: it resolves to a randomly
    chosen real strategy and runs a full session. Treating it as non-evolving
    (the two sets look nearly identical) would make the guard decline exactly
    the round it exists to save."""
    from RoboPhD.researcher import ParallelAgentResearcher

    assert "random" not in ParallelAgentResearcher._NON_EVOLVING_STRATEGIES
    assert "random" in ParallelAgentResearcher.NON_FILE_STRATEGIES, (
        "random stays schedulable without a strategy directory"
    )
    stub = _guard_stub()
    changed, resolved = _run_guard(stub, evolution_strategy="random")
    assert changed is True
    assert resolved["evolution_strategy"] == "greedy"
    assert stub.reachability_guard_state["displaced_strategy"] == "random"


def test_greedy_is_sticky_once_fired():
    """Alternating back would spend an evolution session on the worst
    candidate in the run. The verdict only deteriorates from here — a greedy
    round still spends evaluations, so the horizon shrinks — so there is
    nothing to go back for."""
    stub = _guard_stub()
    _run_guard(stub, iteration=5)
    assert stub.config_manager.get_config(5)["evolution_strategy"] == "greedy"

    stub.config_manager.set_current_iteration(6)
    changed, resolved = _run_guard(stub, iteration=6)
    assert changed is False, "nothing to change; greedy is inherited"
    assert resolved["evolution_strategy"] == "greedy"


def test_sticky_is_not_a_latch_when_the_horizon_grows_back():
    """The property that makes stickiness safe. --extend or a raised budget
    can make agents reachable again, and the guard has to notice rather than
    hold the run in greedy for the rest of its life."""
    stub = _guard_stub()
    _run_guard(stub, iteration=5)
    assert stub.config_manager.get_config(5)["evolution_strategy"] == "greedy"

    # An --extend-sized budget increase: the horizon reopens.
    stub.config_manager.set_current_iteration(6)
    changed, resolved = _run_guard(stub, iteration=6, evaluation_budget=100_000)
    assert changed is True
    assert resolved["evolution_strategy"] == "use_your_judgment", (
        "the displaced strategy must come back, not some default"
    )
    assert stub.reachability_guard_state == {}


def test_restore_only_reverts_what_the_guard_itself_displaced():
    """A greedy round the user or meta-evolution asked for is not the
    guard's to undo, however the horizon looks."""
    stub = _guard_stub(reachability_guard_state={})
    changed, _ = _run_guard(
        stub, iteration=6, evolution_strategy="greedy",
        evaluation_budget=100_000,
    )
    assert changed is False


def test_guard_state_survives_a_checkpoint_round_trip():
    """The displaced strategy spans iterations, so a resume must not lose
    what evolution should return to."""
    import json

    stub = _guard_stub()
    _run_guard(stub, iteration=5)
    revived = json.loads(json.dumps(stub.reachability_guard_state))
    assert revived == {"fired_at": 5, "displaced_strategy": "use_your_judgment"}


def test_guard_ignores_untested_agents():
    """An agent with test_count 0 sits at the default 1500 by convention,
    not by result; counting it as a rival would invent a leader."""
    stub = _guard_stub(performance_records={
        "untested": {"elo": 1500.0, "test_count": 0},
    })
    changed, _ = _run_guard(stub)
    assert changed is False, "no rated agents means nothing to overtake"


# --- King of the Hill exclusion -----------------------------------------------


def _koth(**extra):
    from RoboPhD.config_manager import ConfigManager, ConfigSource

    cfg = {"agents_per_iteration": 2, "oldest_agent_wins_ties": True}
    cfg.update(extra)
    ConfigManager().set_initial_config(cfg, ConfigSource.CLI)


def test_guard_cannot_be_enabled_on_a_koth_run():
    """agents_per_iteration=2 with oldest_agent_wins_ties is King of the Hill.

    The guard's QUESTION is wrong here, not just its answer. A KotH run's
    result is whichever agent won the last round, full stop — which is why
    runner_utils.find_last_winner exists and documents that "the last-round
    winner may differ from the Elo leader". A newly evolved agent becomes the
    output by winning one round whatever its rating, so no agent is ever dead
    weight and greedy saves nothing.
    """
    with pytest.raises(ValueError, match="King-of-the-Hill"):
        _koth(elo_reachability_guard=True)


def test_koth_without_the_guard_is_fine():
    _koth()


def test_two_agents_alone_is_not_koth():
    """The tie rule is what identifies the format; a two-agent round-robin
    with ordinary tie handling is not it."""
    from RoboPhD.config_manager import ConfigManager, ConfigSource

    ConfigManager().set_initial_config(
        {"agents_per_iteration": 2, "elo_reachability_guard": True},
        ConfigSource.CLI,
    )


def test_koth_conflict_is_caught_when_assembled_across_deltas():
    """Neither delta is wrong on its own, so a per-delta check would miss it.
    The combination is validated on the RESOLVED config for this reason."""
    from RoboPhD.config_manager import ConfigManager, ConfigSource

    cm = ConfigManager()
    cm.set_initial_config(
        {"elo_reachability_guard": True, "agents_per_iteration": 2},
        ConfigSource.CLI,
    )
    cm.set_current_iteration(3)
    with pytest.raises(ValueError, match="King-of-the-Hill"):
        cm.apply_delta(3, {"oldest_agent_wins_ties": True}, ConfigSource.SCHEDULE)


def test_koth_error_names_both_escape_routes():
    from RoboPhD.config_manager import ConfigManager, ConfigSource

    with pytest.raises(ValueError) as exc:
        ConfigManager().set_initial_config(
            {"agents_per_iteration": 2, "oldest_agent_wins_ties": True,
             "elo_reachability_guard": True},
            ConfigSource.CLI,
        )
    msg = str(exc.value)
    assert "elo_reachability_guard" in msg
    assert "agents_per_iteration" in msg


# --- unproven verdicts fail safe ----------------------------------------------


def test_budget_exhaustion_declines_to_fire():
    """The point of searching for existence is that "unreachable" is a proof.
    A search that ran out of budget without finding a winning line has proved
    nothing, so firing there would undo the whole reframing. It must read as
    reachable, and say why."""
    field = {"leader": 2300.0, "b": 1520.0, "c": 1500.0, "d": 1490.0, "e": 1480.0}
    verdict = assess_reachability(
        field, rounds_remaining=5, agents_per_iteration=5,
    )
    assert verdict.reachable, "fired on a verdict it could not prove"
    assert verdict.search_exhaustive is False
    assert "unproven" in verdict.reason


def test_a_proven_unreachable_still_fires():
    """The fail-safe must not swallow the real case."""
    verdict = assess_reachability(
        {"leader": 2200.0, "b": 1500.0, "c": 1480.0},
        rounds_remaining=1, agents_per_iteration=3,
    )
    assert not verdict.reachable
    assert verdict.search_exhaustive is True


def test_replay_script_does_not_double_apply_clone_penalties():
    """Regression, and a subtle one: the two callers need OPPOSITE handling.

    researcher._reachability_verdict reads performance_records['elo'], which
    _recalculate_all_elo_scores has already penalised, so it must strip.
    scripts/elo_reachability.py replays test_history itself and stops before
    the penalty step, so its ratings are already pre-penalty and stripping
    again adds 200 per clone — inventing unbeatable phantom leaders and
    producing false "unreachable" verdicts. That cost two false suppressions
    and nine spurious firing runs across a 152-run sweep before it was found.
    """
    import ast

    src = (REPO_ROOT / "scripts" / "elo_reachability.py").read_text()
    # A CALL, not the string — the file names the function in a comment
    # explaining precisely why it must not be called.
    calls = [
        n for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.Call)
        and getattr(n.func, "id", getattr(n.func, "attr", None))
        == "strip_clone_penalties"
    ]
    assert not calls, (
        f"the replay script's ratings are already pre-penalty; stripping them "
        f"double-counts the clone penalty (call at line {calls[0].lineno if calls else '?'})"
    )
    assert "clone_penalties=penalties" in src, (
        "it must still PASS the penalties, which are re-applied at comparison "
        "time because leadership is judged post-penalty"
    )


def test_verdicts_are_deterministic_across_processes():
    """A reachability verdict must not depend on PYTHONHASHSEED.

    _weak_orderings used to yield sets. calculate_elo_updates applies pair
    updates sequentially against a mutating dict, so its result depends on
    the order agents are enumerated in — and set-of-strings iteration order
    varies between processes. The same field could be judged reachable on
    one run and unreachable on the next, which showed up as an intermittently
    failing test before it was understood as a real defect.
    """
    import subprocess
    import sys as _sys

    prog = (
        "from RoboPhD.elo_reachability import search_reachable;"
        "f={'leader':2300.0,'b':1520.0,'c':1500.0,'d':1490.0,'e':1480.0};"
        "ok,fin,c=search_reachable(f,rounds=4,agents_per_iteration=5);"
        "print(f'{ok}|{c}|{fin[\"<challenger>\"]:.6f}')"
    )
    seen = set()
    for seed in ("0", "1", "12345"):
        out = subprocess.run(
            [_sys.executable, "-c", prog], capture_output=True, text=True,
            cwd=str(REPO_ROOT), env={"PYTHONHASHSEED": seed, "PATH": os.environ["PATH"],
                                     "HOME": os.environ.get("HOME", "")},
        )
        line = [l for l in out.stdout.splitlines() if "|" in l]
        assert line, f"subprocess produced no verdict: {out.stderr[-400:]}"
        seen.add(line[-1])
    assert len(seen) == 1, f"verdict varied with hash seed: {seen}"


def test_tiers_are_ordered_not_sets():
    """The structural half of the guard above: a set here reintroduces the
    nondeterminism regardless of what the end-to-end test happens to hit."""
    for tiers in _weak_orderings(["a", "b", "c"]):
        for tier in tiers:
            assert isinstance(tier, tuple), (
                f"tier {tier!r} is a {type(tier).__name__}; ordered tiers are "
                f"what make the Elo update reproducible"
            )


def _write_fake_checkpoint(tmp_path, *, agents_per_iteration, oldest_wins):
    import json

    ckpt = {
        "test_history": [
            {"a": {"average_score": 0.9}, "b": {"average_score": 0.4}},
            {"a": {"average_score": 0.8}, "b": {"average_score": 0.5}},
            {"a": {"average_score": 0.7}, "b": {"average_score": 0.6}},
        ],
        "iteration_fresh_evals": [20, 20, 20],
        "performance_records": {"a": {"elo": 1550.0}, "b": {"elo": 1450.0}},
        "num_iterations": 10,
        "config_manager": {"resolved_configs": {"1": {
            "agents_per_iteration": agents_per_iteration,
            "oldest_agent_wins_ties": oldest_wins,
            "evaluation_budget": 200,
        }}},
    }
    d = tmp_path / "run"
    d.mkdir(parents=True, exist_ok=True)
    (d / "checkpoint.json").write_text(json.dumps(ckpt))
    return d


def _run_replay_script(run_dir):
    import subprocess
    import sys as _sys

    return subprocess.run(
        [_sys.executable, str(REPO_ROOT / "scripts" / "elo_reachability.py"),
         str(run_dir)],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )


def test_replay_script_refuses_koth_runs(tmp_path):
    """The retrospective must exclude what the runtime forbids.

    Sweeping the archive without this measured 31 KotH runs the guard can
    never legally run on, and the results looked plausible rather than
    broken: a KotH ladder freezes an early leader once it stops playing, so
    unreachable verdicts appear with lots of runway and read as deep
    multi-round saves. That inflated "18 runs fire with 2+ rounds of runway,
    up to horizon 6" out of a true figure of 2 runs at horizon 2.

    Behavioural rather than source-level: an earlier AST guard here looked
    for a hand-rolled `agents_per_iteration == 2` and silently missed an
    injected copy that used a local alias, which is precisely the false
    confidence a structural check invites.
    """
    koth = _write_fake_checkpoint(tmp_path, agents_per_iteration=2, oldest_wins=True)
    result = _run_replay_script(koth)
    assert result.returncode != 0, "replayed a run the guard cannot be enabled on"
    assert "King-of-the-Hill" in result.stderr + result.stdout


def test_replay_script_accepts_an_ordinary_run(tmp_path):
    """The exclusion must be specific — two agents alone, or KotH tie-handling
    at a wider round-robin, are both legal."""
    for api, oldest in ((3, True), (2, False), (3, False)):
        run = _write_fake_checkpoint(
            tmp_path / f"c{api}{oldest}", agents_per_iteration=api,
            oldest_wins=oldest)
        result = _run_replay_script(run)
        assert result.returncode == 0, (
            f"refused a legal config (agents_per_iteration={api}, "
            f"oldest_agent_wins_ties={oldest}): {result.stdout[-300:]}"
        )


def test_replay_script_delegates_the_koth_definition():
    """One definition of an illegal configuration, not two that can drift."""
    src = (REPO_ROOT / "scripts" / "elo_reachability.py").read_text()
    assert "_validate_resolved_semantics" in src, (
        "the replay script must reuse ConfigManager's definition, not carry "
        "its own copy"
    )


def test_a_single_rival_still_produces_a_round():
    """Regression: when the pool holds one agent it is BOTH the forced slot
    and the entire free pool, so the de-duplication that stops the free slot
    repeating the forced one used to skip the only viable set and yield
    nothing at all. The search then could not expand and reported
    "no line of play passes X" about a lone 1500-rated agent the challenger
    would simply beat.

    Reachable in normal operation only via --min-history 0 (the floor gates
    the early iterations where a pool this small occurs), but a proven-
    unreachable verdict that is flatly wrong is worth closing regardless.
    """
    from RoboPhD.elo_reachability import _round_participants

    elos = {"solo": 1500.0, CHALLENGER_ID: 1500.0}
    sets = list(_round_participants(elos, CHALLENGER_ID, 2, 0, "solo"))
    assert sets == [["solo"]], f"expected one short round, got {sets}"

    reachable, final, _ = search_reachable(
        {"solo": 1500.0}, rounds=1, agents_per_iteration=3,
        previous_winner="solo",
    )
    assert reachable
    assert final[CHALLENGER_ID] > INITIAL_ELO


def test_short_rounds_when_the_pool_is_smaller_than_the_round_robin():
    """agents_per_iteration is a cap, not a requirement — the selector fills
    what it can. The search must model that rather than assuming a full field."""
    for n_rivals in (1, 2, 3):
        elos = {f"a{i}": 1500.0 + i for i in range(n_rivals)}
        reachable, _, complete = search_reachable(
            elos, rounds=2, agents_per_iteration=5,
        )
        assert complete, f"search failed to terminate with {n_rivals} rival(s)"
        assert reachable, f"a level field with {n_rivals} rival(s) must be reachable"


# --- default-on must not conscript King-of-the-Hill runs ----------------------
#
# The guard being the default changes what a KotH conflict means. An explicit
# enable is still an error worth stopping for; a run that never mentioned the
# guard must not fail at startup quoting a flag its operator never passed —
# that is the misdirection the validator's own message exists to avoid. 20% of
# archived runs are this shape.


def test_default_on_does_not_reject_a_koth_run(tmp_path):
    from RoboPhD.config_manager import ConfigManager, ConfigSource

    ConfigManager().set_initial_config(
        {"agents_per_iteration": 2, "oldest_agent_wins_ties": True},
        ConfigSource.CLI,
    )   # must not raise


def test_explicit_enable_on_koth_is_still_an_error():
    from RoboPhD.config_manager import ConfigManager, ConfigSource

    with pytest.raises(ValueError, match="King-of-the-Hill"):
        ConfigManager().set_initial_config(
            {"agents_per_iteration": 2, "oldest_agent_wins_ties": True,
             "elo_reachability_guard": True},
            ConfigSource.CLI,
        )


def test_explicit_enable_across_deltas_on_koth_is_still_an_error():
    """The conflict can be assembled without any single delta being wrong,
    which is why this is validated on the resolved config."""
    from RoboPhD.config_manager import ConfigManager, ConfigSource

    cm = ConfigManager()
    cm.set_initial_config(
        {"elo_reachability_guard": True, "agents_per_iteration": 2},
        ConfigSource.CLI,
    )
    cm.set_current_iteration(3)
    with pytest.raises(ValueError, match="King-of-the-Hill"):
        cm.apply_delta(3, {"oldest_agent_wins_ties": True}, ConfigSource.SCHEDULE)


def test_guard_is_inert_at_runtime_on_a_koth_run():
    """Not rejecting is not enough — the guard must actually decline to act,
    or a defaulted-on KotH run would still lose its evolution rounds."""
    stub = _guard_stub()
    changed, resolved = _run_guard(
        stub, iteration=5, agents_per_iteration=2, oldest_agent_wins_ties=True,
    )
    assert changed is False
    assert resolved["evolution_strategy"] == "use_your_judgment"


def test_guard_still_acts_on_two_agents_without_koth_tie_handling():
    """The exemption is the FORMAT, not the round-robin size."""
    stub = _guard_stub()
    changed, _ = _run_guard(
        stub, iteration=5, agents_per_iteration=2, oldest_agent_wins_ties=False,
    )
    assert changed is True


# --- resume must respect the stored value, on or off --------------------------
#
# A default flip is a change to NEW runs. An in-flight campaign resumed after
# the flip must keep whatever it was running under, or the scoring behaviour
# changes mid-run — the same concern that made the judge-default flip
# (13706a15) engineer stored-value-wins semantics.
#
# The shielding is doubly redundant, which is worth knowing before anyone
# "simplifies" either half. A checkpoint stores BOTH iteration_configs[0] (the
# defaults snapshot as of the run's start, read by _resolve_base_config in
# preference to live get_defaults()) and resolved_configs[1] (which the N>=2
# inheritance chain terminates on). Measured: dropping either one alone still
# resolves the old value; only dropping both lets the live default through.
#
# It is also why no example packs this key into engine_overrides — that is
# re-applied as a delta on resume, so it would override the stored value. The
# guard arrives only from a user's own --engine-config, so a flagless resume
# adds nothing.


def _resume(initial_config, *, stored_defaults=None):
    """Round-trip a ConfigManager through a checkpoint and resolve ahead.

    `stored_defaults` overwrites the iteration-0 snapshot, which is how a
    checkpoint written under different code defaults actually looks.
    """
    import json

    from RoboPhD.config_manager import ConfigManager, ConfigSource

    cm = ConfigManager()
    cm.set_initial_config(dict(initial_config), ConfigSource.CLI)
    for key, value in (stored_defaults or {}).items():
        cm.iteration_configs[0][key] = value
        cm.resolved_configs[0][key] = value
        if key not in initial_config:
            cm.resolved_configs[1][key] = value
    revived = ConfigManager.from_checkpoint(
        json.loads(json.dumps(cm.to_checkpoint()))
    )
    revived.set_current_iteration(9)
    return revived.get_config(9)


def test_resume_keeps_an_explicit_enable():
    assert _resume({"elo_reachability_guard": True})["elo_reachability_guard"] is True


def test_resume_keeps_an_explicit_disable():
    """The case that would silently change a run's behaviour if the flip
    leaked through: someone turned the guard OFF, then resumed."""
    assert _resume({"elo_reachability_guard": False})["elo_reachability_guard"] is False


def test_resume_of_a_pre_flip_run_does_not_gain_the_guard():
    """A campaign started while the default was False, resumed afterwards,
    must not silently acquire the guard for its remaining iterations."""
    cfg = _resume({}, stored_defaults={"elo_reachability_guard": False})
    assert cfg["elo_reachability_guard"] is False, (
        "an in-flight run gained the guard mid-campaign; the checkpoint's "
        "stored iteration_configs[0] / resolved_configs[1] are what shield it "
        "from a default flip, and something has stopped reading them"
    )


def test_resume_of_a_run_predating_the_key_leaves_it_absent():
    """Checkpoints written before the key existed have no entry at all. The
    guard reads it with .get(), so absent means off — which is the correct
    reading of 'this run never had the feature'."""
    import json

    from RoboPhD.config_manager import ConfigManager, ConfigSource

    cm = ConfigManager()
    cm.set_initial_config({}, ConfigSource.CLI)
    for d in (cm.iteration_configs[0], cm.resolved_configs[0],
              cm.resolved_configs[1]):
        d.pop("elo_reachability_guard", None)
    revived = ConfigManager.from_checkpoint(
        json.loads(json.dumps(cm.to_checkpoint()))
    )
    revived.set_current_iteration(9)
    assert revived.get_config(9).get("elo_reachability_guard") in (None, False)


def test_no_example_packs_the_guard_into_engine_overrides():
    """engine_overrides is re-applied as a delta on resume, so anything an
    example packs there unconditionally would override the stored value.
    The guard must arrive only from a user's own --engine-config."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2] / "examples"
    offenders = [
        str(p.relative_to(root.parent))
        for p in root.glob("*/main.py")
        if "elo_reachability" in p.read_text()
    ]
    assert not offenders, (
        f"these examples would push the guard through engine_overrides and "
        f"clobber a resumed run's stored value: {offenders}"
    )
