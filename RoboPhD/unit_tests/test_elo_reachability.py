"""Unit tests for RoboPhD/elo_reachability.py.

Two reasons this file is load-bearing beyond ordinary coverage:

  - The Elo update had NO tests at all before it moved here out of
    researcher.py, despite deciding which agent a run returns.
  - The reachability verdict *ends evolution* for an iteration. A false
    "unreachable" silently throws away an evolution round, and nothing
    downstream would look wrong.
"""
import math

import pytest

from RoboPhD.elo_reachability import (
    CHALLENGER_ID,
    INITIAL_ELO,
    K_FACTOR,
    ReachabilityVerdict,
    apply_clone_penalties,
    assess_reachability,
    best_case_projection,
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


# --- ordering enumeration -----------------------------------------------------


@pytest.mark.parametrize("n,count", [(0, 1), (1, 1), (2, 3), (3, 13), (4, 75)])
def test_weak_ordering_counts_are_fubini(n, count):
    items = [f"a{i}" for i in range(n)]
    assert len(list(_weak_orderings(items))) == count


def test_weak_orderings_include_the_all_tied_case():
    orderings = list(_weak_orderings(["a", "b"]))
    assert [{"a", "b"}] in orderings


# --- best-case projection -----------------------------------------------------


def test_challenger_enters_at_base_rating():
    projected, _ = best_case_projection(
        {"a": 1600.0}, rounds=0, agents_per_iteration=3
    )
    assert projected[CHALLENGER_ID] == INITIAL_ELO


def test_projection_climbs_with_more_rounds():
    field = {"a": 1700.0, "b": 1650.0, "c": 1600.0}
    one, _ = best_case_projection(field, rounds=1, agents_per_iteration=3)
    three, _ = best_case_projection(field, rounds=3, agents_per_iteration=3)
    assert three[CHALLENGER_ID] > one[CHALLENGER_ID]


def test_zero_rounds_leaves_the_field_untouched():
    field = {"a": 1700.0, "b": 1650.0}
    projected, _ = best_case_projection(field, rounds=0, agents_per_iteration=3)
    assert projected["a"] == 1700.0 and projected["b"] == 1650.0


def test_projection_is_exact_against_a_hand_run_round():
    """The projection must agree with calculate_elo_updates applied
    directly — it is the same ladder, not a model of it."""
    field = {"a": 1500.0, "b": 1500.0}
    projected, _ = best_case_projection(field, rounds=1, agents_per_iteration=3)
    manual = calculate_elo_updates(
        {CHALLENGER_ID: 1500.0, "a": 1500.0, "b": 1500.0},
        {CHALLENGER_ID: {"average_score": 3.0},
         "a": {"average_score": 2.0},
         "b": {"average_score": 1.0}},
    )
    assert projected[CHALLENGER_ID] == pytest.approx(manual[CHALLENGER_ID])


def test_search_is_flagged_when_truncated():
    """Past MAX_EXHAUSTIVE_OPPONENTS the search stops being exhaustive, and
    the verdict has to say so rather than imply a proof it did not run."""
    field = {f"a{i}": 1600.0 + i for i in range(12)}
    _, exhaustive = best_case_projection(
        field, rounds=1, agents_per_iteration=12
    )
    assert exhaustive is False
    _, exhaustive_small = best_case_projection(
        field, rounds=1, agents_per_iteration=3
    )
    assert exhaustive_small is True


# --- the verdict --------------------------------------------------------------


def test_long_horizon_short_circuits_without_projecting():
    verdict = assess_reachability(
        {"a": 5000.0}, rounds_remaining=10, agents_per_iteration=3, min_rounds=3
    )
    assert verdict.reachable
    assert not verdict.projection_ran, (
        "a long horizon should skip the projection entirely, not compute and "
        "discard it"
    )


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
    assert verdict.projection_ran
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
    assert not verdict.projection_ran


def test_more_rounds_can_only_help():
    """Monotonicity: extra rounds must never turn a reachable verdict
    unreachable. A violation would mean the per-round search is fighting
    itself badly enough to invert the answer."""
    field = {"leader": 1750.0, "b": 1600.0, "c": 1550.0}
    verdicts = [
        assess_reachability(
            field, rounds_remaining=r, agents_per_iteration=3, min_rounds=99
        ).reachable
        for r in range(1, 8)
    ]
    first_true = next((i for i, v in enumerate(verdicts) if v), None)
    if first_true is not None:
        assert all(verdicts[first_true:]), (
            f"reachability flip-flopped across horizons: {verdicts}"
        )


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


def test_guard_is_off_by_default():
    """It ends evolution early, so it must be opted into rather than
    silently applied to every existing run configuration."""
    from RoboPhD.config_manager import ConfigManager

    defaults = ConfigManager().get_defaults()
    assert defaults["elo_reachability_guard"] is False
    assert defaults["elo_reachability_min_rounds"] == 3


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
                           "evaluation_budget": 100})
    cm.set_current_iteration(overrides.get("iteration", 5))
    stub = SimpleNamespace(
        # Pulled from the class rather than hardcoded, so a rename fails
        # here instead of silently diverging from what the guard reads.
        _NON_EVOLVING_STRATEGIES=ParallelAgentResearcher._NON_EVOLVING_STRATEGIES,
        config_manager=cm,
        reachability_guard_state={},
        # A runaway leader with a nearly-exhausted budget: unreachable.
        performance_records={
            "leader": {"elo": 2200.0, "test_count": 5},
            "b": {"elo": 1500.0, "test_count": 3},
            "c": {"elo": 1450.0, "test_count": 3},
        },
        clone_detections=[],
        iteration_fresh_evals=[30, 30, 30],   # 90 of 100 spent
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


def test_guard_rewrites_the_strategy_when_unreachable():
    stub = _guard_stub()
    changed, resolved = _run_guard(stub)
    assert changed is True
    assert resolved["evolution_strategy"] == "greedy"
    assert stub.reachability_guard_state["fired_at"] == 5
    assert stub.reachability_guard_state["restore_to"] == "use_your_judgment"


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
    assert stub.reachability_guard_state["restore_to"] == "random"


def test_guard_restores_the_displaced_strategy_next_round():
    """Without this the greedy delta persists and evolution never resumes,
    because config values carry forward until overridden."""
    stub = _guard_stub()
    _run_guard(stub, iteration=5)
    assert stub.config_manager.get_config(5)["evolution_strategy"] == "greedy"

    stub.config_manager.set_current_iteration(6)
    changed, resolved = _run_guard(stub, iteration=6)
    assert changed is True
    assert resolved["evolution_strategy"] == "use_your_judgment"
    assert stub.reachability_guard_state == {}


def test_guard_never_fires_twice_consecutively():
    """The alternation rule, asserted on the round after a firing even
    though the numbers there are even worse (a greedy round spends budget
    too, so the horizon only shrinks)."""
    stub = _guard_stub()
    _run_guard(stub, iteration=5)
    stub.config_manager.set_current_iteration(6)
    _, resolved = _run_guard(stub, iteration=6)
    assert resolved["evolution_strategy"] != "greedy"


def test_guard_state_survives_a_checkpoint_round_trip():
    """The alternation rule spans two iterations, so a resume between them
    must not lose the fact that a greedy round just happened."""
    import json

    stub = _guard_stub()
    _run_guard(stub, iteration=5)
    revived = json.loads(json.dumps(stub.reachability_guard_state))
    assert revived == {"fired_at": 5, "restore_to": "use_your_judgment"}


def test_guard_ignores_untested_agents():
    """An agent with test_count 0 sits at the default 1500 by convention,
    not by result; counting it as a rival would invent a leader."""
    stub = _guard_stub(performance_records={
        "untested": {"elo": 1500.0, "test_count": 0},
    })
    changed, _ = _run_guard(stub)
    assert changed is False, "no rated agents means nothing to overtake"
