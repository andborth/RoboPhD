#!/usr/bin/env python3
"""
Monte Carlo simulation of RoboPhD budget allocation.

Given a fixed budget of evaluations and a per-iteration improvement rate,
which N (problems per iteration) maximizes the expected true accuracy of
the top-ELO agent?

Core tension: smaller N = more iterations = higher accuracy ceiling, but
noisier ELO rankings may lose track of the best agent. Larger N = fewer
iterations = lower ceiling, but more reliable rankings.

ELO system, agent selection, and winner logic are faithfully ported from
RoboPhD/researcher.py.
"""

import argparse
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# ELO system (ported from researcher.py:1911-1968)
# ---------------------------------------------------------------------------

def calculate_elo_updates(
    current_elos: Dict[str, float],
    iteration_results: Dict[str, Dict],
    k: int = 32,
) -> Dict[str, float]:
    """Calculate updated ELO scores based on head-to-head results."""
    updated_elos = current_elos.copy()
    agents = list(iteration_results.keys())

    # Group agents by accuracy to identify ties
    accuracy_groups: Dict[float, List[str]] = {}
    for agent in agents:
        acc = iteration_results[agent]["accuracy"]
        accuracy_groups.setdefault(acc, []).append(agent)

    # Ties within groups -> draw (0.5 each)
    for group in accuracy_groups.values():
        if len(group) > 1:
            for i, agent1 in enumerate(group):
                for agent2 in group[i + 1 :]:
                    elo1 = updated_elos[agent1]
                    elo2 = updated_elos[agent2]
                    expected1 = 1 / (1 + 10 ** ((elo2 - elo1) / 400))
                    expected2 = 1 / (1 + 10 ** ((elo1 - elo2) / 400))
                    updated_elos[agent1] += k * (0.5 - expected1)
                    updated_elos[agent2] += k * (0.5 - expected2)

    # Wins/losses between different accuracy groups
    sorted_groups = sorted(accuracy_groups.keys(), reverse=True)
    for i, higher_acc in enumerate(sorted_groups[:-1]):
        for lower_acc in sorted_groups[i + 1 :]:
            for winner in accuracy_groups[higher_acc]:
                for loser in accuracy_groups[lower_acc]:
                    winner_elo = updated_elos[winner]
                    loser_elo = updated_elos[loser]
                    expected_winner = 1 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
                    expected_loser = 1 / (1 + 10 ** ((winner_elo - loser_elo) / 400))
                    updated_elos[winner] += k * (1 - expected_winner)
                    updated_elos[loser] += k * (0 - expected_loser)

    return updated_elos


def recalculate_all_elos(
    test_history: List[Dict[str, Dict]],
) -> Dict[str, float]:
    """Recalculate all ELO scores from scratch (ported from researcher.py:1970-2009)."""
    cumulative = {}
    for iteration_data in test_history:
        for agent in iteration_data:
            if agent not in cumulative:
                cumulative[agent] = 1500.0
        current_for_iter = {a: cumulative[a] for a in iteration_data}
        updated = calculate_elo_updates(current_for_iter, iteration_data)
        for agent, new_elo in updated.items():
            cumulative[agent] = new_elo
    return cumulative


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------

class AgentState:
    """Minimal per-agent bookkeeping mirroring researcher.py performance_records."""

    def __init__(self, agent_id: str, true_accuracy: float):
        self.agent_id = agent_id
        self.true_accuracy = true_accuracy
        self.elo = 1500.0
        self.test_count = 0
        self.last_win_iteration: Optional[int] = None
        self.last_test_iteration: Optional[int] = None


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def compute_iterations(n: int, budget: int, agents_per_iteration: int) -> int:
    """Return number of iterations that fit within the budget.

    Cost model:
      - Round 1: N evals (1 agent, no deep focus)
      - Round 2: agents_per_iteration * N (agents_per_iteration agents; deep focus
        means the evolved agent already tested once during evolution, but we still
        evaluate all selected agents on the iteration's N problems)
      - Round 3+: (agents_per_iteration + 1) * N (deep focus adds one extra test)

    Iterations continue until cumulative evals >= budget (overshoot allowed).
    """
    cumulative = 0
    iters = 0
    while cumulative < budget:
        iters += 1
        if iters == 1:
            cumulative += n
        elif iters == 2:
            cumulative += agents_per_iteration * n
        else:
            cumulative += (agents_per_iteration + 1) * n
        # Allow this iteration even if it overshoots
    return iters


def get_pending_winners(
    agents: Dict[str, AgentState],
) -> List[str]:
    """Find agents that won but haven't been tested since (researcher.py:2100-2129)."""
    pending = []
    for aid, state in agents.items():
        if state.last_win_iteration is not None:
            last_test = state.last_test_iteration if state.last_test_iteration is not None else -1
            if last_test <= state.last_win_iteration:
                pending.append(aid)
    # Sort: most recent win first, then ELO descending
    pending.sort(key=lambda a: (-agents[a].last_win_iteration, -agents[a].elo))
    return pending


def select_agents(
    agents: Dict[str, AgentState],
    k: int,
    iteration: int,
    evolved_agent_id: Optional[str],
    rng: random.Random,
) -> List[str]:
    """Select agents for an iteration (researcher.py:2417-2592)."""
    selected: List[str] = []
    available = set(agents.keys())

    # Priority 1: Pending winners
    pending = get_pending_winners(agents)
    for aid in pending:
        if aid in available and len(selected) < k:
            selected.append(aid)
            available.discard(aid)

    # Priority 2: Newly evolved agent (always gets a slot)
    if evolved_agent_id and evolved_agent_id in available:
        if len(selected) >= k:
            # Drop a random pending winner to make room
            dropped = rng.choice(selected)
            selected.remove(dropped)
            available.add(dropped)
        selected.append(evolved_agent_id)
        available.discard(evolved_agent_id)

    if len(selected) >= k:
        final = selected[:k]
        for aid in final:
            agents[aid].last_test_iteration = iteration
        return final

    # Priority 3: Untested agents
    slots = k - len(selected)
    untested = [a for a in available if agents[a].test_count == 0]
    if untested and slots > 0:
        if len(untested) > slots:
            chosen = rng.sample(untested, slots)
        else:
            chosen = list(untested)
        selected.extend(chosen)
        available -= set(chosen)
        slots = k - len(selected)

    # Priority 4: ELO-based (random from top 2*slots with ELO > 1500)
    if slots > 0:
        tested = [a for a in available if agents[a].test_count > 0]
        high_elo = sorted(
            [(a, agents[a].elo) for a in tested if agents[a].elo > 1500],
            key=lambda x: x[1],
            reverse=True,
        )
        if high_elo:
            pool_size = min(slots * 2, len(high_elo))
            pool = [a for a, _ in high_elo[:pool_size]]
            num = min(slots, len(pool))
            elo_chosen = rng.sample(pool, num)
            selected.extend(elo_chosen)
            available -= set(elo_chosen)
            slots = k - len(selected)

        # Fallback: agents with ELO <= 1500, deterministic by ELO
        if slots > 0:
            already = set(selected)
            low_elo = sorted(
                [(a, agents[a].elo) for a in tested if a not in already and agents[a].elo <= 1500],
                key=lambda x: x[1],
                reverse=True,
            )
            for a, _ in low_elo[:slots]:
                selected.append(a)

    final = selected[:k]
    for aid in final:
        agents[aid].last_test_iteration = iteration
    return final


def run_single_simulation(
    n: int,
    budget: int,
    seed_accuracy: float,
    improvement: float,
    agents_per_iteration: int,
    rng: random.Random,
) -> Tuple[float, float, int]:
    """Run one Monte Carlo simulation.

    Returns:
        (top_elo_true_accuracy, best_agent_true_accuracy, num_iterations)
    """
    num_iterations = compute_iterations(n, budget, agents_per_iteration)

    # Initialize seed agent
    agents: Dict[str, AgentState] = {}
    agent_counter = 0

    def make_agent(true_acc: float) -> str:
        nonlocal agent_counter
        aid = f"agent_{agent_counter}"
        agent_counter += 1
        agents[aid] = AgentState(aid, true_acc)
        return aid

    seed_id = make_agent(seed_accuracy)

    test_history: List[Dict[str, Dict]] = []

    for iteration in range(1, num_iterations + 1):
        evolved_id: Optional[str] = None

        if iteration == 1:
            # First iteration: only the seed agent, no evolution
            evolved_id = None
        else:
            # Evolution: new agent based on the winner of the previous iteration
            # Find winner(s) from the last iteration results
            last_results = test_history[-1]
            max_acc = max(v["accuracy"] for v in last_results.values())
            winners = [a for a, v in last_results.items() if v["accuracy"] == max_acc]
            winner = rng.choice(winners)

            # Mark winner
            agents[winner].last_win_iteration = iteration - 1

            # Evolve: new agent's true accuracy = winner's TRUE accuracy + improvement
            winner_true_acc = agents[winner].true_accuracy
            new_true_acc = min(winner_true_acc + improvement, 1.0)
            evolved_id = make_agent(new_true_acc)

        # Select agents
        selected = select_agents(agents, agents_per_iteration, iteration, evolved_id, rng)

        # Evaluate: each agent answers N binary questions (Bernoulli with p=true_accuracy)
        iteration_results: Dict[str, Dict] = {}
        for aid in selected:
            p = agents[aid].true_accuracy
            correct = sum(1 for _ in range(n) if rng.random() < p)
            observed_acc = correct / n
            iteration_results[aid] = {"accuracy": observed_acc}
            agents[aid].test_count += 1

        test_history.append(iteration_results)

        # Find and record winner(s) for this iteration
        max_acc = max(v["accuracy"] for v in iteration_results.values())
        winners = [a for a, v in iteration_results.items() if v["accuracy"] == max_acc]
        for w in winners:
            agents[w].last_win_iteration = iteration

    # Recalculate ELO from scratch
    elo_scores = recalculate_all_elos(test_history)
    for aid, elo in elo_scores.items():
        agents[aid].elo = elo

    # Top-ELO agent
    top_elo_agent = max(agents.values(), key=lambda a: a.elo)
    best_agent = max(agents.values(), key=lambda a: a.true_accuracy)

    return top_elo_agent.true_accuracy, best_agent.true_accuracy, num_iterations


def run_simulations(
    n: int,
    budget: int,
    seed_accuracy: float,
    improvement: float,
    agents_per_iteration: int,
    num_simulations: int,
    base_seed: int,
) -> Dict:
    """Run all simulations for a given N and collect statistics."""
    top_elo_accuracies = []
    best_accuracies = []
    is_best_count = 0
    within_2pct_count = 0
    iteration_counts = []

    for sim in range(num_simulations):
        rng = random.Random(base_seed + sim)
        top_acc, best_acc, num_iters = run_single_simulation(
            n, budget, seed_accuracy, improvement, agents_per_iteration, rng
        )
        top_elo_accuracies.append(top_acc)
        best_accuracies.append(best_acc)
        iteration_counts.append(num_iters)
        if abs(top_acc - best_acc) < 1e-9:
            is_best_count += 1
        if top_acc >= best_acc - 0.02 - 1e-9:
            within_2pct_count += 1

    top_sorted = sorted(top_elo_accuracies)
    mean_acc = sum(top_elo_accuracies) / len(top_elo_accuracies)

    def percentile(sorted_list, p):
        """Linear interpolation percentile (matches numpy default)."""
        n = len(sorted_list)
        idx = p / 100 * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return sorted_list[lo] * (1 - frac) + sorted_list[hi] * frac

    return {
        "n": n,
        "num_iterations": iteration_counts[0],  # deterministic given N and budget
        "num_agents": iteration_counts[0],  # one new agent per iteration (including seed)
        "ceiling": min(seed_accuracy + improvement * (iteration_counts[0] - 1), 1.0),
        "mean_accuracy": mean_acc,
        "p25": percentile(top_sorted, 25),
        "p50": percentile(top_sorted, 50),
        "p75": percentile(top_sorted, 75),
        "p_is_best": is_best_count / num_simulations,
        "p_within_2pct": within_2pct_count / num_simulations,
        "mean_best_accuracy": sum(best_accuracies) / len(best_accuracies),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Monte Carlo simulation of RoboPhD budget allocation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/simulate_budget_allocation.py
  python scripts/simulate_budget_allocation.py --budget 1200 --n-values 30 50 75 100
  python scripts/simulate_budget_allocation.py --num-simulations 1000  # quick run
        """,
    )
    parser.add_argument("--budget", type=int, default=600, help="Total evaluation budget (default: 600)")
    parser.add_argument("--seed-accuracy", type=float, default=0.40, help="Seed agent true accuracy (default: 0.40)")
    parser.add_argument("--improvement", type=float, default=0.02, help="True accuracy improvement per evolution (default: 0.02)")
    parser.add_argument("--n-values", type=int, nargs="+", default=[10, 15, 20, 25, 30, 40, 50], help="Problems per iteration to compare (default: 10 15 20 25 30 40 50)")
    parser.add_argument("--num-simulations", type=int, default=10000, help="Monte Carlo simulations per N (default: 10000)")
    parser.add_argument("--agents-per-iteration", type=int, default=3, help="Agents tested per iteration (default: 3)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    print(f"Budget: {args.budget} evaluations")
    print(f"Seed accuracy: {args.seed_accuracy:.0%}")
    print(f"Improvement per evolution: +{args.improvement:.0%}")
    print(f"Agents per iteration: {args.agents_per_iteration}")
    print(f"Simulations per N: {args.num_simulations:,}")
    print(f"Random seed: {args.seed}")
    print()

    # Compute analytical iteration counts
    print("Analytical iteration counts:")
    for n in sorted(args.n_values):
        iters = compute_iterations(n, args.budget, args.agents_per_iteration)
        ceiling = min(args.seed_accuracy + args.improvement * (iters - 1), 1.0)
        # Compute actual budget used
        total = 0
        for i in range(1, iters + 1):
            if i == 1:
                total += n
            elif i == 2:
                total += args.agents_per_iteration * n
            else:
                total += (args.agents_per_iteration + 1) * n
        print(f"  N={n:3d}: {iters} iterations ({total} evals), ceiling {ceiling:.0%}")
    print()

    # Run simulations
    results = []
    for n in sorted(args.n_values):
        print(f"Simulating N={n}...", end=" ", flush=True)
        result = run_simulations(
            n=n,
            budget=args.budget,
            seed_accuracy=args.seed_accuracy,
            improvement=args.improvement,
            agents_per_iteration=args.agents_per_iteration,
            num_simulations=args.num_simulations,
            base_seed=args.seed,
        )
        results.append(result)
        print(f"done (mean={result['mean_accuracy']:.2%}, P(best)={result['p_is_best']:.1%})")

    # Summary table
    print()
    print("=" * 105)
    print(f"{'N':>4}  {'Iters':>5}  {'Ceiling':>8}  {'Mean':>8}  {'p25':>8}  {'p50':>8}  {'p75':>8}  {'P(best)':>8}  {'P(≤2%)':>8}  {'Gap':>8}")
    print("-" * 105)
    for r in results:
        gap = r["ceiling"] - r["mean_accuracy"]
        print(
            f"{r['n']:4d}  {r['num_iterations']:5d}  "
            f"{r['ceiling']:7.2%}  {r['mean_accuracy']:7.2%}  "
            f"{r['p25']:7.2%}  {r['p50']:7.2%}  {r['p75']:7.2%}  "
            f"{r['p_is_best']:7.1%}  {r['p_within_2pct']:7.1%}  "
            f"{gap:7.2%}"
        )
    print("=" * 105)
    print()
    print("Legend:")
    print("  N         = problems per iteration")
    print("  Iters     = number of iterations within budget")
    print("  Ceiling   = best possible true accuracy (no noise)")
    print("  Mean      = mean true accuracy of top-ELO agent across simulations")
    print("  p25/p50/p75 = percentiles of top-ELO agent true accuracy")
    print("  P(best)   = probability top-ELO agent IS the best agent")
    print("  P(<=2%)   = probability top-ELO agent is within 2% of best")
    print("  Gap       = ceiling - mean (accuracy lost to ELO noise)")

    # Find optimal N
    best = max(results, key=lambda r: r["mean_accuracy"])
    print()
    print(f"Optimal N = {best['n']} (mean accuracy {best['mean_accuracy']:.2%}, "
          f"ceiling {best['ceiling']:.2%}, gap {best['ceiling'] - best['mean_accuracy']:.2%})")


if __name__ == "__main__":
    main()
