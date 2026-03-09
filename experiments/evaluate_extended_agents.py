from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents import AdaptiveQTableAgent, PriorityHeuristicAgent, RandomAgent
from src.checkers.env import Checkers6x6Env


def load_q_table(path: Path) -> dict:
    items = np.load(path, allow_pickle=True)
    return dict(items.tolist())


def _wilson_interval(successes: float, total: float, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    phat = successes / total
    denom = 1.0 + (z * z) / total
    center = (phat + (z * z) / (2.0 * total)) / denom
    margin = z * math.sqrt((phat * (1.0 - phat) + (z * z) / (4.0 * total)) / total) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def _select_action(agent, env: Checkers6x6Env) -> int:
    if isinstance(agent, AdaptiveQTableAgent):
        return agent.select_move_index(env._obs(), env.legal_moves, exploit_only=True)
    if isinstance(agent, PriorityHeuristicAgent):
        if agent.player != env.player:
            agent = PriorityHeuristicAgent(player=env.player)
        return agent.select_action(env)
    return agent.select_action(env)


def play_game(black_agent, red_agent, seed: int) -> dict:
    env = Checkers6x6Env(seed=seed)
    env.reset(seed=seed)
    steps = 0

    while True:
        action = _select_action(black_agent if env.player == "b" else red_agent, env)
        _, _, terminated, truncated, info = env.step(action)
        steps += 1

        if terminated or truncated:
            winner = info.get("winner", "draw")
            black_return = 1.0 if winner == "b" else -1.0 if winner == "r" else 0.0
            return {
                "winner": winner,
                "steps": steps,
                "truncated": bool(truncated),
                "black_return": black_return,
            }


def matchup(agent_a_factory, agent_b_factory, games: int, seed: int, alternate_start: bool = True) -> dict[str, float]:
    a_wins = 0
    a_losses = 0
    draws = 0
    lengths: list[int] = []
    returns: list[float] = []
    truncations = 0

    for i in range(games):
        a_is_black = (i % 2 == 0) or (not alternate_start)
        black = agent_a_factory(i) if a_is_black else agent_b_factory(i)
        red = agent_b_factory(i) if a_is_black else agent_a_factory(i)

        game = play_game(black, red, seed=seed + i)
        black_return = float(game["black_return"])
        a_return = black_return if a_is_black else -black_return

        lengths.append(int(game["steps"]))
        returns.append(a_return)
        truncations += int(game["truncated"])

        if a_return > 0:
            a_wins += 1
        elif a_return < 0:
            a_losses += 1
        else:
            draws += 1

    n = max(1, games)
    return {
        "games": float(games),
        "wins": float(a_wins),
        "losses": float(a_losses),
        "draws": float(draws),
        "win_rate": a_wins / n,
        "loss_rate": a_losses / n,
        "draw_rate": draws / n,
        "avg_length": float(np.mean(lengths)) if lengths else 0.0,
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "truncations": float(truncations),
    }


def evaluate_for_seed(q: dict, games: int, seed: int, alternate_start: bool) -> dict[str, dict[str, float]]:
    def rl_factory(i: int):
        return AdaptiveQTableAgent(q_table=q, seed=seed + 10_000 + i)

    def heur_factory(i: int):
        return PriorityHeuristicAgent(player="b")

    def rand_factory(i: int):
        return RandomAgent(seed=seed + 20_000 + i)

    return {
        "RL vs Random": matchup(rl_factory, rand_factory, games, seed + 1000, alternate_start),
        "RL vs Heuristic": matchup(rl_factory, heur_factory, games, seed + 2000, alternate_start),
        "Heuristic vs Random": matchup(heur_factory, rand_factory, games, seed + 3000, alternate_start),
    }


def aggregate_over_seeds(per_seed: dict[str, dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    labels = next(iter(per_seed.values())).keys()
    aggregate: dict[str, dict[str, float]] = {}
    for label in labels:
        metrics = next(iter(per_seed.values()))[label].keys()
        label_stats: dict[str, float] = {}
        for m in metrics:
            vals = np.array([per_seed[s][label][m] for s in per_seed], dtype=np.float64)
            label_stats[f"{m}_mean"] = float(np.mean(vals))
            label_stats[f"{m}_std"] = float(np.std(vals))

        pooled_games = float(sum(per_seed[s][label]["games"] for s in per_seed))
        pooled_wins = float(sum(per_seed[s][label]["wins"] for s in per_seed))
        ci_low, ci_high = _wilson_interval(pooled_wins, pooled_games)
        label_stats["pooled_games"] = pooled_games
        label_stats["pooled_wins"] = pooled_wins
        label_stats["win_rate_ci95_low"] = ci_low
        label_stats["win_rate_ci95_high"] = ci_high
        aggregate[label] = label_stats
    return aggregate


def bar_plot(results: dict[str, float], out_path: Path) -> None:
    labels = list(results.keys())
    values = [results[k] for k in labels]
    plt.figure(figsize=(9, 5))
    plt.bar(labels, values)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Winrate")
    plt.title("Agent comparison (extended)")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate RL/Heuristic/Random agents (extended RL)")
    p.add_argument("--q-table", type=str, default="experiments/results/q_table.npy")
    p.add_argument("--games", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-seeds", type=int, default=5)
    p.add_argument("--alternate-start", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--out", type=str, default="experiments/results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    q = load_q_table(Path(args.q_table))
    seeds = [args.seed + 10_000 * i for i in range(args.num_seeds)]
    per_seed: dict[str, dict[str, dict[str, float]]] = {}
    for s in seeds:
        per_seed[f"seed_{s}"] = evaluate_for_seed(q, args.games, s, args.alternate_start)

    aggregate = aggregate_over_seeds(per_seed)

    for label, stats in aggregate.items():
        print(
            f"{label}: "
            f"win={stats['win_rate_mean']:.3f}, "
            f"loss={stats['loss_rate_mean']:.3f}, "
            f"draw={stats['draw_rate_mean']:.3f}, "
            f"avg_len={stats['avg_length_mean']:.2f}, "
            f"return={stats['mean_return_mean']:.3f}, "
            f"ci95=[{stats['win_rate_ci95_low']:.3f},{stats['win_rate_ci95_high']:.3f}]"
        )

    bar_plot(
        {label: stats["win_rate_mean"] for label, stats in aggregate.items()},
        out_dir / "head_to_head_winrates.png",
    )

    summary = {
        "config": {
            "games": args.games,
            "base_seed": args.seed,
            "num_seeds": args.num_seeds,
            "alternate_start": args.alternate_start,
            "mode": "extended",
        },
        "per_seed": per_seed,
        "aggregate": aggregate,
    }
    (out_dir / "evaluation_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
