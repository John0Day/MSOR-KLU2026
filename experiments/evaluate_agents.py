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

from agents import HeuristicAgent, QTableAgent, RandomAgent
from env import Checkers6x6Env


def load_q_table(path: Path) -> dict:
    items = np.load(path, allow_pickle=True)
    return dict(items.tolist())


def _material_balance_black(board: list[list[str]]) -> int:
    values = {"b": 1, "B": 2, "r": 1, "R": 2}
    black = 0
    red = 0
    for row in board:
        for p in row:
            if p in ("b", "B"):
                black += values[p]
            elif p in ("r", "R"):
                red += values[p]
    return black - red


def _wilson_interval(successes: float, total: float, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    phat = successes / total
    denom = 1.0 + (z * z) / total
    center = (phat + (z * z) / (2.0 * total)) / denom
    margin = z * math.sqrt((phat * (1.0 - phat) + (z * z) / (4.0 * total)) / total) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def play_game(black_agent, red_agent, seed: int) -> dict:
    env = Checkers6x6Env(seed=seed)
    env.reset(seed=seed)
    steps = 0
    black_captures = 0
    red_captures = 0
    black_promotions = 0
    red_promotions = 0

    while True:
        action_player = env.player
        action = black_agent.select_action(env) if action_player == "b" else red_agent.select_action(env)
        chosen = env.legal_moves[action]
        moving_piece = env.board[chosen.from_row][chosen.from_col]
        was_capture = chosen.captured is not None

        _, _, terminated, truncated, info = env.step(action)
        steps += 1

        if was_capture:
            if action_player == "b":
                black_captures += 1
            else:
                red_captures += 1

        landed_piece = env.board[chosen.to_row][chosen.to_col]
        was_promotion = (
            moving_piece in ("b", "r")
            and landed_piece in ("B", "R")
            and moving_piece.lower() == landed_piece.lower()
        )
        if was_promotion:
            if action_player == "b":
                black_promotions += 1
            else:
                red_promotions += 1

        if terminated or truncated:
            winner = info.get("winner", "draw")
            black_return = 1.0 if winner == "b" else -1.0 if winner == "r" else 0.0
            return {
                "winner": winner,
                "steps": steps,
                "truncated": bool(truncated),
                "black_return": black_return,
                "black_captures": float(black_captures),
                "red_captures": float(red_captures),
                "black_promotions": float(black_promotions),
                "red_promotions": float(red_promotions),
                "material_balance_black": float(_material_balance_black(env.board)),
            }


def matchup(agent_a, agent_b, games: int, seed: int, alternate_start: bool = True) -> dict[str, float]:
    a_wins = 0
    a_losses = 0
    draws = 0
    lengths: list[int] = []
    returns: list[float] = []
    captures: list[float] = []
    promotions: list[float] = []
    material_diffs: list[float] = []
    truncations = 0

    for i in range(games):
        a_is_black = (i % 2 == 0) or (not alternate_start)
        black = agent_a if a_is_black else agent_b
        red = agent_b if a_is_black else agent_a

        game = play_game(black, red, seed=seed + i)
        black_return = float(game["black_return"])
        a_return = black_return if a_is_black else -black_return
        a_captures = float(game["black_captures"] if a_is_black else game["red_captures"])
        a_promotions = float(game["black_promotions"] if a_is_black else game["red_promotions"])
        a_material_diff = float(game["material_balance_black"] if a_is_black else -game["material_balance_black"])

        lengths.append(int(game["steps"]))
        returns.append(a_return)
        captures.append(a_captures)
        promotions.append(a_promotions)
        material_diffs.append(a_material_diff)
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
        "avg_captures": float(np.mean(captures)) if captures else 0.0,
        "avg_promotions": float(np.mean(promotions)) if promotions else 0.0,
        "terminal_material_diff": float(np.mean(material_diffs)) if material_diffs else 0.0,
        "truncations": float(truncations),
    }


def bar_plot(results: dict[str, float], out_path: Path) -> None:
    labels = list(results.keys())
    values = [results[k] for k in labels]
    plt.figure(figsize=(9, 5))
    plt.bar(labels, values)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Winrate")
    plt.title("Agent comparison")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate RL/Heuristic/Random agents")
    p.add_argument("--q-table", type=str, default="experiments/results/q_table.npy")
    p.add_argument("--games", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-seeds", type=int, default=1)
    p.add_argument("--alternate-start", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--out", type=str, default="experiments/results")
    return p.parse_args()


def evaluate_for_seed(q: dict, games: int, seed: int, alternate_start: bool) -> dict[str, dict[str, float]]:
    rl = QTableAgent(q, epsilon=0.0, seed=seed)
    heuristic = HeuristicAgent()
    random_agent = RandomAgent(seed=seed + 1)

    return {
        "RL vs Random": matchup(rl, random_agent, games, seed + 1000, alternate_start),
        "RL vs Heuristic": matchup(rl, heuristic, games, seed + 2000, alternate_start),
        "Heuristic vs Random": matchup(heuristic, random_agent, games, seed + 3000, alternate_start),
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


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    q = load_q_table(Path(args.q_table))
    seeds = [args.seed + 10000 * i for i in range(args.num_seeds)]
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
            f"cap={stats['avg_captures_mean']:.2f}, "
            f"prom={stats['avg_promotions_mean']:.2f}, "
            f"mat={stats['terminal_material_diff_mean']:.2f}, "
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
        },
        "per_seed": per_seed,
        "aggregate": aggregate,
    }
    (out_dir / "evaluation_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
