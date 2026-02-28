from __future__ import annotations

import argparse
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


def play_game(black_agent, red_agent, seed: int) -> str:
    env = Checkers6x6Env(seed=seed)
    env.reset(seed=seed)

    while True:
        action = black_agent.select_action(env) if env.player == "b" else red_agent.select_action(env)
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            return info.get("winner", "draw")


def matchup(black_agent, red_agent, games: int, seed: int) -> tuple[float, float]:
    black_wins = 0
    red_wins = 0
    for i in range(games):
        winner = play_game(black_agent, red_agent, seed=seed + i)
        if winner == "b":
            black_wins += 1
        elif winner == "r":
            red_wins += 1

    total = max(1, black_wins + red_wins)
    return black_wins / total, red_wins / total


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
    p.add_argument("--out", type=str, default="experiments/results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    q = load_q_table(Path(args.q_table))
    rl = QTableAgent(q, epsilon=0.0, seed=args.seed)
    heuristic = HeuristicAgent()
    random_agent = RandomAgent(seed=args.seed + 1)

    rl_vs_rand, _ = matchup(rl, random_agent, args.games, args.seed + 1000)
    rl_vs_heur, _ = matchup(rl, heuristic, args.games, args.seed + 2000)
    heur_vs_rand, _ = matchup(heuristic, random_agent, args.games, args.seed + 3000)

    summary = {
        "RL vs Random": rl_vs_rand,
        "RL vs Heuristic": rl_vs_heur,
        "Heuristic vs Random": heur_vs_rand,
    }

    for k, v in summary.items():
        print(f"{k}: {v:.3f}")

    bar_plot(summary, out_dir / "head_to_head_winrates.png")


if __name__ == "__main__":
    main()
