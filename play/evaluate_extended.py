from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents import AdaptiveQTableAgent, PriorityHeuristicAgent
from src.checkers.env import Checkers6x6Env


def load_q_table(path: Path) -> dict:
    items = np.load(path, allow_pickle=True)
    return dict(items.tolist())


def run_episode(
    env: Checkers6x6Env,
    agent: AdaptiveQTableAgent,
    opponent: str,
    agent_player: str,
    render: bool,
    sleep: float,
) -> str:
    env.reset()
    done = False
    winner = "draw"

    while not done:
        if env.player == agent_player:
            idx = agent.select_move_index(env._obs(), env.legal_moves, exploit_only=True)
        else:
            if opponent == "random":
                idx = int(np.random.randint(0, len(env.legal_moves)))
            else:
                idx = PriorityHeuristicAgent(player=env.player).select_action(env)

        _, _, term, trunc, info = env.step(idx)
        done = bool(term or trunc)
        if render:
            env.render()
            time.sleep(sleep)
        if done:
            winner = str(info.get("winner", "draw"))

    return winner


def main() -> None:
    p = argparse.ArgumentParser(description="Extended play/eval for trained Q-table")
    p.add_argument("--q-table", type=str, default="experiments/results/q_table.npy")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--opponent", choices=["random", "heuristic"], default="random")
    p.add_argument("--agent-color", choices=["b", "r"], default="b")
    p.add_argument("--render", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--sleep", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    np.random.seed(args.seed)
    q = load_q_table(Path(args.q_table))
    env = Checkers6x6Env(seed=args.seed)
    agent = AdaptiveQTableAgent(q_table=q, seed=args.seed)

    wins = 0
    losses = 0
    draws = 0

    for i in range(args.episodes):
        winner = run_episode(
            env,
            agent,
            opponent=args.opponent,
            agent_player=args.agent_color,
            render=args.render,
            sleep=args.sleep,
        )
        if winner == args.agent_color:
            wins += 1
        elif winner in ("b", "r"):
            losses += 1
        else:
            draws += 1
        print(f"Episode {i+1}/{args.episodes} | winner={winner}")

    print(
        f"Results vs {args.opponent}: wins={wins} ({wins/args.episodes:.3f}), "
        f"losses={losses} ({losses/args.episodes:.3f}), draws={draws} ({draws/args.episodes:.3f})"
    )


if __name__ == "__main__":
    main()
