"""Wrapper that proxies extended evaluation into the provided core module."""

from __future__ import annotations

import argparse
import importlib.util
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = ROOT / "chandan_core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))


def _load_module(module_name: str, path: Path):
    """Dynamically import ``module_name`` from ``path``."""

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module {module_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ensure_core_qtable(q_table_path: Path, core_q_path: Path) -> None:
    """Convert ``.npy`` tables to pickle so the core code can read them."""

    if q_table_path.suffix == ".pkl":
        shutil.copy2(q_table_path, core_q_path)
        return
    items = np.load(q_table_path, allow_pickle=True)
    q_table = dict(items.tolist())
    with core_q_path.open("wb") as f:
        pickle.dump(q_table, f)


def parse_args() -> argparse.Namespace:
    """Parse CLI options for the extended evaluator."""

    p = argparse.ArgumentParser(description="Play/evaluate using Chandan core policy")
    p.add_argument("--q-table", type=str, default="experiments/results/q_table.pkl")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--opponent", choices=["random", "heuristic"], default="random")
    p.add_argument("--agent-color", choices=["b", "r"], default="b")
    p.add_argument("--render", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--sleep", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    """Load the core module and report win/loss/draw statistics."""

    args = parse_args()
    _ensure_core_qtable(Path(args.q_table), CORE_DIR / "q_table.pkl")

    play_mod = _load_module("chandan_play", CORE_DIR / "play.py")
    env = play_mod.make_env()
    agent = play_mod.load_agent(env)
    agent_player_id = 0 if args.agent_color == "b" else 1

    wins = losses = draws = 0
    for i in range(args.episodes):
        winner = play_mod.run_episode_play(
            env,
            agent,
            opponent_type=args.opponent,
            render=args.render,
            sleep_sec=args.sleep,
            seed=args.seed + i,
            agent_player_id=agent_player_id,
        )
        if winner == -1:
            draws += 1
        elif winner == agent_player_id:
            wins += 1
        else:
            losses += 1

    n = max(1, args.episodes)
    print(
        f"Results vs {args.opponent} over {args.episodes} episodes: "
        f"win={wins/n:.3f}, loss={losses/n:.3f}, draw={draws/n:.3f}"
    )


if __name__ == "__main__":
    main()
