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
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module {module_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _to_runner_metrics(stats_path: Path, out_path: Path) -> None:
    data = np.load(stats_path, allow_pickle=True)
    np.savez_compressed(
        out_path,
        rewards=np.asarray(data["rewards"], dtype=np.float64),
        winners=np.asarray(data["winners"], dtype=np.int8),
        episode_lengths=np.asarray(data["episode_lengths"], dtype=np.int32),
        num_episodes=int(np.asarray(data["num_episodes"]).item()),
        eval_win_random=np.asarray(data["eval_win_random"], dtype=np.float64),
        eval_win_heuristic=np.asarray(data["eval_win_heuristic"], dtype=np.float64),
        eval_win_black_heuristic=np.asarray(data["eval_win_p1_heuristic"], dtype=np.float64),
        eval_win_red_heuristic=np.asarray(data["eval_win_p2_heuristic"], dtype=np.float64),
        q_table_sizes=np.asarray(data["q_table_sizes"], dtype=np.int32),
    )


def _pkl_to_npy(q_pkl: Path, q_npy: Path) -> None:
    with q_pkl.open("rb") as f:
        q_table = pickle.load(f)
    arr = np.array(list(q_table.items()), dtype=object)
    np.save(q_npy, arr, allow_pickle=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train extended RL using Chandan core pipeline")
    p.add_argument("--episodes", type=int, default=100000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--eval-interval", type=int, default=1000)
    p.add_argument("--eval-games", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="experiments/results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    core_train = _load_module("chandan_train", CORE_DIR / "train.py")

    # Keep CLI compatibility: these two args are fixed in original Chandan training loop.
    if args.eval_interval != 1000 or args.eval_games != 80:
        print("[note] Chandan training keeps internal eval settings; --eval-interval/--eval-games are ignored.")

    # Override Chandan's internal fixed seed call to honor runner --seed.
    original_set_seed = core_train.set_seed

    def _set_seed_override(_seed: int = 42):
        return original_set_seed(args.seed)

    core_train.set_seed = _set_seed_override
    core_train.train(num_episodes=args.episodes, gamma=args.gamma)

    core_q = CORE_DIR / "q_table.pkl"
    core_stats = CORE_DIR / "training_stats.npz"
    if not core_q.exists() or not core_stats.exists():
        raise FileNotFoundError("Chandan core training did not produce expected artifacts.")

    shutil.copy2(core_q, out_dir / "q_table.pkl")
    shutil.copy2(core_stats, out_dir / "training_stats.npz")
    _pkl_to_npy(out_dir / "q_table.pkl", out_dir / "q_table.npy")
    _to_runner_metrics(out_dir / "training_stats.npz", out_dir / "training_metrics.npz")

    print(f"Saved Chandan-compatible artifacts to {out_dir}")


if __name__ == "__main__":
    main()
