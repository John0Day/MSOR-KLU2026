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


def _ensure_chandan_stats(metrics_path: Path, core_stats_path: Path) -> None:
    data = np.load(metrics_path, allow_pickle=True)
    if "eval_win_p1_heuristic" in data.files and "eval_win_p2_heuristic" in data.files:
        shutil.copy2(metrics_path, core_stats_path)
        return

    np.savez_compressed(
        core_stats_path,
        rewards=np.asarray(data["rewards"], dtype=np.float32),
        winners=np.asarray(data["winners"], dtype=np.int8),
        episode_lengths=np.asarray(data["episode_lengths"], dtype=np.int32),
        num_episodes=int(np.asarray(data["num_episodes"]).item()),
        eval_win_random=np.asarray(data["eval_win_random"], dtype=np.float32),
        eval_win_heuristic=np.asarray(data["eval_win_heuristic"], dtype=np.float32),
        eval_win_p1_heuristic=np.asarray(data.get("eval_win_black_heuristic", np.array([])), dtype=np.float32),
        eval_win_p2_heuristic=np.asarray(data.get("eval_win_red_heuristic", np.array([])), dtype=np.float32),
        q_table_sizes=np.asarray(data["q_table_sizes"], dtype=np.int32),
    )


def _ensure_chandan_qtable(q_table_path: Path, core_q_path: Path) -> None:
    if q_table_path.suffix == ".pkl":
        shutil.copy2(q_table_path, core_q_path)
        return

    items = np.load(q_table_path, allow_pickle=True)
    q_table = dict(items.tolist())
    with core_q_path.open("wb") as f:
        pickle.dump(q_table, f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Chandan 1:1 plots")
    p.add_argument("--metrics", type=str, default="experiments/results/training_metrics.npz")
    p.add_argument("--out", type=str, default="experiments/results")
    p.add_argument("--window", type=int, default=1000)
    p.add_argument("--q-table", type=str, default="experiments/results/q_table.npy")
    p.add_argument("--perf-games", type=int, default=1000)
    p.add_argument("--perf-seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    core_stats = CORE_DIR / "training_stats.npz"
    core_q = CORE_DIR / "q_table.pkl"

    _ensure_chandan_stats(Path(args.metrics), core_stats)
    _ensure_chandan_qtable(Path(args.q_table), core_q)

    core_plots = _load_module("chandan_plots", CORE_DIR / "plots.py")
    core_plots.plot_learning_curve(window=max(10, args.window))
    try:
        core_plots.plot_state_space_growth()
    except ZeroDivisionError:
        print("[note] Skipped state-space growth plot (no evaluation checkpoints in metrics).")
    core_plots.plot_game_length(window=max(10, args.window))
    try:
        core_plots.plot_p1_vs_p2_eval()
    except Exception:
        print("[note] Skipped P1/P2 eval split plot (required eval keys unavailable).")
    core_plots.performance_distribution(num_games=max(10, args.perf_games))

    for name in [
        "learning_curve_win_rate.png",
        "state_space_growth.png",
        "game_length.png",
        "eval_p1_vs_p2_win_rates.png",
        "performance_distribution.png",
    ]:
        src = CORE_DIR / name
        if src.exists():
            shutil.copy2(src, out_dir / name)

    # Keep compatibility aliases used elsewhere in this repo.
    if (out_dir / "eval_p1_vs_p2_win_rates.png").exists():
        shutil.copy2(out_dir / "eval_p1_vs_p2_win_rates.png", out_dir / "eval_black_vs_red_win_rates.png")
    if (out_dir / "performance_distribution.png").exists():
        shutil.copy2(out_dir / "performance_distribution.png", out_dir / "performance_distribution_vs_heuristic.png")

    print(f"Saved extended plots to {out_dir}")


if __name__ == "__main__":
    main()
