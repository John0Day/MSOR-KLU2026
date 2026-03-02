from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce full pipeline: training + evaluation")
    p.add_argument("--episodes", type=int, default=8000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--opponent", choices=["random", "heuristic"], default="heuristic")
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--num-seeds", type=int, default=5)
    p.add_argument("--alternate-start", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--out", type=str, default="experiments/results_final")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    out = Path(args.out)

    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", "/tmp")

    train_cmd = [
        sys.executable,
        str(root / "experiments" / "train_q_learning.py"),
        "--episodes",
        str(args.episodes),
        "--seed",
        str(args.seed),
        "--opponent",
        args.opponent,
        "--out",
        str(out),
    ]

    eval_cmd = [
        sys.executable,
        str(root / "experiments" / "evaluate_agents.py"),
        "--q-table",
        str(out / "q_table.npy"),
        "--games",
        str(args.games),
        "--seed",
        str(args.seed),
        "--num-seeds",
        str(args.num_seeds),
        "--out",
        str(out),
    ]
    if args.alternate_start:
        eval_cmd.append("--alternate-start")
    else:
        eval_cmd.append("--no-alternate-start")

    print("Running reproducible pipeline")
    print(f"Output directory: {out}")

    subprocess.run(train_cmd, cwd=root, env=env, check=True)
    subprocess.run(eval_cmd, cwd=root, env=env, check=True)

    print("\nPipeline complete. Key artifacts:")
    for name in [
        "q_table.npy",
        "training_metrics.npz",
        "reward_curve.png",
        "episode_length_curve.png",
        "winrate_over_training.png",
        "head_to_head_winrates.png",
        "evaluation_summary.json",
    ]:
        pth = out / name
        print(f"- {pth} {'OK' if pth.exists() else 'MISSING'}")


if __name__ == "__main__":
    main()
