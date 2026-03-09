from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def moving_avg(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return values.astype(np.float64)
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(values.astype(np.float64), kernel, mode="valid")


def plot_learning_curve(data: dict[str, np.ndarray], out_dir: Path, window: int) -> None:
    rewards = data["rewards"]
    winners = data["winners"]
    eval_random = data.get("eval_win_random", np.array([], dtype=np.float64))
    eval_heur = data.get("eval_win_heuristic", np.array([], dtype=np.float64))

    if winners.dtype.kind in {"U", "S", "O"}:
        win_flags = (winners == "b").astype(np.float64)
    else:
        win_flags = (winners == 0).astype(np.float64)

    train_win_ma = moving_avg(win_flags, window)
    x_train = np.arange(len(train_win_ma)) + max(1, window)
    x_eval = (np.arange(len(eval_random)) + 1) * 1000

    plt.figure(figsize=(11, 6))
    plt.plot(x_train, train_win_ma, label="Training win rate (MA)", alpha=0.45)
    if len(eval_random) > 0:
        plt.plot(x_eval, eval_random, label="Eval win rate vs random", linewidth=2)
    if len(eval_heur) > 0:
        plt.plot(x_eval, eval_heur, label="Eval win rate vs heuristic", linewidth=2)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Episode")
    plt.ylabel("Win rate")
    plt.title("Training vs Evaluation Win Rate")
    plt.grid(alpha=0.25)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "learning_curve_win_rate.png", dpi=160)
    plt.close()

    reward_ma = moving_avg(rewards, window)
    x_reward = np.arange(len(reward_ma)) + max(1, window)
    plt.figure(figsize=(11, 5))
    plt.plot(x_reward, reward_ma)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward Moving Average")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "reward_curve_extended.png", dpi=160)
    plt.close()


def plot_state_space_growth(data: dict[str, np.ndarray], out_dir: Path) -> None:
    sizes = data.get("q_table_sizes", np.array([], dtype=np.int32))
    if len(sizes) == 0:
        return
    x = (np.arange(len(sizes)) + 1) * 1000
    plt.figure(figsize=(10, 5))
    plt.plot(x, sizes, marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Q-table entries")
    plt.title("Q-table Growth")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "state_space_growth.png", dpi=160)
    plt.close()


def plot_game_length(data: dict[str, np.ndarray], out_dir: Path, window: int) -> None:
    lengths = data["episode_lengths"].astype(np.float64)
    length_ma = moving_avg(lengths, window)
    x = np.arange(len(length_ma)) + max(1, window)
    plt.figure(figsize=(10, 5))
    plt.plot(x, length_ma, color="tab:green")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title("Game Length (Moving Average)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "game_length.png", dpi=160)
    plt.close()


def plot_color_split(data: dict[str, np.ndarray], out_dir: Path) -> None:
    wr_b = data.get("eval_win_black_heuristic", np.array([], dtype=np.float64))
    wr_r = data.get("eval_win_red_heuristic", np.array([], dtype=np.float64))
    if len(wr_b) == 0 or len(wr_r) == 0:
        return
    x = (np.arange(len(wr_b)) + 1) * 1000
    plt.figure(figsize=(10, 5))
    plt.plot(x, wr_b, label="As black vs heuristic")
    plt.plot(x, wr_r, label="As red vs heuristic")
    plt.ylim(0.0, 1.0)
    plt.xlabel("Episode")
    plt.ylabel("Win rate")
    plt.title("Evaluation Split by Color")
    plt.grid(alpha=0.25)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "eval_black_vs_red_win_rates.png", dpi=160)
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Extended plots from training metrics")
    p.add_argument("--metrics", type=str, default="experiments/results/training_metrics.npz")
    p.add_argument("--out", type=str, default="experiments/results")
    p.add_argument("--window", type=int, default=500)
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = dict(np.load(Path(args.metrics), allow_pickle=True))

    plot_learning_curve(data, out_dir, max(10, args.window))
    plot_state_space_growth(data, out_dir)
    plot_game_length(data, out_dir, max(10, args.window))
    plot_color_split(data, out_dir)
    print(f"Saved extended plots to {out_dir}")


if __name__ == "__main__":
    main()
