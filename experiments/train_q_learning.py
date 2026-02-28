from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents import HeuristicAgent, QTableAgent, RandomAgent, state_hash
from env import Checkers6x6Env


@dataclass
class TrainConfig:
    episodes: int = 8000
    alpha: float = 0.15
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.9993
    eval_interval: int = 250
    eval_games: int = 80
    seed: int = 42


def moving_average(values: list[float], window: int) -> np.ndarray:
    if len(values) < window:
        return np.array(values, dtype=np.float64)
    kernel = np.ones(window) / window
    return np.convolve(np.array(values, dtype=np.float64), kernel, mode="valid")


def _greedy_action(q: dict, obs: dict, legal_n: int) -> int:
    if legal_n == 0:
        return 0
    s = state_hash(obs)
    vals = [q.get((s, a), 0.0) for a in range(legal_n)]
    return int(np.argmax(vals))


def play_game(q_agent: QTableAgent, opponent, seed: int) -> tuple[int, int]:
    env = Checkers6x6Env(seed=seed)
    env.reset(seed=seed)
    steps = 0

    while True:
        steps += 1
        if env.player == "b":
            action = q_agent.select_action(env)
            _, _, terminated, truncated, info = env.step(action)
        else:
            action = opponent.select_action(env)
            _, _, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            winner = info.get("winner")
            if winner == "b":
                return 1, steps
            if winner == "r":
                return -1, steps
            return 0, steps


def evaluate_q_agent(q: dict, opponent_name: str, games: int, seed: int) -> float:
    q_agent = QTableAgent(q, epsilon=0.0, seed=seed)
    opponent = RandomAgent(seed=seed + 1) if opponent_name == "random" else HeuristicAgent()

    wins = 0
    for i in range(games):
        result, _ = play_game(q_agent, opponent, seed=seed + 1000 + i)
        if result > 0:
            wins += 1
    return wins / games


def train_q_learning(config: TrainConfig, opponent_name: str = "heuristic"):
    rng = np.random.default_rng(config.seed)
    env = Checkers6x6Env(seed=config.seed)
    q: dict[tuple[tuple[tuple[int, ...], int], int], float] = {}

    opponent = RandomAgent(seed=config.seed + 7) if opponent_name == "random" else HeuristicAgent()

    rewards: list[float] = []
    episode_lengths: list[int] = []
    eval_steps: list[int] = []
    eval_vs_random: list[float] = []
    eval_vs_heuristic: list[float] = []

    epsilon = config.epsilon_start

    for ep in range(config.episodes):
        obs, _ = env.reset(seed=config.seed + ep)
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            if env.player != "b":
                action_opp = opponent.select_action(env)
                obs, r_opp, term, trunc, _ = env.step(action_opp)
                total_reward += -r_opp
                done = term or trunc
                steps += 1
                continue

            s = state_hash(obs)
            legal_n = len(env.legal_moves)
            if legal_n == 0:
                break

            if rng.random() < epsilon:
                a = int(rng.integers(0, legal_n))
            else:
                a = _greedy_action(q, obs, legal_n)

            next_obs, r, term, trunc, _ = env.step(a)
            total_reward += r
            done = term or trunc
            steps += 1

            if done:
                target = r
                q[(s, a)] = q.get((s, a), 0.0) + config.alpha * (target - q.get((s, a), 0.0))
                obs = next_obs
                continue

            terminal_reward = 0.0
            while not done and env.player != "b":
                action_opp = opponent.select_action(env)
                next_obs, r_opp, term, trunc, _ = env.step(action_opp)
                total_reward += -r_opp
                done = term or trunc
                steps += 1
                if done:
                    terminal_reward = -r_opp
                    break

            if done:
                target = terminal_reward
            else:
                next_legal_n = len(env.legal_moves)
                next_best = 0.0
                if next_legal_n > 0:
                    s2 = state_hash(next_obs)
                    next_best = max(q.get((s2, a2), 0.0) for a2 in range(next_legal_n))
                target = r + config.gamma * next_best

            old = q.get((s, a), 0.0)
            q[(s, a)] = old + config.alpha * (target - old)
            obs = next_obs

        rewards.append(total_reward)
        episode_lengths.append(steps)
        epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)

        if (ep + 1) % config.eval_interval == 0:
            e_seed = config.seed + 50000 + ep
            wr_rand = evaluate_q_agent(q, "random", config.eval_games, e_seed)
            wr_heur = evaluate_q_agent(q, "heuristic", config.eval_games, e_seed + 1000)
            eval_steps.append(ep + 1)
            eval_vs_random.append(wr_rand)
            eval_vs_heuristic.append(wr_heur)
            print(
                f"Episode {ep + 1}/{config.episodes} | eps={epsilon:.3f} "
                f"| wr_vs_random={wr_rand:.2f} | wr_vs_heuristic={wr_heur:.2f}"
            )

    return {
        "q": q,
        "rewards": rewards,
        "episode_lengths": episode_lengths,
        "eval_steps": eval_steps,
        "eval_vs_random": eval_vs_random,
        "eval_vs_heuristic": eval_vs_heuristic,
        "config": config,
    }


def save_results(results: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    q = results["q"]
    q_arr = np.array(list(q.items()), dtype=object)
    np.save(out_dir / "q_table.npy", q_arr, allow_pickle=True)

    np.savez(
        out_dir / "training_metrics.npz",
        rewards=np.array(results["rewards"], dtype=np.float64),
        episode_lengths=np.array(results["episode_lengths"], dtype=np.int32),
        eval_steps=np.array(results["eval_steps"], dtype=np.int32),
        eval_vs_random=np.array(results["eval_vs_random"], dtype=np.float64),
        eval_vs_heuristic=np.array(results["eval_vs_heuristic"], dtype=np.float64),
    )


def plot_training(results: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    rewards = results["rewards"]
    lengths = results["episode_lengths"]
    eval_steps = results["eval_steps"]
    eval_vs_random = results["eval_vs_random"]
    eval_vs_heuristic = results["eval_vs_heuristic"]

    reward_ma = moving_average(rewards, window=100)
    length_ma = moving_average(lengths, window=100)

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.2, label="Reward per episode")
    if len(reward_ma) > 0:
        x_ma = np.arange(len(reward_ma)) + max(0, len(rewards) - len(reward_ma))
        plt.plot(x_ma, reward_ma, label="Reward moving average (100)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "reward_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(lengths, alpha=0.25, label="Episode length")
    if len(length_ma) > 0:
        x_ma = np.arange(len(length_ma)) + max(0, len(lengths) - len(length_ma))
        plt.plot(x_ma, length_ma, label="Length moving average (100)")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title("Episode length")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "episode_length_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(eval_steps, eval_vs_random, marker="o", label="Winrate vs Random")
    plt.plot(eval_steps, eval_vs_heuristic, marker="o", label="Winrate vs Heuristic")
    plt.ylim(0.0, 1.0)
    plt.xlabel("Training episode")
    plt.ylabel("Winrate")
    plt.title("Winrate over training time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "winrate_over_training.png", dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tabular Q-learning on 6x6 checkers")
    parser.add_argument("--episodes", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--opponent", choices=["random", "heuristic"], default="heuristic")
    parser.add_argument("--out", type=str, default="experiments/results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(episodes=args.episodes, seed=args.seed)
    results = train_q_learning(cfg, opponent_name=args.opponent)

    out_dir = Path(args.out)
    save_results(results, out_dir)
    plot_training(results, out_dir)

    print(f"Saved outputs to {out_dir}")
    if results["eval_steps"]:
        print(
            f"Final winrates: vs Random={results['eval_vs_random'][-1]:.3f}, "
            f"vs Heuristic={results['eval_vs_heuristic'][-1]:.3f}"
        )


if __name__ == "__main__":
    main()
