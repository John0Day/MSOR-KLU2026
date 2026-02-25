#!/usr/bin/env python3
"""
Q-learning training script for 6x6 Checkers.

This script:
- Loads the Gymnasium environment
- Implements tabular Q-learning
- Trains for N episodes
- Plots reward curves
- Saves the Q-table (optional)
"""

import numpy as np
import matplotlib.pyplot as plt
from env_checkers import CheckersEnv


# -----------------------------
# 1. State Hashing
# -----------------------------
def state_hash(obs):
    """
    Convert observation dict into a hashable key:
    - Flattened board (tuple)
    - Player to move (0 or 1)
    """
    b = tuple(obs["board"].flatten().tolist())
    return (b, int(obs["player"]))


# -----------------------------
# 2. Q-learning Training Loop
# -----------------------------
def train_q_learning(
    episodes=3000,
    alpha=0.1,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=0.999
):
    env = CheckersEnv()
    q = {}  # Q-table: dict[(state, action)] = value
    rewards_per_ep = []

    epsilon = epsilon_start

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            s = state_hash(obs)
            legal_n = len(env.legal_moves)

            if legal_n == 0:
                break

            # ε-greedy action selection
            if np.random.rand() < epsilon:
                a = np.random.randint(0, legal_n)
            else:
                vals = [q.get((s, a_idx), 0.0) for a_idx in range(legal_n)]
                a = int(np.argmax(vals))

            next_obs, r, terminated, truncated, info = env.step(a)
            total_reward += r
            s2 = state_hash(next_obs)

            # Q-learning update
            if terminated or truncated:
                target = r
            else:
                next_legal_n = len(env.legal_moves)
                next_vals = [q.get((s2, a_idx), 0.0) for a_idx in range(next_legal_n)]
                # CRITICAL FIX: Zero-sum update.
                # If the next state is good for the opponent (max(next_vals) is high),
                # it is BAD for the current player. We subtract the opponent's best value.
                target = r - gamma * (max(next_vals) if next_vals else 0.0)

            old = q.get((s, a), 0.0)
            q[(s, a)] = old + alpha * (target - old)

            obs = next_obs
            done = terminated or truncated

        rewards_per_ep.append(total_reward)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if (ep + 1) % 500 == 0:
            print(f"Episode {ep+1}/{episodes}, reward={total_reward:.2f}, epsilon={epsilon:.3f}")

    return q, rewards_per_ep


# -----------------------------
# 3. Main Training Execution
# -----------------------------
if __name__ == "__main__":
    np.random.seed(0)

    q, rewards = train_q_learning(episodes=3000)

    # Moving average for smoother visualization
    window = 50
    ma = np.convolve(rewards, np.ones(window)/window, mode="valid")

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.3, label="Episode reward")
    plt.plot(range(window-1, len(rewards)), ma, label=f"{window}-episode moving average")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Q-learning Training Rewards")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Optional: save Q-table
    # np.save("q_table.npy", q)
