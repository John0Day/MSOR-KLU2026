#!/usr/bin/env python3
"""
Evaluation script: Q-learning agent vs Heuristic agent.

This script:
- Loads the Gym environment
- Loads a trained Q-table
- Plays N matches between Q-learning and heuristic
- Computes win-rate statistics
- Plots results
"""

import numpy as np
import matplotlib.pyplot as plt

from env_checkers import CheckersEnv
from agents import HeuristicAgent
from train_q_learning import state_hash


# -----------------------------
# 1. Q-learning Agent Wrapper
# -----------------------------
class QLearningAgent:
    def __init__(self, q_table):
        self.q = q_table

    def select_action(self, env):
        s = state_hash(env._obs())
        legal_n = len(env.legal_moves)

        if legal_n == 0:
            return 0

        # Choose action with highest Q-value
        vals = [self.q.get((s, a_idx), 0.0) for a_idx in range(legal_n)]
        return int(np.argmax(vals))


# -----------------------------
# 2. Play One Game
# -----------------------------
def play_game(q_agent, h_agent, verbose=False):
    env = CheckersEnv()
    obs, _ = env.reset()
    done = False

    while not done:
        if env.player == "b":
            # Q-learning plays Black
            action = q_agent.select_action(env)
        else:
            # Heuristic plays Red
            action = h_agent.select_action(env)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if verbose:
            env.render()

    # Reward is from perspective of the agent that just moved
    # If Q-learning is Black:
    #   reward = +1 → Q-learning wins
    #   reward = -1 → Q-learning loses
    return reward


# -----------------------------
# 3. Evaluation Loop
# -----------------------------
def evaluate(q_table, games=200):
    q_agent = QLearningAgent(q_table)
    h_agent = HeuristicAgent()

    q_wins = 0
    h_wins = 0

    for i in range(games):
        r = play_game(q_agent, h_agent)
        if r > 0:
            q_wins += 1
        else:
            h_wins += 1

        if (i + 1) % 20 == 0:
            print(f"Game {i+1}/{games} completed")

    return q_wins, h_wins


# -----------------------------
# 4. Main Execution
# -----------------------------
if __name__ == "__main__":
    # Load Q-table from training
    # If you saved it: q = np.load("q_table.npy", allow_pickle=True).item()
    # Otherwise, retrain quickly:
    from train_q_learning import train_q_learning
    q, _ = train_q_learning(episodes=2000)

    q_wins, h_wins = evaluate(q, games=200)

    print("\n=== Evaluation Results ===")
    print(f"Q-learning wins:   {q_wins}")
    print(f"Heuristic wins:    {h_wins}")
    print(f"Q-learning win rate: {q_wins / (q_wins + h_wins):.2f}")

    # Bar plot
    plt.figure(figsize=(6, 4))
    plt.bar(["Q-learning", "Heuristic"], [q_wins, h_wins], color=["blue", "red"])
    plt.title("Q-learning vs Heuristic (200 games)")
    plt.ylabel("Wins")
    plt.tight_layout()
    plt.show()
