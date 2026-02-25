#!/usr/bin/env python3
"""
Heuristic agents for 6x6 Checkers.

This file contains:
- A static evaluation function (material-based)
- A rule-based move selector
- A Gym-compatible agent wrapper
"""

from copy import deepcopy
from game_core import (
    BOARD_SIZE, all_legal_moves, apply_move
)


# -----------------------------
# 1. Board Evaluation Function
# -----------------------------
def evaluate_board(board):
    """
    Simple material-based evaluation:
    - Black men: +1
    - Black kings: +2
    - Red men: -1
    - Red kings: -2

    Positive score favors Black.
    Negative score favors Red.
    """
    score = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            p = board[r][c]
            if p == "b":
                score += 1
            elif p == "B":
                score += 2
            elif p == "r":
                score -= 1
            elif p == "R":
                score -= 2
    return score


# -----------------------------
# 2. Heuristic Move Selector
# -----------------------------
def heuristic_move(board, player, forced_piece=None):
    """
    Chooses the move that maximizes the evaluation function
    (for Black) or minimizes it (for Red).

    This is your rule-based strategy required by the assignment.
    """
    moves = all_legal_moves(board, player, forced_piece)
    if not moves:
        return None

    best_move = None

    if player == "b":
        best_score = -1e9
        for m in moves:
            tmp = deepcopy(board)
            apply_move(tmp, m)
            s = evaluate_board(tmp)
            if s > best_score:
                best_score = s
                best_move = m
    else:
        best_score = 1e9
        for m in moves:
            tmp = deepcopy(board)
            apply_move(tmp, m)
            s = evaluate_board(tmp)
            if s < best_score:
                best_score = s
                best_move = m

    return best_move


# ---------------------------------------
# 3. Gym-compatible Heuristic Agent Class
# ---------------------------------------
class HeuristicAgent:
    """
    A simple agent that can play inside the Gym environment.

    Usage:
        agent = HeuristicAgent()
        action = agent.select_action(env)
    """

    def select_action(self, env):
        """
        Given a Gym environment (CheckersEnv),
        choose the best legal move index.
        """
        moves = env.legal_moves
        if not moves:
            return 0  # no legal moves → environment will handle loss

        # Evaluate each successor state
        best_idx = 0
        best_score = -1e9 if env.player == "b" else 1e9

        for i, m in enumerate(moves):
            tmp = deepcopy(env.board)
            apply_move(tmp, m)
            s = evaluate_board(tmp)

            if env.player == "b":
                if s > best_score:
                    best_score = s
                    best_idx = i
            else:
                if s < best_score:
                    best_score = s
                    best_idx = i

        return best_idx
