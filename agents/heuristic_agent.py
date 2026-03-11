"""Heuristic-based agents for the 6x6 checkers environment."""

from __future__ import annotations

import random
from dataclasses import dataclass

from src.checkers.core import BOARD_SIZE, Move, all_legal_moves, apply_move, clone_board


@dataclass
class Weights:
    """Grouping of evaluation weights used by :func:`evaluate_board`."""

    material: float = 1.0
    mobility: float = 0.15
    advancement: float = 0.08


def _material_score(board: list[list[str]], player: str) -> float:
    """Return signed material advantage for ``player``."""

    value = {"b": 1.0, "B": 2.0, "r": 1.0, "R": 2.0}
    me = 0.0
    opp = 0.0
    for row in board:
        for p in row:
            if p == ".":
                continue
            if (player == "b" and p in ("b", "B")) or (player == "r" and p in ("r", "R")):
                me += value[p]
            else:
                opp += value[p]
    return me - opp


def _mobility_score(board: list[list[str]], player: str) -> float:
    """Compare how many legal moves each player currently has."""

    opp = "r" if player == "b" else "b"
    return float(len(all_legal_moves(board, player)) - len(all_legal_moves(board, opp)))


def _advancement_score(board: list[list[str]], player: str) -> float:
    """Favor pieces that have progressed toward promotion."""

    score = 0.0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            p = board[r][c]
            if p == "b":
                score += r / (BOARD_SIZE - 1)
            elif p == "r":
                score += (BOARD_SIZE - 1 - r) / (BOARD_SIZE - 1)
    return score if player == "b" else -score


def evaluate_board(board: list[list[str]], player: str, weights: Weights | None = None) -> float:
    """Weighted sum of material, mobility, and advancement signals."""

    w = weights or Weights()
    return (
        w.material * _material_score(board, player)
        + w.mobility * _mobility_score(board, player)
        + w.advancement * _advancement_score(board, player)
    )


def _immediate_counter_captures(board: list[list[str]], player: str) -> int:
    """Count opponent replies that would capture immediately after a move."""

    opp = "r" if player == "b" else "b"
    return sum(1 for m in all_legal_moves(board, opp) if m.captured is not None)


class HeuristicAgent:
    """Greedy move selector driven by static board evaluation heuristics."""

    def __init__(self, weights: Weights | None = None):
        self.weights = weights or Weights()

    def select_move_index(self, board: list[list[str]], player: str, legal_moves: list[Move]) -> int:
        """Return the index of the highest-scoring move under the heuristics."""

        if not legal_moves:
            return 0

        capture_moves = [m for m in legal_moves if m.captured is not None]
        candidates = capture_moves if capture_moves else legal_moves

        best_idx = 0
        best_key = (-10**9, -10**9)

        for idx, move in enumerate(legal_moves):
            if move not in candidates:
                continue

            nxt = clone_board(board)
            apply_move(nxt, move)
            immediate_risk = -_immediate_counter_captures(nxt, player)
            eval_score = evaluate_board(nxt, player, self.weights)
            key = (immediate_risk, eval_score)
            if key > best_key:
                best_key = key
                best_idx = idx

        return best_idx

    def select_action(self, env) -> int:
        """Adapter that expects a gym-style env with ``board``/``legal_moves``."""

        return self.select_move_index(env.board, env.player, env.legal_moves)


class PriorityHeuristicAgent:
    """
    Priority-based heuristic (capture > promotion > edge > center > random).
    """

    def __init__(self, player: str):
        self.player = player

    def _is_promotion(self, board: list[list[str]], move: Move) -> bool:
        """True if ``move`` promotes one of the agent's men to a king."""

        piece = board[move.from_row][move.from_col]
        if self.player == "b" and piece == "b" and move.to_row == BOARD_SIZE - 1:
            return True
        if self.player == "r" and piece == "r" and move.to_row == 0:
            return True
        return False

    def _center_score(self, move: Move) -> float:
        """Heuristic rewarding moves that finish nearer the board center."""

        center = (BOARD_SIZE - 1) / 2.0
        dr = move.to_row - center
        dc = move.to_col - center
        return float((BOARD_SIZE - 1) - (dr * dr + dc * dc) ** 0.5)

    def select_move_index(self, board: list[list[str]], legal_moves: list[Move]) -> int:
        """Pick a move by following capture/promotion/positional priorities."""

        if not legal_moves:
            return 0
        capture_moves = [m for m in legal_moves if m.captured is not None]
        if capture_moves:
            chosen = random.choice(capture_moves)
            return legal_moves.index(chosen)

        promo_moves = [m for m in legal_moves if self._is_promotion(board, m)]
        if promo_moves:
            chosen = random.choice(promo_moves)
            return legal_moves.index(chosen)

        edge_moves = [m for m in legal_moves if m.to_col in (0, BOARD_SIZE - 1)]
        if edge_moves:
            chosen = random.choice(edge_moves)
            return legal_moves.index(chosen)

        scored = sorted(enumerate(legal_moves), key=lambda x: self._center_score(x[1]), reverse=True)
        if scored and self._center_score(scored[0][1]) > 0:
            best = self._center_score(scored[0][1])
            top = [i for i, m in scored if self._center_score(m) == best]
            return random.choice(top)
        return int(random.randrange(len(legal_moves)))

    def select_action(self, env) -> int:
        """Adapter that reads ``board`` and ``legal_moves`` from ``env``."""

        return self.select_move_index(env.board, env.legal_moves)
