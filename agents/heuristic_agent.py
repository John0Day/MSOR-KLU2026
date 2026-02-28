from __future__ import annotations

from dataclasses import dataclass

from src.checkers.core import BOARD_SIZE, Move, all_legal_moves, apply_move, clone_board


@dataclass
class Weights:
    material: float = 1.0
    mobility: float = 0.15
    advancement: float = 0.08


def _material_score(board: list[list[str]], player: str) -> float:
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
    opp = "r" if player == "b" else "b"
    return float(len(all_legal_moves(board, player)) - len(all_legal_moves(board, opp)))


def _advancement_score(board: list[list[str]], player: str) -> float:
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
    w = weights or Weights()
    return (
        w.material * _material_score(board, player)
        + w.mobility * _mobility_score(board, player)
        + w.advancement * _advancement_score(board, player)
    )


def _immediate_counter_captures(board: list[list[str]], player: str) -> int:
    opp = "r" if player == "b" else "b"
    return sum(1 for m in all_legal_moves(board, opp) if m.captured is not None)


class HeuristicAgent:
    def __init__(self, weights: Weights | None = None):
        self.weights = weights or Weights()

    def select_move_index(self, board: list[list[str]], player: str, legal_moves: list[Move]) -> int:
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
        return self.select_move_index(env.board, env.player, env.legal_moves)
