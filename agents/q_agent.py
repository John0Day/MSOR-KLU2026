from __future__ import annotations

import numpy as np

from src.checkers.core import BOARD_SIZE, Move

def state_hash(obs: dict) -> tuple[tuple[int, ...], int]:
    board = tuple(obs["board"].reshape(-1).tolist())
    return board, int(obs["player_to_move"])


def encode_board_state(board: list[list[str]]) -> tuple[int, ...]:
    mapping = {".": 0, "b": 1, "B": 2, "r": 3, "R": 4}
    out: list[int] = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            out.append(mapping[board[r][c]])
    return tuple(out)


class QTableAgent:
    def __init__(self, q_table: dict[tuple[tuple[tuple[int, ...], int], int], float], epsilon: float = 0.0, seed: int = 0):
        self.q = q_table
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

    def select_move_index(self, board: list[list[str]], player: str, legal_moves: list[Move]) -> int:
        legal_n = len(legal_moves)
        if legal_n == 0:
            return 0

        player_to_move = 0 if player == "b" else 1
        s = (encode_board_state(board), player_to_move)

        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, legal_n))

        values = [self.q.get((s, a), 0.0) for a in range(legal_n)]
        return int(np.argmax(values))

    def select_action(self, env) -> int:
        return self.select_move_index(env.board, env.player, env.legal_moves)
