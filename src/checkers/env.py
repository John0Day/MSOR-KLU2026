from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .core import BOARD_SIZE, Move, all_legal_moves, apply_move, create_board


PIECE_TO_INT = {
    ".": 0,
    "b": 1,
    "B": 2,
    "r": 3,
    "R": 4,
}

INT_TO_PIECE = {v: k for k, v in PIECE_TO_INT.items()}


class Checkers6x6Env(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        render_mode: str | None = None,
        seed: int | None = 0,
        max_moves: int = 64,
        max_turns: int = 200,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.max_moves = max_moves
        self.max_turns = max_turns

        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(low=0, high=4, shape=(BOARD_SIZE, BOARD_SIZE), dtype=np.int8),
                "player_to_move": spaces.Discrete(2),  # 0=black, 1=red
            }
        )
        self.action_space = spaces.Discrete(max_moves)

        self._rng = np.random.default_rng(seed)
        self.board: list[list[str]] = create_board()
        self.player: str = "b"
        self.forced_piece: tuple[int, int] | None = None
        self.legal_moves: list[Move] = []
        self._move_count = 0

    def _encode_board(self) -> np.ndarray:
        arr = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                arr[r, c] = PIECE_TO_INT[self.board[r][c]]
        return arr

    def _obs(self) -> dict[str, Any]:
        return {
            "board": self._encode_board(),
            "player_to_move": 0 if self.player == "b" else 1,
        }

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(self.max_moves, dtype=np.int8)
        legal = min(len(self.legal_moves), self.max_moves)
        mask[:legal] = 1
        return mask

    def _refresh_legal_moves(self) -> None:
        self.legal_moves = all_legal_moves(self.board, self.player, self.forced_piece)

    def _winner_if_terminal(self) -> str | None:
        if self.legal_moves:
            return None
        return "r" if self.player == "b" else "b"

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.board = create_board()
        self.player = "b"
        self.forced_piece = None
        self._move_count = 0
        self._refresh_legal_moves()
        return self._obs(), {"action_mask": self.action_mask()}

    def step(self, action: int):
        self._move_count += 1

        if self._move_count > self.max_turns:
            return self._obs(), 0.0, False, True, {
                "winner": "draw",
                "action_mask": self.action_mask(),
            }

        if not self.legal_moves:
            winner = self._winner_if_terminal()
            return self._obs(), -1.0, True, False, {
                "winner": winner,
                "invalid_action": True,
                "action_mask": self.action_mask(),
            }

        if action < 0 or action >= len(self.legal_moves) or action >= self.max_moves:
            winner = "r" if self.player == "b" else "b"
            return self._obs(), -1.0, True, False, {
                "winner": winner,
                "invalid_action": True,
                "action_mask": self.action_mask(),
            }

        move = self.legal_moves[action]
        was_capture, was_promoted = apply_move(self.board, move)

        if was_capture and not was_promoted:
            next_caps = [
                m
                for m in all_legal_moves(self.board, self.player, forced_from=(move.to_row, move.to_col))
                if m.captured is not None
            ]
            if next_caps:
                self.forced_piece = (move.to_row, move.to_col)
                self.legal_moves = next_caps
                return self._obs(), 0.0, False, False, {"action_mask": self.action_mask()}

        self.forced_piece = None
        self.player = "r" if self.player == "b" else "b"
        self._refresh_legal_moves()

        if not self.legal_moves:
            return self._obs(), 1.0, True, False, {
                "winner": "r" if self.player == "b" else "b",
                "action_mask": self.action_mask(),
            }

        return self._obs(), 0.0, False, False, {"action_mask": self.action_mask()}

    def render(self):
        print("\n  a b c d e f")
        for r in range(BOARD_SIZE):
            row_label = BOARD_SIZE - r
            pieces = [self.board[r][c] for c in range(BOARD_SIZE)]
            print(f"{row_label} " + " ".join(pieces))
        print(f"Player to move: {'Black' if self.player == 'b' else 'Red'}")
