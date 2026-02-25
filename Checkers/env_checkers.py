#!/usr/bin/env python3
"""
Gymnasium environment for 6x6 Checkers.

This wraps the pure logic from game_core.py into a reinforcement-learning
compatible environment with:
- reset()
- step()
- observation_space
- action_space
- reproducible seeds
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple

from game_core import (
    BOARD_SIZE, create_board, all_legal_moves, apply_move
)


class CheckersEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode: Optional[str] = None, seed: int = 0):
        super().__init__()
        self.render_mode = render_mode

        # Observation: board (6x6) + player-to-move
        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=-2, high=2, shape=(BOARD_SIZE, BOARD_SIZE), dtype=np.int8),
            "player": spaces.Discrete(2)  # 0 = black, 1 = red
        })

        # Action: index into legal move list (max 40 moves)
        self.max_moves = 40
        self.action_space = spaces.Discrete(self.max_moves)

        # Internal state
        self._rng = np.random.default_rng(seed)
        self.board = None
        self.player = None
        self.forced_piece: Optional[Tuple[int, int]] = None
        self.legal_moves = []

    # ------------------ Helpers ------------------

    def _encode_board(self):
        """
        Convert board symbols to integers:
        . → 0
        b → 1
        B → 2
        r → -1
        R → -2
        """
        mapping = {".": 0, "b": 1, "B": 2, "r": -1, "R": -2}
        arr = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                arr[r, c] = mapping[self.board[r][c]]
        return arr

    def _obs(self):
        return {
            "board": self._encode_board(),
            "player": 0 if self.player == "b" else 1
        }

    # ------------------ Gym API ------------------

    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.board = create_board()
        self.player = "b"
        self.forced_piece = None
        self.legal_moves = all_legal_moves(self.board, self.player, self.forced_piece)

        return self._obs(), {}

    def step(self, action: int):
        """
        Executes one action index from the legal move list.
        Returns:
            obs, reward, terminated, truncated, info
        """

        # No legal moves → current player loses
        if not self.legal_moves:
            return self._obs(), -1.0, True, False, {}

        # Invalid action index
        if action >= len(self.legal_moves):
            return self._obs(), -1.0, True, False, {"info": "invalid_action"}

        move = self.legal_moves[action]
        was_capture, was_promoted = apply_move(self.board, move)

        # Multi-jump continuation
        if was_capture and not was_promoted:
            next_caps = [
                m for m in all_legal_moves(
                    self.board, self.player,
                    forced_from=(move.to_row, move.to_col)
                )
                if m.captured
            ]
            if next_caps:
                self.forced_piece = (move.to_row, move.to_col)
                self.legal_moves = next_caps
                return self._obs(), 0.0, False, False, {}

        # Normal turn switch
        self.forced_piece = None
        self.player = "r" if self.player == "b" else "b"
        self.legal_moves = all_legal_moves(self.board, self.player, self.forced_piece)

        # Terminal check
        if not self.legal_moves:
            return self._obs(), 1.0, True, False, {}

        return self._obs(), 0.0, False, False, {}

    # ------------------ Rendering ------------------

    def render(self):
        """
        Simple text-based rendering.
        """
        arr = self._encode_board()
        symbols = {0: ".", 1: "b", 2: "B", -1: "r", -2: "R"}

        print("\n  a b c d e f")
        for r in range(BOARD_SIZE):
            row_label = BOARD_SIZE - r
            row_syms = [symbols[arr[r, c]] for c in range(BOARD_SIZE)]
            print(f"{row_label} " + " ".join(row_syms))

        print(f"Player to move: {'Black' if self.player == 'b' else 'Red'}")
