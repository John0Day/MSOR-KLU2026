"""Baseline agent that chooses uniformly among legal moves."""

from __future__ import annotations

import numpy as np


class RandomAgent:
    """Stateless random policy used for sanity checks and comparisons."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def select_move_index(self, legal_n: int) -> int:
        """Return a random index in ``[0, legal_n)`` (0 when no moves exist)."""

        if legal_n == 0:
            return 0
        return int(self.rng.integers(0, legal_n))

    def select_action(self, env) -> int:
        """Read the number of legal moves from ``env`` and sample uniformly."""

        return self.select_move_index(len(env.legal_moves))
