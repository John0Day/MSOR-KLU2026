from __future__ import annotations

import numpy as np


class RandomAgent:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def select_move_index(self, legal_n: int) -> int:
        if legal_n == 0:
            return 0
        return int(self.rng.integers(0, legal_n))

    def select_action(self, env) -> int:
        return self.select_move_index(len(env.legal_moves))
