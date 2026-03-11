"""Tabular Q-learning agents and helpers for the 6x6 checkers project."""

from __future__ import annotations

import numpy as np

from src.checkers.core import BOARD_SIZE, Move

def state_hash(obs: dict) -> tuple[tuple[int, ...], int]:
    """Hash an observation dict into (board tuple, current-player flag)."""

    board = tuple(obs["board"].reshape(-1).tolist())
    return board, int(obs["player_to_move"])


def encode_board_state(board: list[list[str]]) -> tuple[int, ...]:
    """Map a character board into an integer tuple that is hashable."""

    mapping = {".": 0, "b": 1, "B": 2, "r": 3, "R": 4}
    out: list[int] = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            out.append(mapping[board[r][c]])
    return tuple(out)


class QTableAgent:
    """Simple epsilon-greedy agent backed by a fixed Q-table."""

    def __init__(
        self,
        q_table: dict[tuple[tuple[tuple[int, ...], int], int], float],
        epsilon: float = 0.0,
        seed: int = 0,
    ):
        self.q = q_table
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

    def select_move_index(self, board: list[list[str]], player: str, legal_moves: list[Move]) -> int:
        """Select a move index given a raw board + player turn."""

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
        """Gym-style adapter that queries ``env`` attributes."""

        return self.select_move_index(env.board, env.player, env.legal_moves)


def move_to_action(move: Move) -> tuple[int, int, int, int]:
    """Encode a move so it can be used as a dictionary key."""

    return (move.from_row, move.from_col, move.to_row, move.to_col)


def canonical_state_hash(obs: dict) -> tuple[int, ...]:
    """Return a player-invariant encoding that always treats the mover as black.

    The board is rotated + recolored when red is to move and only playable
    squares are retained so the representation stays compact.
    """

    board = np.array(obs["board"], dtype=np.int8)
    player_to_move = int(obs["player_to_move"])
    if player_to_move == 1:
        board = np.flip(board)
        mapping = np.array([0, 3, 4, 1, 2], dtype=np.int8)
        board = mapping[board]
    playable: list[int] = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if (r + c) % 2 == 1:
                playable.append(int(board[r, c]))
    return tuple(playable)


class AdaptiveQTableAgent:
    """Q-table agent with state-dependent exploration and adaptive updates."""

    def __init__(self, q_table: dict | None = None, seed: int = 0, n0: float = 100.0):
        self.q: dict[tuple[tuple[int, ...], tuple[int, int, int, int]], float] = q_table or {}
        self.rng = np.random.default_rng(seed)
        self.n0 = n0
        self.sa_visits: dict[tuple[tuple[int, ...], tuple[int, int, int, int]], int] = {}
        self.s_visits: dict[tuple[int, ...], int] = {}

    def get_q(self, s: tuple[int, ...], a: tuple[int, int, int, int]) -> float:
        """Return the stored value for ``(s, a)`` (0.0 if unseen)."""

        return self.q.get((s, a), 0.0)

    def _epsilon(self, s: tuple[int, ...], exploit_only: bool) -> float:
        """Decrease exploration with every visit unless ``exploit_only`` is set."""

        if exploit_only:
            return 0.0
        n = self.s_visits.get(s, 0) + 1
        self.s_visits[s] = n
        return float(self.n0 / (self.n0 + n))

    def greedy_action(self, s: tuple[int, ...], legal_moves: list[Move]) -> int:
        """Return the argmax action index for a canonical state hash."""

        if not legal_moves:
            return 0
        values = [self.get_q(s, move_to_action(m)) for m in legal_moves]
        return int(np.argmax(values))

    def select_move_index(self, obs: dict, legal_moves: list[Move], exploit_only: bool = False) -> int:
        """Epsilon-greedy policy over the canonical state hash."""

        if not legal_moves:
            return 0
        s = canonical_state_hash(obs)
        eps = self._epsilon(s, exploit_only=exploit_only)
        if self.rng.random() < eps:
            return int(self.rng.integers(0, len(legal_moves)))
        return self.greedy_action(s, legal_moves)

    def update_q(
        self,
        s: tuple[int, ...],
        action: tuple[int, int, int, int],
        reward: float,
        s_next: tuple[int, ...] | None,
        legal_next_moves: list[Move],
        gamma: float,
    ) -> None:
        """Incrementally update Q(s, a) with a decaying step size."""

        key = (s, action)
        n = self.sa_visits.get(key, 0) + 1
        self.sa_visits[key] = n
        alpha = max(0.005, 1.0 / float(np.sqrt(n)))
        old = self.q.get(key, 0.0)
        next_best = 0.0
        if s_next is not None and legal_next_moves:
            next_best = max(self.get_q(s_next, move_to_action(m)) for m in legal_next_moves)
        target = reward + gamma * next_best
        self.q[key] = old + alpha * (target - old)
