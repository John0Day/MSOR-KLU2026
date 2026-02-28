from __future__ import annotations

import argparse
import sys
import tkinter as tk
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents import HeuristicAgent, QTableAgent, RandomAgent
from src.checkers.core import (
    BOARD_SIZE,
    all_legal_moves,
    apply_move,
    create_board,
    in_bounds,
    owner,
    square_name,
)

SQUARE_SIZE = 90
LIGHT = "#F0D9B5"
DARK = "#B58863"


def load_ai(opponent: str, q_table: str, seed: int):
    if opponent == "random":
        return RandomAgent(seed=seed)
    if opponent == "heuristic":
        return HeuristicAgent()
    q_path = Path(q_table)
    if not q_path.exists():
        raise FileNotFoundError(f"Q-table not found: {q_path}")
    q = dict(np.load(q_path, allow_pickle=True).tolist())
    return QTableAgent(q, epsilon=0.0, seed=seed)


def ai_move_index(ai, board, player, legal_moves):
    if isinstance(ai, HeuristicAgent):
        return ai.select_move_index(board, player, legal_moves)
    if isinstance(ai, QTableAgent):
        return ai.select_move_index(board, player, legal_moves)
    if isinstance(ai, RandomAgent):
        return ai.select_move_index(len(legal_moves))
    return 0


class CheckersHvAIGUI:
    def __init__(self, root: tk.Tk, opponent: str, human_color: str, q_table: str, seed: int):
        self.root = root
        self.human_color = human_color
        self.ai_color = "r" if human_color == "b" else "b"
        self.ai = load_ai(opponent, q_table, seed)
        self.opponent = opponent

        self.root.title(f"6x6 Checkers - Human vs {opponent}")

        self.board = create_board()
        self.player = "b"
        self.forced_piece = None
        self.selected = None
        self.game_over = False

        self.canvas = tk.Canvas(
            root,
            width=BOARD_SIZE * SQUARE_SIZE,
            height=BOARD_SIZE * SQUARE_SIZE,
        )
        self.canvas.pack()

        self.status = tk.Label(root, text="", font=("Arial", 14))
        self.status.pack(pady=5)

        self.canvas.bind("<Button-1>", self.on_click)

        self.render()
        self.update_status()
        self.root.after(150, self.maybe_ai_turn)

    def draw_board(self):
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                color = DARK if (r + c) % 2 else LIGHT
                self.canvas.create_rectangle(
                    c * SQUARE_SIZE,
                    r * SQUARE_SIZE,
                    (c + 1) * SQUARE_SIZE,
                    (r + 1) * SQUARE_SIZE,
                    fill=color,
                    outline="",
                )

    def draw_piece(self, r, c, color, king=False):
        x = c * SQUARE_SIZE + SQUARE_SIZE // 2
        y = r * SQUARE_SIZE + SQUARE_SIZE // 2
        rad = SQUARE_SIZE // 2 - 12

        self.canvas.create_oval(
            x - rad,
            y - rad,
            x + rad,
            y + rad,
            fill=color,
            outline="black",
            width=2,
        )

        if king:
            self.canvas.create_text(x, y, text="K", fill="gold", font=("Arial", 20, "bold"))

    def highlight(self, r, c, color="blue"):
        self.canvas.create_rectangle(
            c * SQUARE_SIZE,
            r * SQUARE_SIZE,
            (c + 1) * SQUARE_SIZE,
            (r + 1) * SQUARE_SIZE,
            outline=color,
            width=3,
        )

    def render(self):
        self.canvas.delete("all")
        self.draw_board()

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                p = self.board[r][c]
                if p in ("b", "B"):
                    self.draw_piece(r, c, "black", king=(p == "B"))
                elif p in ("r", "R"):
                    self.draw_piece(r, c, "red", king=(p == "R"))

        if self.selected:
            self.highlight(*self.selected, color="blue")
        if self.forced_piece:
            self.highlight(*self.forced_piece, color="orange")

    def update_status(self):
        legal = all_legal_moves(self.board, self.player, self.forced_piece)
        if not legal:
            winner = "Red" if self.player == "b" else "Black"
            self.status.config(text=f"{winner} wins!")
            self.game_over = True
            return

        turn = "Black" if self.player == "b" else "Red"
        who = "Human" if self.player == self.human_color else f"AI ({self.opponent})"
        if self.forced_piece:
            r, c = self.forced_piece
            self.status.config(text=f"{who} ({turn}) must continue from {square_name(r, c)}")
        else:
            self.status.config(text=f"Turn: {who} ({turn})")

    def on_click(self, event):
        if self.game_over or self.player != self.human_color:
            return

        r = event.y // SQUARE_SIZE
        c = event.x // SQUARE_SIZE
        if not in_bounds(r, c):
            return

        legal = all_legal_moves(self.board, self.player, self.forced_piece)
        if not legal:
            return

        if self.selected is None:
            if owner(self.board[r][c]) != self.player:
                return
            if self.forced_piece and (r, c) != self.forced_piece:
                return
            self.selected = (r, c)
            self.render()
            return

        fr, fc = self.selected
        tr, tc = r, c

        chosen = None
        for m in legal:
            if (m.from_row, m.from_col, m.to_row, m.to_col) == (fr, fc, tr, tc):
                chosen = m
                break

        if chosen is None:
            self.selected = None
            self.render()
            return

        self.apply_and_advance(chosen)

    def apply_and_advance(self, chosen):
        was_cap, was_prom = apply_move(self.board, chosen)

        if was_cap and not was_prom:
            next_caps = [
                m
                for m in all_legal_moves(self.board, self.player, forced_from=(chosen.to_row, chosen.to_col))
                if m.captured is not None
            ]
            if next_caps:
                self.forced_piece = (chosen.to_row, chosen.to_col)
                self.selected = None
                self.render()
                self.update_status()
                if self.player == self.ai_color:
                    self.root.after(250, self.maybe_ai_turn)
                return

        self.forced_piece = None
        self.selected = None
        self.player = "r" if self.player == "b" else "b"

        self.render()
        self.update_status()
        if not self.game_over and self.player == self.ai_color:
            self.root.after(250, self.maybe_ai_turn)

    def maybe_ai_turn(self):
        if self.game_over or self.player != self.ai_color:
            return

        legal = all_legal_moves(self.board, self.player, self.forced_piece)
        if not legal:
            self.update_status()
            return

        idx = ai_move_index(self.ai, self.board, self.player, legal)
        chosen = legal[idx]
        self.apply_and_advance(chosen)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play 6x6 checkers in GUI against AI")
    parser.add_argument("--opponent", choices=["random", "heuristic", "rl"], default="heuristic")
    parser.add_argument("--human-color", choices=["b", "r"], default="b")
    parser.add_argument("--q-table", type=str, default="experiments/results/q_table.npy")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = tk.Tk()
    CheckersHvAIGUI(root, args.opponent, args.human_color, args.q_table, args.seed)
    root.mainloop()


if __name__ == "__main__":
    main()
