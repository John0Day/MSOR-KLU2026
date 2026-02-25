#!/usr/bin/env python3
"""
Tkinter GUI for 6x6 Checkers (Human vs Human)

This GUI uses the pure game logic from game_core.py.
No AI, no RL, no Gym — just a clean interactive board.
"""

import tkinter as tk
from game_core import (
    BOARD_SIZE, create_board, all_legal_moves, apply_move,
    owner, square_name, in_bounds
)

SQUARE_SIZE = 90
LIGHT = "#F0D9B5"   # light wood
DARK = "#B58863"    # dark wood


class CheckersGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("6x6 Checkers – Human vs Human")

        # Game state
        self.board = create_board()
        self.player = "b"
        self.forced_piece = None
        self.selected = None

        # Canvas
        self.canvas = tk.Canvas(
            root,
            width=BOARD_SIZE * SQUARE_SIZE,
            height=BOARD_SIZE * SQUARE_SIZE
        )
        self.canvas.pack()

        # Status label
        self.status = tk.Label(root, text="", font=("Arial", 14))
        self.status.pack(pady=5)

        # Bind mouse click
        self.canvas.bind("<Button-1>", self.on_click)

        self.render()
        self.update_status()

    # ------------------ Drawing ------------------

    def draw_board(self):
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                color = DARK if (r + c) % 2 else LIGHT
                self.canvas.create_rectangle(
                    c * SQUARE_SIZE, r * SQUARE_SIZE,
                    (c + 1) * SQUARE_SIZE, (r + 1) * SQUARE_SIZE,
                    fill=color, outline=""
                )

    def draw_piece(self, r, c, color, king=False):
        x = c * SQUARE_SIZE + SQUARE_SIZE // 2
        y = r * SQUARE_SIZE + SQUARE_SIZE // 2
        rad = SQUARE_SIZE // 2 - 12

        self.canvas.create_oval(
            x - rad, y - rad, x + rad, y + rad,
            fill=color, outline="black", width=2
        )

        if king:
            self.canvas.create_text(x, y, text="K", fill="gold", font=("Arial", 20, "bold"))

    def highlight(self, r, c, color="blue"):
        self.canvas.create_rectangle(
            c * SQUARE_SIZE, r * SQUARE_SIZE,
            (c + 1) * SQUARE_SIZE, (r + 1) * SQUARE_SIZE,
            outline=color, width=3
        )

    def render(self):
        self.canvas.delete("all")
        self.draw_board()

        # Draw pieces
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                p = self.board[r][c]
                if p in ("b", "B"):
                    self.draw_piece(r, c, "black", king=(p == "B"))
                elif p in ("r", "R"):
                    self.draw_piece(r, c, "red", king=(p == "R"))

        # Highlight selected piece
        if self.selected:
            self.highlight(*self.selected, color="blue")

        # Highlight forced piece (multi-jump)
        if self.forced_piece:
            self.highlight(*self.forced_piece, color="orange")

    # ------------------ Game Flow ------------------

    def update_status(self):
        legal = all_legal_moves(self.board, self.player, self.forced_piece)
        if not legal:
            winner = "Red" if self.player == "b" else "Black"
            self.status.config(text=f"{winner} wins!")
            return

        if self.forced_piece:
            r, c = self.forced_piece
            self.status.config(
                text=f"{self.player.upper()} must continue from {square_name(r, c)}"
            )
        else:
            self.status.config(text=f"{self.player.upper()}'s turn")

    def on_click(self, event):
        r = event.y // SQUARE_SIZE
        c = event.x // SQUARE_SIZE

        if not in_bounds(r, c):
            return

        legal = all_legal_moves(self.board, self.player, self.forced_piece)
        if not legal:
            return

        # First click: select piece
        if self.selected is None:
            if owner(self.board[r][c]) != self.player:
                return
            if self.forced_piece and (r, c) != self.forced_piece:
                return
            self.selected = (r, c)
            self.render()
            return

        # Second click: attempt move
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

        # Apply move
        was_cap, was_prom = apply_move(self.board, chosen)

        # Multi-jump check
        if was_cap and not was_prom:
            next_caps = [
                m for m in all_legal_moves(
                    self.board, self.player,
                    forced_from=(chosen.to_row, chosen.to_col)
                )
                if m.captured
            ]
            if next_caps:
                self.forced_piece = (chosen.to_row, chosen.to_col)
                self.selected = None
                self.render()
                self.update_status()
                return

        # Normal turn switch
        self.forced_piece = None
        self.selected = None
        self.player = "r" if self.player == "b" else "b"

        self.render()
        self.update_status()


def main():
    root = tk.Tk()
    CheckersGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
