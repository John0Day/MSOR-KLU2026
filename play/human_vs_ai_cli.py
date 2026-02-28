from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents import HeuristicAgent, QTableAgent, RandomAgent
from src.checkers.core import (
    all_legal_moves,
    apply_move,
    create_board,
    parse_move,
    print_board,
    square_name,
)


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


def current_player_name(player: str) -> str:
    return "Black" if player == "b" else "Red"


def ai_move_index(ai, board, player, legal_moves):
    if isinstance(ai, HeuristicAgent):
        return ai.select_move_index(board, player, legal_moves)
    if isinstance(ai, QTableAgent):
        return ai.select_move_index(board, player, legal_moves)
    if isinstance(ai, RandomAgent):
        return ai.select_move_index(len(legal_moves))
    return 0


def play_human_vs_ai(opponent: str, human_color: str, q_table: str, seed: int) -> None:
    board = create_board()
    player = "b"
    forced_piece: tuple[int, int] | None = None

    ai = load_ai(opponent, q_table, seed)
    ai_color = "r" if human_color == "b" else "b"

    print("6x6 Checkers (Human vs Computer)")
    print(f"Human: {current_player_name(human_color)} | AI: {current_player_name(ai_color)} ({opponent})")
    print("Enter moves like: b6 a5")
    print("Type 'q' to quit.\n")

    while True:
        legal_moves = all_legal_moves(board, player, forced_from=forced_piece)
        if not legal_moves:
            winner = "r" if player == "b" else "b"
            print_board(board)
            print(f"{current_player_name(winner)} wins!")
            break

        print_board(board)
        if forced_piece is not None:
            print(f"{current_player_name(player)} must continue jumping with {square_name(*forced_piece)}")

        if player == human_color:
            prompt = f"{current_player_name(player)} to move > "
            try:
                move_text = input(prompt).strip()
            except EOFError:
                print("\nInput ended. Game ended.")
                break
            if move_text.lower() in {"q", "quit", "exit"}:
                print("Game ended.")
                break

            parsed = parse_move(move_text)
            if parsed is None:
                print("Invalid input. Use format like 'b6 a5'.")
                continue

            (fr, fc), (tr, tc) = parsed
            selected = None
            for move in legal_moves:
                if (move.from_row, move.from_col, move.to_row, move.to_col) == (fr, fc, tr, tc):
                    selected = move
                    break
            if selected is None:
                print("Illegal move.")
                continue
        else:
            idx = ai_move_index(ai, board, player, legal_moves)
            selected = legal_moves[idx]
            print(
                f"AI ({opponent}) plays: "
                f"{square_name(selected.from_row, selected.from_col)} {square_name(selected.to_row, selected.to_col)}"
            )

        was_capture, was_promoted = apply_move(board, selected)

        if was_capture and not was_promoted:
            next_caps = [
                m
                for m in all_legal_moves(board, player, forced_from=(selected.to_row, selected.to_col))
                if m.captured is not None
            ]
            if next_caps:
                forced_piece = (selected.to_row, selected.to_col)
                continue

        forced_piece = None
        player = "r" if player == "b" else "b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play 6x6 checkers in CLI against AI")
    parser.add_argument("--opponent", choices=["random", "heuristic", "rl"], default="heuristic")
    parser.add_argument("--human-color", choices=["b", "r"], default="b")
    parser.add_argument("--q-table", type=str, default="experiments/results/q_table.npy")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    play_human_vs_ai(args.opponent, args.human_color, args.q_table, args.seed)


if __name__ == "__main__":
    main()
