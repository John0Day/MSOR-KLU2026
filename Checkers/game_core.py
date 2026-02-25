#!/usr/bin/env python3
"""
Core 6x6 checkers game logic.

This file contains:
- Board representation
- Move generation
- Forced captures and multi-jumps
- Promotion
- Helpers for coordinates and printing

No UI, no RL, no Gym here.
"""

from dataclasses import dataclass

# Environment parameter: size of the board (6x6). This directly shapes the STATE space.
BOARD_SIZE = 6

# State encoding: character used to represent an empty cell in the board state.
EMPTY = "."

@dataclass(frozen=True)
class Move:
    """
    Immutable representation of a move.

    Attributes:
        from_row, from_col: origin square
        to_row, to_col: destination square
        captured: (row, col) of captured piece, or None
    """
    from_row: int
    from_col: int
    to_row: int
    to_col: int
    captured: tuple[int, int] | None = None


def create_board() -> list[list[str]]:
    """
    Initializes the initial STATE of the environment: a 6x6 grid with starting pieces.

    - Black pieces ("b") on the top 2 rows, only on dark squares ((row+col)%2==1).
    - Red pieces ("r") on the bottom 2 rows, only on dark squares.
    """
    board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

    # Place black pieces ("b") on the top 2 rows, only on dark squares.
    for row in range(2):
        for col in range(BOARD_SIZE):
            if (row + col) % 2 == 1:
                board[row][col] = "b"

    # Place red pieces ("r") on the bottom 2 rows, only on dark squares.
    for row in range(BOARD_SIZE - 2, BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if (row + col) % 2 == 1:
                board[row][col] = "r"

    return board


def print_board(board: list[list[str]]) -> None:
    """
    Console rendering helper: prints the current board state.
    """
    print("\n  a b c d e f")
    for row in range(BOARD_SIZE):
        label = BOARD_SIZE - row
        print(f"{label} " + " ".join(board[row]))
    print()


def in_bounds(row: int, col: int) -> bool:
    """
    Checks whether (row, col) is inside the board.
    """
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE


def owner(piece: str) -> str | None:
    """
    Maps a board symbol to its owner:
    - "b", "B" -> "b"
    - "r", "R" -> "r"
    - "."      -> None
    """
    if piece in ("b", "B"):
        return "b"
    if piece in ("r", "R"):
        return "r"
    return None
 

def move_dirs(piece: str) -> list[tuple[int, int]]:
    """
    Returns allowed direction vectors (dr, dc) for a piece type.
    - Men move forward only.
    - Kings move in all 4 diagonals.
    """
    if piece == "b":
        # Black men move "down" the board: increasing row index.
        return [(1, -1), (1, 1)]
    if piece == "r":
        # Red men move "up" the board: decreasing row index.
        return [(-1, -1), (-1, 1)]
    # Kings ("B" or "R") can move in all 4 diagonal directions.
    return [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def piece_moves(board: list[list[str]], row: int, col: int) -> tuple[list[Move], list[Move]]:
    """
    Computes all legal NORMAL moves and CAPTURE moves for a single piece at (row, col).

    Returns:
        normals: list of non-capture moves
        captures: list of capture moves
    """
    piece = board[row][col]

    if piece == EMPTY:
        return [], []

    normals: list[Move] = []
    captures: list[Move] = []

    enemy = "r" if owner(piece) == "b" else "b"

    for dr, dc in move_dirs(piece):
        nr, nc = row + dr, col + dc

        # Normal move: adjacent diagonal square is empty.
        if in_bounds(nr, nc) and board[nr][nc] == EMPTY:
            normals.append(Move(row, col, nr, nc))

        # Capture move: jump over adjacent enemy onto empty landing square.
        jr, jc = row + 2 * dr, col + 2 * dc
        if (
            in_bounds(jr, jc)
            and board[jr][jc] == EMPTY
            and in_bounds(nr, nc)
            and owner(board[nr][nc]) == enemy
        ):
            captures.append(Move(row, col, jr, jc, captured=(nr, nc)))

    return normals, captures


def all_legal_moves(
    board: list[list[str]], player: str, forced_from: tuple[int, int] | None = None
) -> list[Move]:
    """
    Generates all legal moves for a given player in the current state.

    - If any capture exists, ONLY captures are legal (forced capture rule).
    - If forced_from is not None, only moves for that piece are generated (multi-jump).
    """
    normals: list[Move] = []
    captures: list[Move] = []

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if owner(board[row][col]) != player:
                continue

            if forced_from is not None and (row, col) != forced_from:
                continue

            n_moves, c_moves = piece_moves(board, row, col)
            normals.extend(n_moves)
            captures.extend(c_moves)

    return captures if captures else normals


def promote_if_needed(piece: str, row: int) -> str:
    """
    King promotion rule:
    - Black man reaching last row becomes "B".
    - Red man reaching first row becomes "R".
    """
    if piece == "b" and row == BOARD_SIZE - 1:
        return "B"
    if piece == "r" and row == 0:
        return "R"
    return piece


def apply_move(board: list[list[str]], move: Move) -> tuple[bool, bool]:
    """
    Applies a move to the board.

    Returns:
        was_capture: True if a piece was captured
        was_promoted: True if the moving piece was promoted to king
    """
    piece = board[move.from_row][move.from_col]
    board[move.from_row][move.from_col] = EMPTY
    board[move.to_row][move.to_col] = piece

    was_capture = False
    if move.captured is not None:
        cr, cc = move.captured
        board[cr][cc] = EMPTY
        was_capture = True

    promoted_piece = promote_if_needed(board[move.to_row][move.to_col], move.to_row)
    was_promoted = promoted_piece != board[move.to_row][move.to_col]
    board[move.to_row][move.to_col] = promoted_piece

    return was_capture, was_promoted


def parse_square(token: str) -> tuple[int, int] | None:
    """
    Converts a human coordinate like "b6" into (row, col) indices.
    """
    token = token.strip().lower()
    if len(token) != 2:
        return None

    col_ch, row_ch = token[0], token[1]

    if col_ch < "a" or col_ch > "f" or row_ch < "1" or row_ch > "6":
        return None

    col = ord(col_ch) - ord("a")
    row = BOARD_SIZE - int(row_ch)
    return row, col


def square_name(row: int, col: int) -> str:
    """
    Converts internal (row, col) back to user coordinate like "b6".
    """
    return f"{chr(ord('a') + col)}{BOARD_SIZE - row}"


def parse_move(text: str) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """
    Parses a move string "b6 a5" into ((from_row, from_col), (to_row, to_col)).
    """
    parts = text.strip().lower().split()
    if len(parts) != 2:
        return None

    from_sq = parse_square(parts[0])
    to_sq = parse_square(parts[1])

    if from_sq is None or to_sq is None:
        return None

    return from_sq, to_sq


def current_player_name(player: str) -> str:
    """
    Maps internal player id ("b"/"r") to display label.
    """
    return "Black" if player == "b" else "Red"
