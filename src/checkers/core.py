"""Core game logic for 6x6 checkers (board, moves, parsing utilities)."""

from __future__ import annotations

from dataclasses import dataclass

BOARD_SIZE = 6
EMPTY = "."


@dataclass(frozen=True)
class Move:
    """Immutable move descriptor used throughout the engine."""

    from_row: int
    from_col: int
    to_row: int
    to_col: int
    captured: tuple[int, int] | None = None


def create_board() -> list[list[str]]:
    """Construct the starting position with two rows of men per side."""

    board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

    for row in range(2):
        for col in range(BOARD_SIZE):
            if (row + col) % 2 == 1:
                board[row][col] = "b"

    for row in range(BOARD_SIZE - 2, BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if (row + col) % 2 == 1:
                board[row][col] = "r"

    return board


def clone_board(board: list[list[str]]) -> list[list[str]]:
    """Return a shallow copy of the board grid."""

    return [row[:] for row in board]


def in_bounds(row: int, col: int) -> bool:
    """True when (row, col) sits on the BOARD_SIZE square."""

    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE


def owner(piece: str) -> str | None:
    """Map a board symbol to its owning player (or ``None`` for empty)."""

    if piece in ("b", "B"):
        return "b"
    if piece in ("r", "R"):
        return "r"
    return None


def move_dirs(piece: str) -> list[tuple[int, int]]:
    """Return the diagonal direction vectors allowed for the given piece."""

    if piece == "b":
        return [(1, -1), (1, 1)]
    if piece == "r":
        return [(-1, -1), (-1, 1)]
    return [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def piece_moves(board: list[list[str]], row: int, col: int) -> tuple[list[Move], list[Move]]:
    """Generate (normal, capture) moves for the piece at ``(row, col)``."""

    piece = board[row][col]
    if piece == EMPTY:
        return [], []

    normals: list[Move] = []
    captures: list[Move] = []
    piece_owner = owner(piece)
    if piece_owner is None:
        return [], []
    enemy = "r" if piece_owner == "b" else "b"
    is_king = piece in ("B", "R")

    for dr, dc in move_dirs(piece):
        if is_king:
            nr, nc = row + dr, col + dc
            while in_bounds(nr, nc) and board[nr][nc] == EMPTY:
                normals.append(Move(row, col, nr, nc))
                nr += dr
                nc += dc

            if in_bounds(nr, nc) and owner(board[nr][nc]) == enemy:
                jr, jc = nr + dr, nc + dc
                while in_bounds(jr, jc) and board[jr][jc] == EMPTY:
                    captures.append(Move(row, col, jr, jc, captured=(nr, nc)))
                    jr += dr
                    jc += dc
        else:
            nr, nc = row + dr, col + dc
            if in_bounds(nr, nc) and board[nr][nc] == EMPTY:
                normals.append(Move(row, col, nr, nc))

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
    """Generate all moves for ``player`` respecting capture/forced jump rules."""

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
    """Promote a man to king if it just landed on the last rank."""

    if piece == "b" and row == BOARD_SIZE - 1:
        return "B"
    if piece == "r" and row == 0:
        return "R"
    return piece


def apply_move(board: list[list[str]], move: Move) -> tuple[bool, bool]:
    """Mutate ``board`` by applying ``move`` and return capture/promotion flags."""

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


def has_pieces(board: list[list[str]], player: str) -> bool:
    """Check whether ``player`` owns any piece on the board."""

    for row in board:
        for piece in row:
            if owner(piece) == player:
                return True
    return False


def print_board(board: list[list[str]]) -> None:
    """Pretty-print the board using algebraic-style labels."""

    print("\n  a b c d e f")
    for row in range(BOARD_SIZE):
        label = BOARD_SIZE - row
        print(f"{label} " + " ".join(board[row]))
    print()


def parse_square(token: str) -> tuple[int, int] | None:
    """Parse coordinates like ``\"b5\"`` into zero-based indices."""

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
    """Convert zero-based coordinates back to labels such as ``\"b5\"``."""

    return f"{chr(ord('a') + col)}{BOARD_SIZE - row}"


def parse_move(text: str) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Parse ``\"b6 a5\"`` textual moves into endpoint coordinates."""

    parts = text.strip().lower().split()
    if len(parts) != 2:
        return None
    from_sq = parse_square(parts[0])
    to_sq = parse_square(parts[1])
    if from_sq is None or to_sq is None:
        return None
    return from_sq, to_sq
