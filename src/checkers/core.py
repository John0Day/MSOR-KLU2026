from __future__ import annotations

from dataclasses import dataclass

BOARD_SIZE = 6
EMPTY = "."


@dataclass(frozen=True)
class Move:
    from_row: int
    from_col: int
    to_row: int
    to_col: int
    captured: tuple[int, int] | None = None


def create_board() -> list[list[str]]:
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
    return [row[:] for row in board]


def in_bounds(row: int, col: int) -> bool:
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE


def owner(piece: str) -> str | None:
    if piece in ("b", "B"):
        return "b"
    if piece in ("r", "R"):
        return "r"
    return None


def move_dirs(piece: str) -> list[tuple[int, int]]:
    if piece == "b":
        return [(1, -1), (1, 1)]
    if piece == "r":
        return [(-1, -1), (-1, 1)]
    return [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def piece_moves(board: list[list[str]], row: int, col: int) -> tuple[list[Move], list[Move]]:
    piece = board[row][col]
    if piece == EMPTY:
        return [], []

    normals: list[Move] = []
    captures: list[Move] = []
    enemy = "r" if owner(piece) == "b" else "b"

    for dr, dc in move_dirs(piece):
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
    if piece == "b" and row == BOARD_SIZE - 1:
        return "B"
    if piece == "r" and row == 0:
        return "R"
    return piece


def apply_move(board: list[list[str]], move: Move) -> tuple[bool, bool]:
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
    for row in board:
        for piece in row:
            if owner(piece) == player:
                return True
    return False


def print_board(board: list[list[str]]) -> None:
    print("\n  a b c d e f")
    for row in range(BOARD_SIZE):
        label = BOARD_SIZE - row
        print(f"{label} " + " ".join(board[row]))
    print()


def parse_square(token: str) -> tuple[int, int] | None:
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
    return f"{chr(ord('a') + col)}{BOARD_SIZE - row}"


def parse_move(text: str) -> tuple[tuple[int, int], tuple[int, int]] | None:
    parts = text.strip().lower().split()
    if len(parts) != 2:
        return None
    from_sq = parse_square(parts[0])
    to_sq = parse_square(parts[1])
    if from_sq is None or to_sq is None:
        return None
    return from_sq, to_sq
