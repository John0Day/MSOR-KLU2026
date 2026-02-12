#!/usr/bin/env python3
# Shebang: lets you run this file as an executable script on Unix-like systems (e.g., ./game.py)
# Not part of the game model; it's OS/runtime metadata.

from dataclasses import dataclass
# dataclass: convenient way to define immutable/structured records (we use it for Move objects)

BOARD_SIZE = 6
# Environment parameter: size of the board (6x6). This directly shapes the STATE space.

EMPTY = "."
# State encoding: character used to represent an empty cell in the board state.


@dataclass(frozen=True)
# Dataclass defining an ACTION representation.
# frozen=True makes Move immutable (safer; moves can't be modified after creation).
class Move:
    # Action components: from-square and to-square coordinates (row/col indices)
    from_row: int
    from_col: int
    to_row: int
    to_col: int
    # captured: if the move is a capture, store the coordinates of the captured piece; else None.
    # This is action metadata that affects the transition function (removing the piece).
    captured: tuple[int, int] | None = None


def create_board() -> list[list[str]]:
    # Initializes the initial STATE of the environment: a 6x6 grid with starting pieces.
    board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    # Place black pieces ("b") on the top 2 rows, only on dark squares ((row+col)%2==1).
    for row in range(2):
        for col in range(BOARD_SIZE):
            if (row + col) % 2 == 1:
                board[row][col] = "b"
    # Place red pieces ("r") on the bottom 2 rows, only on dark squares.
    for row in range(BOARD_SIZE - 2, BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if (row + col) % 2 == 1:
                board[row][col] = "r"
    # Returns the starting board: this is the initial state s0.
    return board


def print_board(board: list[list[str]]) -> None:
    # Rendering function (UI): prints the current STATE to the console.
    # Not part of the game logic; purely visualization for humans.
    print("\n   a b c d e f")
    for row in range(BOARD_SIZE):
        label = BOARD_SIZE - row
        # Prints each row with a human-friendly label (6..1), matching checkers coordinates.
        print(f"{label}  " + " ".join(board[row]))
    print()


def in_bounds(row: int, col: int) -> bool:
    # Environment constraint: checks whether (row,col) is inside the board.
    # Used to ensure actions don't go outside valid state space.
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE


def owner(piece: str) -> str | None:
    # State interpretation: maps a board symbol to its "player owner".
    # "b" and "B" belong to black; "r" and "R" belong to red; "." is None.
    if piece in ("b", "B"):
        return "b"
    if piece in ("r", "R"):
        return "r"
    return None


def move_dirs(piece: str) -> list[tuple[int, int]]:
    # Action constraint (movement model): returns allowed direction vectors (dr, dc).
    # This defines legal action generation for a piece type.
    if piece == "b":
        # Black men move "down" the board: increasing row index.
        return [(1, -1), (1, 1)]
    if piece == "r":
        # Red men move "up" the board: decreasing row index.
        return [(-1, -1), (-1, 1)]
    # Kings ("B" or "R") can move in all 4 diagonal directions.
    return [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def piece_moves(board: list[list[str]], row: int, col: int) -> tuple[list[Move], list[Move]]:
    # Transition feasibility generator: computes all legal NORMAL moves and CAPTURE moves
    # for a single piece at (row,col), given the current STATE (board).
    piece = board[row][col]
    # If empty square, no actions possible.
    if piece == EMPTY:
        return [], []

    normals: list[Move] = []
    captures: list[Move] = []
    # Determine the enemy owner symbol for capture checking.
    enemy = "r" if owner(piece) == "b" else "b"

    # Explore each allowed movement direction based on piece type (man/king).
    for dr, dc in move_dirs(piece):
        nr, nc = row + dr, col + dc
        # Normal move rule: adjacent diagonal square is empty.
        if in_bounds(nr, nc) and board[nr][nc] == EMPTY:
            # This defines a legal action (non-capture).
            normals.append(Move(row, col, nr, nc))
        # Capture move rule: jump over adjacent enemy onto empty landing square.
        jr, jc = row + 2 * dr, col + 2 * dc
        if (
            in_bounds(jr, jc)              # landing square is on board
            and board[jr][jc] == EMPTY     # landing square is empty
            and in_bounds(nr, nc)          # jumped-over square is on board
            and owner(board[nr][nc]) == enemy  # jumped-over piece belongs to enemy
        ):
            # Legal capture action; store captured coordinate for the transition update.
            captures.append(Move(row, col, jr, jc, captured=(nr, nc)))
    # Return action set split by type (normal vs capture).
    return normals, captures


def all_legal_moves(
    board: list[list[str]], player: str, forced_from: tuple[int, int] | None = None
) -> list[Move]:
    # Action space generator for a given player in the current STATE.
    # Implements "forced capture": if any capture exists, ONLY captures are legal.
    # Also supports multi-jump enforcement via forced_from (must continue with same piece).
    normals: list[Move] = []
    captures: list[Move] = []

    # Scan the board to find pieces belonging to the current player.
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if owner(board[row][col]) != player:
                continue
            # If we're in a multi-jump sequence, only generate moves for the forced piece.
            if forced_from is not None and (row, col) != forced_from:
                continue
            # Generate moves for this piece in the current state.
            n_moves, c_moves = piece_moves(board, row, col)
            normals.extend(n_moves)
            captures.extend(c_moves)
    # Rule/constraint: captures are mandatory if any exist.
    return captures if captures else normals


def promote_if_needed(piece: str, row: int) -> str:
    # Transition component: king promotion rule.
    # If a man reaches the far edge, it becomes a king.
    if piece == "b" and row == BOARD_SIZE - 1:
        return "B"  # black man promoted to black king
    if piece == "r" and row == 0:
        return "R"  # red man promoted to red king
    return piece  # no promotion


def apply_move(board: list[list[str]], move: Move) -> tuple[bool, bool]:
    # Transition function: applies an ACTION to the current STATE to produce the next STATE.
    # Returns:
    #   was_capture: whether the action captured a piece (affects multi-jump rule)
    #   promoted: whether promotion occurred (affects whether multi-jump continues here)
    piece = board[move.from_row][move.from_col]  # read piece from state
    board[move.from_row][move.from_col] = EMPTY  # update state: origin becomes empty
    board[move.to_row][move.to_col] = piece      # update state: destination gets piece

    # If capture action, remove the captured enemy piece from the board state.
    if move.captured is not None:
        cr, cc = move.captured
        board[cr][cc] = EMPTY

    # Apply promotion rule after moving.
    promoted_piece = promote_if_needed(board[move.to_row][move.to_col], move.to_row)
    promoted = promoted_piece != board[move.to_row][move.to_col]  # did symbol change?
    board[move.to_row][move.to_col] = promoted_piece              # update state if promoted
    return move.captured is not None, promoted


def parse_square(token: str) -> tuple[int, int] | None:
    # Input parser (UI): converts a human coordinate like "b6" into (row,col) indices.
    # Not part of the MDP; it helps humans choose an action.
    token = token.strip().lower()
    if len(token) != 2:
        return None
    col_ch, row_ch = token[0], token[1]
    # Validate that column is within a..f and row within 1..6 for a 6x6 board.
    if col_ch < "a" or col_ch > "f" or row_ch < "1" or row_ch > "6":
        return None
    col = ord(col_ch) - ord("a")      # map 'a'..'f' -> 0..5
    row = BOARD_SIZE - int(row_ch)    # map '6'..'1' -> 0..5 (top row is 6)
    return row, col


def square_name(row: int, col: int) -> str:
    # Rendering helper: converts internal (row,col) back to user coordinate like "b6".
    return f"{chr(ord('a') + col)}{BOARD_SIZE - row}"


def parse_move(text: str) -> tuple[tuple[int, int], tuple[int, int]] | None:
    # Input parser (UI): expects two squares "from to" (e.g., "b6 a5").
    # Produces the candidate action endpoints (from_sq, to_sq).
    parts = text.strip().lower().split()
    if len(parts) != 2:
        return None
    from_sq = parse_square(parts[0])
    to_sq = parse_square(parts[1])
    if from_sq is None or to_sq is None:
        return None
    return from_sq, to_sq


def current_player_name(player: str) -> str:
    # UI helper: maps internal player id ("b"/"r") to display label.
    return "Black" if player == "b" else "Red"


def main() -> None:
    # Main game loop: runs episodes until terminal condition (win) or user quits.
    board = create_board()
    # Initial state s0: board from create_board()
    player = "b"
    # State variable: whose turn it is ("b" or "r"). This is part of the full game state.
    forced_piece: tuple[int, int] | None = None
    # Control variable for enforcing multi-jump captures:
    # If not None, the player must continue capturing with this piece.

    print("Simple 6x6 Checkers")
    print("Enter moves like: b6 a5")
    print("Type 'q' to quit.\n")

    # Episode loop: each iteration is one decision step (one turn, possibly with forced multi-jump).
    while True:
        # Generate legal actions for the current player in the current state.
        legal_moves = all_legal_moves(board, player, forced_from=forced_piece)
        # Terminal test (OUTCOME): if no legal moves, current player loses.
        if not legal_moves:
            winner = "Red" if player == "b" else "Black"
            print_board(board)
            print(f"{current_player_name(player)} has no legal moves. {winner} wins!")
            # Outcome: terminal state reached (win/loss).
            break

        # Show current state to the user.
        print_board(board)
        # If multi-jump is active, inform the player of the constraint.
        if forced_piece is not None:
            print(f"{current_player_name(player)} must continue jumping with {square_name(*forced_piece)}.")

        # Get user action input (policy provided by human).
        prompt = f"{current_player_name(player)} to move > "
        move_text = input(prompt).strip()
        # Optional termination by user (not game-rule terminal, but user quit).
        if move_text.lower() in {"q", "quit", "exit"}:
            print("Game ended.")
            break

        # Parse human input into a proposed move endpoints.
        parsed = parse_move(move_text)
        if parsed is None:
            # Invalid action format (UI constraint), request again.
            print("Invalid input. Use format like 'b6 a5'.")
            continue
        (fr, fc), (tr, tc) = parsed

        # Validate that the chosen endpoints match one of the legal actions from the environment.
        selected = None
        for move in legal_moves:
            if (move.from_row, move.from_col, move.to_row, move.to_col) == (fr, fc, tr, tc):
                selected = move
                break
        if selected is None:
            # Proposed action is not in the legal action set A(s).
            print("Illegal move.")
            continue

        # Apply the chosen action: state transition s -> s'
        was_capture, was_promoted = apply_move(board, selected)

        # If a capture occurred and no promotion happened, check for continued capture (multi-jump).
        if was_capture and not was_promoted:
            # Generate moves again but only for the piece that just moved (forced continuation).
            next_captures = [
                m for m in all_legal_moves(board, player, forced_from=(selected.to_row, selected.to_col))
                if m.captured is not None
            ]
            # If additional captures exist, enforce the constraint: same player continues with same piece.
            if next_captures:
                forced_piece = (selected.to_row, selected.to_col)
                # Do NOT switch player; same player takes another forced capture step.
                continue

        # If no forced continuation, clear forced piece.
        forced_piece = None
        # Switch turn: transition of "player-to-move" part of the state.
        player = "r" if player == "b" else "b"


if __name__ == "__main__":
    # Standard Python entry point: run main() only when executed directly.
    main()
