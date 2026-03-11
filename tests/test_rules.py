"""Regression tests for the 6x6 rules implementation."""

import pytest

from src.checkers.core import BOARD_SIZE, EMPTY, Move, all_legal_moves, apply_move


def empty_board():
    """Return a BOARD_SIZE x BOARD_SIZE grid filled with EMPTY markers."""

    return [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]


def test_forced_capture_only_returns_capture_moves():
    """When a capture is available only those moves should be returned."""

    board = empty_board()
    board[2][1] = "b"
    board[3][2] = "r"

    moves = all_legal_moves(board, "b")

    assert len(moves) == 1
    assert moves[0].captured == (3, 2)
    assert (moves[0].to_row, moves[0].to_col) == (4, 3)


def test_king_can_slide_multiple_squares_diagonally():
    """Kings should slide along diagonals until blocked."""

    board = empty_board()
    board[2][1] = "B"

    moves = all_legal_moves(board, "b")

    destinations = {(move.to_row, move.to_col) for move in moves if move.captured is None}
    assert (0, 3) in destinations  # up-right slide over multiple squares
    assert (5, 4) in destinations  # down-right slide reaching far edge


def test_king_can_land_any_empty_square_after_capture():
    """Kings must be able to land on any empty square beyond the jump target."""

    board = empty_board()
    board[4][1] = "B"
    board[3][2] = "r"

    moves = all_legal_moves(board, "b")

    capture_targets = {
        (move.to_row, move.to_col)
        for move in moves
        if move.captured == (3, 2)
    }
    assert capture_targets == {(2, 3), (1, 4), (0, 5)}


def test_multi_jump_is_forced_in_env():
    """Environment should enforce another capture when available."""

    pytest.importorskip("gymnasium")
    from src.checkers.env import Checkers6x6Env

    env = Checkers6x6Env(seed=7)
    env.reset(seed=7)

    env.board = empty_board()
    env.board[0][1] = "b"
    env.board[1][2] = "r"
    env.board[3][4] = "r"
    env.player = "b"
    env.forced_piece = None
    env._refresh_legal_moves()

    first_jump_idx = 0
    for i, move in enumerate(env.legal_moves):
        if (move.from_row, move.from_col, move.to_row, move.to_col) == (0, 1, 2, 3):
            first_jump_idx = i
            break

    _, reward, terminated, truncated, _ = env.step(first_jump_idx)

    # Reward shaping: +0.1 capture and -0.005 step penalty.
    assert reward == pytest.approx(0.095)
    assert not terminated
    assert not truncated
    assert env.player == "b"
    assert env.forced_piece == (2, 3)
    assert len(env.legal_moves) == 1
    assert env.legal_moves[0].captured == (3, 4)
    assert (env.legal_moves[0].to_row, env.legal_moves[0].to_col) == (4, 5)


def test_promotion_to_king():
    """Simple pawn reaching back rank should become a king."""

    board = empty_board()
    board[4][1] = "b"
    move = Move(from_row=4, from_col=1, to_row=5, to_col=0)

    was_capture, was_promoted = apply_move(board, move)

    assert not was_capture
    assert was_promoted
    assert board[5][0] == "B"


def test_terminal_detection_when_player_has_no_legal_moves():
    """If the side to move has no moves, the opponent wins immediately."""

    pytest.importorskip("gymnasium")
    from src.checkers.env import Checkers6x6Env

    env = Checkers6x6Env(seed=11)
    env.reset(seed=11)

    env.board = empty_board()
    env.board[0][1] = "b"
    env.player = "r"
    env.forced_piece = None
    env._refresh_legal_moves()

    _, reward, terminated, truncated, info = env.step(0)

    assert terminated
    assert not truncated
    assert reward == -1.0
    assert info["winner"] == "b"
