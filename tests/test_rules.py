import pytest

from src.checkers.core import BOARD_SIZE, EMPTY, Move, all_legal_moves, apply_move


def empty_board():
    return [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]


def test_forced_capture_only_returns_capture_moves():
    board = empty_board()
    board[2][1] = "b"
    board[3][2] = "r"

    moves = all_legal_moves(board, "b")

    assert len(moves) == 1
    assert moves[0].captured == (3, 2)
    assert (moves[0].to_row, moves[0].to_col) == (4, 3)


def test_multi_jump_is_forced_in_env():
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

    assert reward == 0.0
    assert not terminated
    assert not truncated
    assert env.player == "b"
    assert env.forced_piece == (2, 3)
    assert len(env.legal_moves) == 1
    assert env.legal_moves[0].captured == (3, 4)
    assert (env.legal_moves[0].to_row, env.legal_moves[0].to_col) == (4, 5)


def test_promotion_to_king():
    board = empty_board()
    board[4][1] = "b"
    move = Move(from_row=4, from_col=1, to_row=5, to_col=0)

    was_capture, was_promoted = apply_move(board, move)

    assert not was_capture
    assert was_promoted
    assert board[5][0] == "B"


def test_terminal_detection_when_player_has_no_legal_moves():
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
