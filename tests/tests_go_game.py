import numpy as np
from environment.go_game import GoGame
def test_initial_board():
    game = GoGame()
    assert game.board.shape == (12, 12)
    assert np.all(game.board == 0)
    assert game.current_player == 1
    assert game.game_over is False
def test_neighbors():
    game = GoGame()
    assert len(game.get_neighbors(5, 5)) == 4
    assert len(game.get_neighbors(0, 0)) == 2
    assert len(game.get_neighbors(0, 5)) == 3
def test_group_detection():
    game = GoGame()
    game.board[5, 5] = 1
    game.board[5, 6] = 1
    game.board[6, 5] = 1
    group = game.get_group(5, 5)
    assert len(group) == 3
def test_liberties():
    game = GoGame()
    game.board[5, 5] = 1
    group = game.get_group(5, 5)
    liberties = game.get_liberties(group)
    assert len(liberties) == 4
def test_normal_move():
    game = GoGame()
    result = game.make_move(5, 5)
    assert result is True
    assert game.board[5, 5] == 1
    assert game.current_player == -1
def test_occupied_position():
    game = GoGame()
    assert game.make_move(5, 5) is True
    assert game.make_move(5, 5) is False
def test_suicide():
    game = GoGame()
    game.board[4, 5] = -1
    game.board[6, 5] = -1
    game.board[5, 4] = -1
    game.board[5, 6] = -1
    game.current_player = 1
    assert game.make_move(5, 5) is False
    assert game.board[5, 5] == 0
def test_two_passes_end_game():
    game = GoGame()
    assert game.pass_move() is True
    assert game.game_over is False
    assert game.pass_move() is True
    assert game.game_over is True
def test_initial_legal_moves():
    game = GoGame()
    moves = game.get_legal_moves()
    assert len(moves) == 145
    assert "PASS" in moves
def test_action_conversion():
    game = GoGame()
    assert game.position_to_action(0, 0) == 0
    assert game.position_to_action(11, 11) == 143
    assert game.action_to_position(0) == (0, 0)
    assert game.action_to_position(143) == (11, 11)
    assert game.get_pass_action() == 144
def test_state_save_restore():
    game = GoGame()
    game.make_move(5, 5)
    state = game.get_state()
    game.make_move(4, 4)
    game.set_state(state)
    assert game.board[5, 5] == 1
    assert game.board[4, 4] == 0
def test_pass():
    game = GoGame()
    game.pass_move()
    assert game.consecutive_passes == 1
    assert game.game_over is False
    game.pass_move()
    assert game.consecutive_passes == 2
    assert game.game_over is True
def test_count_stones():
    game = GoGame()
    game.board[5, 5] = 1
    game.board[5, 6] = 1
    game.board[6, 5] = -1
    black, white = game.count_stones()
    assert black == 2
    assert white == 1
