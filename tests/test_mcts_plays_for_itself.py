"""Regression test: MCTS must pick moves that are good for the player to move.

A stub evaluator scores positions by material from the perspective of the
player to move.  With a white group in atari, Black's search must capture.
Before the backup-sign fix, search maximised the opponent's value instead.
"""
import numpy as np

from environment.go_game import GoGame
from mcts.mcts import MCTS


class MaterialEvaluator:
    model = None

    def evaluate(self, state):
        n = state.board_size * state.board_size
        policy = np.full(n + 1, 1.0 / (n + 1), dtype=np.float32)
        me = state.current_player
        diff = (state.board == me).sum() - (state.board == -me).sum()
        return policy, float(np.tanh(diff / 4.0))


def test_mcts_captures_group_in_atari():
    game = GoGame(board_size=5)
    # White stones at (1,1),(1,2) surrounded by Black except (1,3).
    for r, c in [(0, 1), (0, 2), (1, 0), (2, 1), (2, 2)]:
        game.board[r, c] = 1
    game.board[1, 1] = game.board[1, 2] = -1
    game.current_player = 1

    searcher = MCTS(model=None, game=game, board_size=5, simulations=300)
    searcher.evaluator = MaterialEvaluator()
    assert searcher.search(game) == 1 * 5 + 3  # capture at (1,3)
