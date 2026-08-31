import numpy as np

from environment.go_game import GoGame
from self_play.self_play import SelfPlay
from mcts.mcts import MCTS


def test_mcts_policy():

    self_play = SelfPlay(
        board_size=12,
        simulations=2
    )

    game = GoGame(12)

    mcts = MCTS(
        game,
        simulations=2
    )

    mcts.search()

    policy = self_play.get_mcts_policy(mcts)

    assert policy.shape == (145,)

    assert np.isclose(
        policy.sum(),
        1.0
    )

def test_generate_games():

    self_play = SelfPlay(
        board_size=12,
        simulations=2
    )

    data = self_play.generate_games(
        num_games=2
    )

    assert len(data) > 0
def test_self_play_creation():

    self_play = SelfPlay(
        board_size=12,
        simulations=2
    )

    assert self_play.board_size == 12
    assert self_play.simulations == 2