from environment.go_game import GoGame
from network.model_loader import load_model
from mcts.network_Evaluator import NetworkEvaluator
from mcts.mcts import MCTS


def test_mcts_with_trained_model():

    game = GoGame(
        board_size=12
    )

    model = load_model(
        "models/latest_model.pt"
    )

    evaluator = NetworkEvaluator(
        model
    )

    mcts = MCTS(
        game,
        evaluator,
        simulations=5
    )

    move = mcts.search()

    assert move is not None