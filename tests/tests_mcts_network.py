import time
from environment.go_game import GoGame
from network.network import GoNetwork
from mcts.mcts import MCTS
from mcts.network_Evaluator import NetworkEvaluator
from mcts.mcts import puct_score
import numpy as np
board = GoGame(12)

model = GoNetwork(12)

mcts = MCTS(
    model=model,
    board_size=12,
    simulations=100
)

for move_number in range(10):

    start = time.time()

    move = mcts.search(board)

    end = time.time()

    print(
        "Move",
        move_number + 1,
        ":",
        move
    )

    print(
        "MCTS time:",
        end - start,
        "seconds"
    )

    assert board.is_legal(move)

    board.play(move)
for simulations in [10, 50, 100, 200, 500]:

    mcts = MCTS(
        model=model,
        board_size=12,
        simulations=simulations
    )

    start = time.time()

    move = mcts.search(board)

    end = time.time()

    print(
        simulations,
        "simulations →",
        end - start,
        "seconds"
    )
def test_puct_score():

    score = puct_score(
        prior=0.5,
        value=0.2,
        parent_visits=10,
        child_visits=2
    )

    print("PUCT score:", score)

    assert isinstance(score, float)
def test_network_evaluator():

    board_size = 12

    # Create neural network
    model = GoNetwork(
        board_size=board_size
    )

    # Create evaluator
    evaluator = NetworkEvaluator(
        model
    )

    # Create empty board
    board = np.zeros(
        (board_size, board_size),
        dtype=np.int8
    )

    # Evaluate board
    policy, value = evaluator.evaluate(
        board
    )

    print("Policy shape:", policy.shape)
    print("Value:", value)
    print("Policy sum:", policy.sum())
    assert policy.shape == (145,)

    assert -1.0 <= value <= 1.0

    assert abs(
        policy.sum() - 1.0
    ) < 1e-5