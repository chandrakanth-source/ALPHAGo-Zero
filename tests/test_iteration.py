from iterations.iteration import (
    AlphaGoZeroIteration
)


def test_iteration_creation():

    iteration = AlphaGoZeroIteration(
        board_size=12,
        simulations=5
    )

    assert iteration is not None

    assert iteration.board_size == 12

    assert iteration.simulations == 5