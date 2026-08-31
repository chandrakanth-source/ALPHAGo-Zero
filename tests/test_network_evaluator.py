import numpy as np

from network.model_loader import load_model
from mcts.network_Evaluator import NetworkEvaluator


def test_network_evaluator():

    model = load_model(
        "models/latest_model.pt"
    )

    evaluator = NetworkEvaluator(
        model
    )

    state = np.zeros(
        (
            3,
            12,
            12
        ),
        dtype=np.float32
    )

    policy, value = evaluator.evaluate(
        state
    )

    assert policy is not None
    assert value is not None

    assert np.isfinite(
        policy
    ).all()

    assert np.isfinite(
        value
    )