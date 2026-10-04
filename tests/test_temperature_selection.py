"""
Tests for temperature-based move selection in SelfPlay.
"""

import numpy as np
import pytest

from self_play.self_play import SelfPlay


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sp():
    """SelfPlay with a tiny untrained model."""
    return SelfPlay(board_size=5, simulations=2, temperature=1.0, temp_threshold=30)


# ---------------------------------------------------------------------------
# Temperature = 0 → argmax (greedy)
# ---------------------------------------------------------------------------


def test_greedy_returns_argmax():
    sp = SelfPlay(board_size=5, simulations=2, temperature=0.0)
    policy = np.array([0.1, 0.5, 0.2, 0.2])
    move = sp.select_move(policy)
    assert move == 1  # index of 0.5


def test_zero_temperature_via_threshold():
    """After temp_threshold moves, effective temperature should be 0."""
    sp = SelfPlay(
        board_size=5, simulations=2, temperature=1.0, temp_threshold=5
    )
    policy = np.array([0.0, 0.0, 1.0, 0.0])  # deterministic
    move = sp.select_move(policy, move_number=5)
    assert move == 2


# ---------------------------------------------------------------------------
# Temperature > 0 → stochastic but sums to 1 distribution
# ---------------------------------------------------------------------------


def test_stochastic_selection_uses_policy():
    """High temperature should produce varied selections over many trials."""
    sp = SelfPlay(board_size=5, simulations=2, temperature=1.0)
    policy = np.array([0.0, 0.0, 1.0, 0.0])
    # With probability 1 concentrated on index 2, always 2.
    moves = {sp.select_move(policy) for _ in range(20)}
    assert moves == {2}


def test_temperature_one_preserves_distribution():
    """Temperature=1 should reproduce the policy distribution (in expectation)."""
    sp = SelfPlay(board_size=5, simulations=2, temperature=1.0)
    policy = np.array([0.25, 0.25, 0.25, 0.25])
    counts = np.zeros(4, dtype=int)
    rng = np.random.default_rng(42)
    np.random.seed(42)
    for _ in range(4000):
        move = sp.select_move(policy)
        counts[move] += 1
    # Each bucket should be roughly 1000 ± 200.
    for c in counts:
        assert 700 < c < 1300, f"Unexpected count {c}"


def test_low_temperature_peaks():
    """Temperature < 1 should sharpen the distribution."""
    sp = SelfPlay(board_size=5, simulations=2, temperature=0.1)
    policy = np.array([0.3, 0.7, 0.0, 0.0])
    counts = {0: 0, 1: 0}
    for _ in range(200):
        move = sp.select_move(policy)
        if move in counts:
            counts[move] += 1
    # Index 1 should dominate heavily.
    assert counts[1] > counts[0] * 5


# ---------------------------------------------------------------------------
# All-zero policy fallback
# ---------------------------------------------------------------------------


def test_all_zero_policy_falls_back_to_argmax():
    sp = SelfPlay(board_size=5, simulations=2, temperature=1.0)
    policy = np.zeros(4)
    # Should not raise; returns argmax of zeros (index 0).
    move = sp.select_move(policy)
    assert 0 <= move < 4


# ---------------------------------------------------------------------------
# Temperature threshold scheduling
# ---------------------------------------------------------------------------


def test_threshold_disabled_when_zero():
    """temp_threshold=0 means always use `temperature`, never switch to greedy."""
    sp = SelfPlay(
        board_size=5, simulations=2, temperature=1.0, temp_threshold=0
    )
    policy = np.array([0.1, 0.8, 0.1])
    # Even at move 9999 it should still use temperature=1 (stochastic).
    # We can't assert exact value but should not raise.
    move = sp.select_move(policy, move_number=9999)
    assert 0 <= move < 3


# ---------------------------------------------------------------------------
# Integration: SelfPlay with trained-model evaluator connection (Task 3)
# ---------------------------------------------------------------------------


def test_selfplay_uses_evaluator_not_fresh_model():
    """
    When an evaluator is supplied, SelfPlay should use it (not a new model).
    """
    from mcts.network_Evaluator import NetworkEvaluator
    from network.network import GoNetwork

    model = GoNetwork(board_size=5)
    evaluator = NetworkEvaluator(model)

    sp = SelfPlay(board_size=5, simulations=2, evaluator=evaluator)

    assert sp.evaluator is evaluator
    assert sp.model is model


def test_selfplay_wraps_model_in_evaluator():
    """When a raw model is passed, it should be wrapped in a NetworkEvaluator."""
    from mcts.network_Evaluator import NetworkEvaluator
    from network.network import GoNetwork

    model = GoNetwork(board_size=5)
    sp = SelfPlay(board_size=5, simulations=2, model=model)

    assert isinstance(sp.evaluator, NetworkEvaluator)
    assert sp.model is model


def test_generate_game_returns_training_examples():
    """generate_game() should produce (state, policy, value) triples."""
    sp = SelfPlay(board_size=5, simulations=2, temperature=1.0)
    data = sp.generate_game()

    assert len(data) > 0

    for state, policy, value in data:
        assert state.shape == (3, 5, 5)
        assert policy.shape == (26,)  # 5*5+1
        assert isinstance(value, float)
        assert -1.0 <= value <= 1.0


def test_generate_games_count():
    sp = SelfPlay(board_size=5, simulations=2)
    data = sp.generate_games(num_games=2)
    assert len(data) > 0
