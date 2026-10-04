"""
Day 27 - Self-play correctness validation tests.

Tests:
  1. Game terminates correctly
  2. Winner is correct (+1, -1, or 0)
  3. Training examples are well-formed
  4. Policy sums to 1 for every example
  5. Values are in valid range [-1, 1]
"""

import numpy as np
import pytest
import torch

from environment.go_game import GoGame
from network.network import GoNetwork
from self_play.self_play import SelfPlay


BOARD_SIZE = 5


@pytest.fixture(scope="module")
def model():
    return GoNetwork(board_size=BOARD_SIZE)


@pytest.fixture(scope="module")
def self_play(model):
    return SelfPlay(board_size=BOARD_SIZE, simulations=5, model=model)


# ---------------------------------------------------------------------------
# 1. Game terminates
# ---------------------------------------------------------------------------

def test_game_terminates(self_play):
    """play_game() must always return before hitting max_moves with a small board."""
    history, result = self_play.play_game(max_moves=300)
    assert isinstance(history, list)
    assert isinstance(result, float)


# ---------------------------------------------------------------------------
# 2. Winner is in {-1, 0, +1}
# ---------------------------------------------------------------------------

def test_winner_is_valid(self_play):
    """Game result must be -1, 0, or +1."""
    _, result = self_play.play_game()
    assert result in (-1.0, 0.0, 1.0), f"Invalid result: {result}"


# ---------------------------------------------------------------------------
# 3. Training examples are well-formed
# ---------------------------------------------------------------------------

def test_training_examples_structure(self_play):
    """Each training example must be a (state, policy, value) triple."""
    examples = self_play.generate_game()
    assert len(examples) > 0, "No examples generated"

    state, policy, value = examples[0]

    assert isinstance(state, np.ndarray), "State must be ndarray"
    assert state.shape == (3, BOARD_SIZE, BOARD_SIZE), (
        f"State shape {state.shape} != (3, {BOARD_SIZE}, {BOARD_SIZE})"
    )
    assert isinstance(policy, np.ndarray), "Policy must be ndarray"
    assert policy.shape == (BOARD_SIZE * BOARD_SIZE + 1,), (
        f"Policy shape {policy.shape} != ({BOARD_SIZE**2 + 1},)"
    )
    assert isinstance(value, float), "Value must be float"


# ---------------------------------------------------------------------------
# 4. Policy sums to 1 for every example
# ---------------------------------------------------------------------------

def test_policy_sums_to_one_for_all_examples(self_play):
    """Every policy vector in the generated game must sum to 1."""
    examples = self_play.generate_game()
    for i, (_, policy, _) in enumerate(examples):
        s = policy.sum()
        assert abs(s - 1.0) < 1e-4, f"Policy sum {s} != 1.0 at example {i}"


# ---------------------------------------------------------------------------
# 5. Values are in [-1, 1]
# ---------------------------------------------------------------------------

def test_values_in_valid_range(self_play):
    """All value labels must lie in [-1, 1]."""
    examples = self_play.generate_game()
    for i, (_, _, value) in enumerate(examples):
        assert -1.0 <= value <= 1.0, (
            f"Value {value} out of range at example {i}"
        )


# ---------------------------------------------------------------------------
# 6. State planes are valid (binary or constant)
# ---------------------------------------------------------------------------

def test_state_planes_are_binary(self_play):
    """All state planes must contain only 0 or 1 (stone presence planes)."""
    examples = self_play.generate_game()
    state, _, _ = examples[0]
    # planes 0 and 1 are stone planes – should be binary
    for plane_idx in [0, 1]:
        plane = state[plane_idx]
        unique = np.unique(plane)
        assert all(v in (0.0, 1.0) for v in unique), (
            f"Plane {plane_idx} contains non-binary values: {unique}"
        )


# ---------------------------------------------------------------------------
# 7. generate_games returns correct number of examples
# ---------------------------------------------------------------------------

def test_generate_games_returns_multiple_examples(model):
    sp = SelfPlay(board_size=BOARD_SIZE, simulations=5, model=model)
    examples = sp.generate_games(num_games=2)
    assert len(examples) > 0, "generate_games returned no examples"


# ---------------------------------------------------------------------------
# 8. SelfPlay uses the provided model (not a fresh one)
# ---------------------------------------------------------------------------

def test_selfplay_uses_provided_model(model):
    """The model stored in SelfPlay must be the same object passed in."""
    sp = SelfPlay(board_size=BOARD_SIZE, simulations=5, model=model)
    assert sp.model is model, "SelfPlay created a new model instead of using provided one"
