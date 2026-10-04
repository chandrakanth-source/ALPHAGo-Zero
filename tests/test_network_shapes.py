"""
Day 27 - Network I/O shape validation tests.

Checks:
  input  = (batch, 3, 12, 12)
  policy = (batch, 145)   [ 144 board positions + 1 pass ]
  value  = (batch, 1)
"""

import pytest
import torch

from network.network import GoNetwork


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model_12():
    m = GoNetwork(board_size=12)
    m.eval()
    return m


@pytest.fixture(scope="module")
def model_5():
    m = GoNetwork(board_size=5)
    m.eval()
    return m


# ---------------------------------------------------------------------------
# 1. Default 12x12 board shapes
# ---------------------------------------------------------------------------

def test_network_policy_shape_12x12(model_12):
    """Policy output for 12x12 board must be (B, 145)."""
    x = torch.randn(2, 3, 12, 12)
    with torch.no_grad():
        policy, value = model_12(x)
    assert policy.shape == (2, 145), f"Policy shape {policy.shape} != (2, 145)"


def test_network_value_shape_12x12(model_12):
    """Value output for 12x12 board must be (B, 1)."""
    x = torch.randn(2, 3, 12, 12)
    with torch.no_grad():
        policy, value = model_12(x)
    assert value.shape == (2, 1), f"Value shape {value.shape} != (2, 1)"


def test_network_policy_sums_to_one_12x12(model_12):
    """Policy must sum to 1 along action dimension (softmax applied)."""
    x = torch.randn(4, 3, 12, 12)
    with torch.no_grad():
        policy, _ = model_12(x)
    sums = policy.sum(dim=1)
    assert torch.allclose(sums, torch.ones(4), atol=1e-5), (
        f"Policy rows don't sum to 1: {sums}"
    )


def test_network_value_in_range_12x12(model_12):
    """Value output must be in [-1, 1] (tanh applied)."""
    x = torch.randn(4, 3, 12, 12)
    with torch.no_grad():
        _, value = model_12(x)
    assert (value >= -1.0).all() and (value <= 1.0).all(), (
        f"Value out of range: min={value.min()}, max={value.max()}"
    )


# ---------------------------------------------------------------------------
# 2. Single-sample batch (batch=1)
# ---------------------------------------------------------------------------

def test_single_sample_shapes(model_12):
    """Batch size 1 must work correctly."""
    x = torch.randn(1, 3, 12, 12)
    with torch.no_grad():
        policy, value = model_12(x)
    assert policy.shape == (1, 145)
    assert value.shape == (1, 1)


# ---------------------------------------------------------------------------
# 3. 5x5 board (used in fast tests)
# ---------------------------------------------------------------------------

def test_network_policy_shape_5x5(model_5):
    x = torch.randn(3, 3, 5, 5)
    with torch.no_grad():
        policy, value = model_5(x)
    assert policy.shape == (3, 26), f"Policy shape {policy.shape} != (3, 26)"
    assert value.shape == (3, 1), f"Value shape {value.shape} != (3, 1)"


# ---------------------------------------------------------------------------
# 4. Network parameter count is reasonable
# ---------------------------------------------------------------------------

def test_network_has_parameters(model_12):
    """Model must have trainable parameters."""
    n = sum(p.numel() for p in model_12.parameters())
    assert n > 0, "Model has no parameters"
    # For a 12x12 board the model should have at least 50k parameters
    assert n > 50_000, f"Model has suspiciously few parameters: {n}"


# ---------------------------------------------------------------------------
# 5. Forward is deterministic in eval mode
# ---------------------------------------------------------------------------

def test_network_deterministic_in_eval(model_12):
    """Same input in eval mode must give identical output."""
    model_12.eval()
    x = torch.randn(1, 3, 12, 12)
    with torch.no_grad():
        p1, v1 = model_12(x)
        p2, v2 = model_12(x)
    assert torch.allclose(p1, p2), "Policy differs across two forward passes"
    assert torch.allclose(v1, v2), "Value differs across two forward passes"
