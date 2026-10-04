"""
Test for train_iteration() – self-contained version.

Generates a small synthetic dataset in a temp dir and verifies that
train_iteration() produces a model file without errors.
"""

import os
import tempfile

import numpy as np
import pytest
import torch

from training.train_iteration import train_iteration
from network.network import GoNetwork


def test_train_iteration(tmp_path):
    board_size = 5
    action_space = board_size * board_size + 1  # 26

    # Create 10 synthetic training examples.
    examples = []
    for _ in range(10):
        state = np.random.rand(3, board_size, board_size).astype(np.float32)
        policy = np.ones(action_space, dtype=np.float32) / action_space
        value = float(np.random.choice([-1.0, 0.0, 1.0]))
        examples.append((state, policy, value))

    data_path = str(tmp_path / "synthetic_data.pt")
    torch.save(examples, data_path)

    # Save a compatible model for warm-start.
    seed_model = GoNetwork(board_size=board_size)
    model_path = str(tmp_path / "seed_model.pt")
    torch.save(seed_model.state_dict(), model_path)

    # Override checkpoint dir to avoid clashing with other tests.
    ckpt_dir = str(tmp_path / "checkpoints")

    candidate = train_iteration(
        data_path=data_path,
        iteration=1,
        board_size=board_size,
        epochs=2,
        batch_size=4,
        model_path=model_path,
        checkpoint_dir=ckpt_dir,
        resume=False,  # no resume — fresh run
    )

    # The returned path should point to an existing model file.
    assert os.path.exists(candidate), f"Candidate model not found at {candidate}"

    # The model must be loadable back.
    from network.model_loader import load_model
    loaded = load_model(candidate, board_size=board_size)
    assert loaded is not None