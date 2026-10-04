"""
Tests for CheckpointManager and Trainer checkpoint / resume functionality.
"""

import os
import tempfile

import pytest
import torch

from network.network import GoNetwork
from training.checkpoint_manager import CheckpointManager
from training.trainer import Trainer


# ---------------------------------------------------------------------------
# CheckpointManager – unit tests
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_ckpt_dir(tmp_path):
    return str(tmp_path / "checkpoints")


@pytest.fixture
def tiny_model():
    return GoNetwork(board_size=5)


@pytest.fixture
def optimizer(tiny_model):
    return torch.optim.SGD(tiny_model.parameters(), lr=0.01)


def test_checkpoint_manager_creates_dir(tmp_ckpt_dir):
    ckpt = CheckpointManager(checkpoint_dir=tmp_ckpt_dir)
    assert os.path.isdir(tmp_ckpt_dir)


def test_checkpoint_not_exists_initially(tmp_ckpt_dir):
    ckpt = CheckpointManager(checkpoint_dir=tmp_ckpt_dir)
    assert not ckpt.checkpoint_exists()


def test_save_creates_file(tmp_ckpt_dir, tiny_model, optimizer):
    ckpt = CheckpointManager(checkpoint_dir=tmp_ckpt_dir)
    path = ckpt.save(
        model=tiny_model,
        optimizer=optimizer,
        iteration=1,
        epoch=0,
        metrics={"loss": 0.5},
    )
    assert os.path.exists(path)


def test_save_creates_latest_shortcut(tmp_ckpt_dir, tiny_model, optimizer):
    ckpt = CheckpointManager(checkpoint_dir=tmp_ckpt_dir)
    ckpt.save(tiny_model, optimizer, iteration=1, epoch=0)
    assert ckpt.checkpoint_exists()


def test_load_latest_restores_weights(tmp_ckpt_dir, tiny_model, optimizer):
    ckpt = CheckpointManager(checkpoint_dir=tmp_ckpt_dir)

    # Modify weights in a known way.
    with torch.no_grad():
        for p in tiny_model.parameters():
            p.fill_(3.14)

    ckpt.save(tiny_model, optimizer, iteration=2, epoch=1)

    # Create a fresh model (random weights).
    fresh = GoNetwork(board_size=5)

    payload = ckpt.load_latest(fresh)

    assert payload is not None
    assert payload["iteration"] == 2
    assert payload["epoch"] == 1

    # Fresh model should now have the saved weights.
    for p_orig, p_fresh in zip(tiny_model.parameters(), fresh.parameters()):
        assert torch.allclose(p_orig, p_fresh)


def test_latest_iteration_returns_correct_value(tmp_ckpt_dir, tiny_model, optimizer):
    ckpt = CheckpointManager(checkpoint_dir=tmp_ckpt_dir)
    assert ckpt.latest_iteration() == 0

    ckpt.save(tiny_model, optimizer, iteration=7, epoch=2)
    assert ckpt.latest_iteration() == 7


def test_latest_epoch_returns_correct_value(tmp_ckpt_dir, tiny_model, optimizer):
    ckpt = CheckpointManager(checkpoint_dir=tmp_ckpt_dir)
    ckpt.save(tiny_model, optimizer, iteration=1, epoch=4)
    assert ckpt.latest_epoch() == 4


def test_pruning_removes_old_checkpoints(tmp_ckpt_dir, tiny_model, optimizer):
    ckpt = CheckpointManager(checkpoint_dir=tmp_ckpt_dir, max_to_keep=2)

    for epoch in range(5):
        ckpt.save(tiny_model, optimizer, iteration=1, epoch=epoch)

    # Only 2 regular checkpoint files should remain (plus latest).
    all_files = [
        f
        for f in os.listdir(tmp_ckpt_dir)
        if f.startswith("checkpoint_iter") and f.endswith(".pt")
    ]
    assert len(all_files) <= 2


def test_list_checkpoints(tmp_ckpt_dir, tiny_model, optimizer):
    ckpt = CheckpointManager(checkpoint_dir=tmp_ckpt_dir)
    assert ckpt.list_checkpoints() == []

    ckpt.save(tiny_model, optimizer, iteration=1, epoch=0)
    ckpt.save(tiny_model, optimizer, iteration=1, epoch=1)

    listing = ckpt.list_checkpoints()
    assert len(listing) == 2


def test_metrics_embedded_in_checkpoint(tmp_ckpt_dir, tiny_model, optimizer):
    ckpt = CheckpointManager(checkpoint_dir=tmp_ckpt_dir)
    ckpt.save(
        tiny_model,
        optimizer,
        iteration=3,
        epoch=0,
        metrics={"loss": 1.23, "policy_loss": 0.5},
    )
    payload = ckpt.load_latest(GoNetwork(board_size=5))
    assert payload["metrics"]["loss"] == pytest.approx(1.23)


# ---------------------------------------------------------------------------
# Trainer – checkpoint integration
# ---------------------------------------------------------------------------


def test_trainer_save_and_load_checkpoint(tmp_path, tiny_model):
    ckpt_dir = str(tmp_path / "ckpt")
    trainer = Trainer(tiny_model, checkpoint_dir=ckpt_dir)

    # Snapshot original weights.
    orig_weights = {
        k: v.clone() for k, v in tiny_model.state_dict().items()
    }

    path = str(tmp_path / "test_ckpt.pt")
    trainer.save_checkpoint(path=path, epoch=3)

    # Mutate model.
    with torch.no_grad():
        for p in tiny_model.parameters():
            p.fill_(0.0)

    # Load back.
    trainer.load_checkpoint(path)

    # Weights should be restored.
    for k, v in tiny_model.state_dict().items():
        assert torch.allclose(v, orig_weights[k]), f"Mismatch in {k}"


def test_trainer_resume_from_latest(tmp_path, tiny_model):
    ckpt_dir = str(tmp_path / "ckpt")
    trainer = Trainer(tiny_model, checkpoint_dir=ckpt_dir)

    # No checkpoint yet → returns (0, 0).
    iteration, epoch = trainer.resume_from_latest()
    assert iteration == 0
    assert epoch == 0

    # Save one.
    trainer.checkpoint_manager.save(
        tiny_model, trainer.optimizer, iteration=5, epoch=2
    )

    iteration, epoch = trainer.resume_from_latest()
    assert iteration == 5
    assert epoch == 2
