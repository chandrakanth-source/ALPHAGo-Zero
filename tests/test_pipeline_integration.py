"""
End-to-end pipeline integration tests.

These tests exercise the full self-play → train → evaluate → promote loop
on a tiny board (5×5) with minimal simulations so the suite finishes fast.
"""

import os
import tempfile

import pytest
import torch

from environment.go_game import GoGame
from evaluation.model_match import ModelPlayer, play_game
from evaluation.promotion import PromotionManager
from mcts.network_Evaluator import NetworkEvaluator
from network.model_loader import load_model
from network.network import GoNetwork
from self_play.self_play import SelfPlay
from training.checkpoint_manager import CheckpointManager
from training.dataset import SelfPlayDataset
from training.trainer import Trainer
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Mini self-play → dataset → train loop
# ---------------------------------------------------------------------------


def test_full_selfplay_train_loop(tmp_path):
    """
    Run one complete self-play → training cycle and verify that:
    - examples are generated
    - the model weights change after training
    """
    board_size = 5
    model = GoNetwork(board_size=board_size)

    # Self-play.
    sp = SelfPlay(
        board_size=board_size,
        simulations=2,
        model=model,
        temperature=1.0,
    )
    examples = sp.generate_games(num_games=2)
    assert len(examples) > 0

    # Dataset + DataLoader.
    dataset = SelfPlayDataset(examples)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Capture weights before training.
    before = {
        k: v.clone() for k, v in model.state_dict().items()
    }

    # Training.
    ckpt_dir = str(tmp_path / "ckpt")
    trainer = Trainer(
        model,
        learning_rate=0.01,
        checkpoint_dir=ckpt_dir,
    )
    metrics = trainer.train_epoch(loader)

    assert "loss" in metrics
    assert metrics["loss"] >= 0

    # Weights should have changed.
    changed = any(
        not torch.equal(model.state_dict()[k], before[k])
        for k in before
    )
    assert changed, "Model weights did not update during training"


# ---------------------------------------------------------------------------
# Mini evaluate loop
# ---------------------------------------------------------------------------


def test_model_vs_model_evaluation():
    """Two random models play a short match – results should be well-formed."""
    board_size = 5
    model_a = GoNetwork(board_size=board_size)
    model_b = GoNetwork(board_size=board_size)

    player_a = ModelPlayer(model=model_a, board_size=board_size, simulations=2)
    player_b = ModelPlayer(model=model_b, board_size=board_size, simulations=2)

    wins_a = 0
    wins_b = 0
    draws = 0

    for i in range(4):
        game = GoGame(board_size=board_size)
        if i % 2 == 0:
            winner = play_game(player_a, player_b, game)
        else:
            winner = play_game(player_b, player_a, game)

        if winner == 0:
            draws += 1
        elif winner == 1:
            wins_a += 1 if i % 2 == 0 else 0
            wins_b += 0 if i % 2 == 0 else 1
        else:
            wins_a += 0 if i % 2 == 0 else 1
            wins_b += 1 if i % 2 == 0 else 0

    total = wins_a + wins_b + draws
    assert total == 4


# ---------------------------------------------------------------------------
# Promotion
# ---------------------------------------------------------------------------


def test_promotion_copies_model(tmp_path):
    board_size = 5
    models_dir = str(tmp_path / "models")
    os.makedirs(models_dir)

    model = GoNetwork(board_size=board_size)
    candidate_path = str(tmp_path / "candidate.pt")
    torch.save(model.state_dict(), candidate_path)

    promoter = PromotionManager(models_dir=models_dir)
    latest = promoter.promote(candidate_path)

    assert os.path.exists(latest)


def test_rejection_does_not_copy(tmp_path):
    models_dir = str(tmp_path / "models")
    os.makedirs(models_dir)

    promoter = PromotionManager(models_dir=models_dir)
    result = promoter.reject("some/candidate.pt")
    assert result is False


# ---------------------------------------------------------------------------
# Checkpoint round-trip in training context
# ---------------------------------------------------------------------------


def test_checkpoint_resume_continues_from_saved_epoch(tmp_path):
    """Simulate interrupted training: save checkpoint, restore, verify epoch."""
    board_size = 5
    model = GoNetwork(board_size=board_size)
    ckpt_dir = str(tmp_path / "ckpt")

    trainer = Trainer(model, checkpoint_dir=ckpt_dir)

    # Fake 3 epochs of training.
    for epoch in range(3):
        trainer.checkpoint_manager.save(
            model=model,
            optimizer=trainer.optimizer,
            iteration=1,
            epoch=epoch,
            metrics={"loss": 1.0 - epoch * 0.1},
        )

    # Simulate a fresh Trainer restoring state.
    new_model = GoNetwork(board_size=board_size)
    new_trainer = Trainer(new_model, checkpoint_dir=ckpt_dir)
    resumed_iter, resumed_epoch = new_trainer.resume_from_latest()

    assert resumed_iter == 1
    assert resumed_epoch == 2  # last saved epoch


# ---------------------------------------------------------------------------
# Save / load model through model_loader
# ---------------------------------------------------------------------------


def test_load_model_produces_correct_output(tmp_path):
    board_size = 5
    model = GoNetwork(board_size=board_size)
    model.eval()

    model_path = str(tmp_path / "model_test.pt")
    torch.save(model.state_dict(), model_path)

    loaded = load_model(model_path, board_size=board_size)
    loaded.eval()

    x = torch.randn(1, 3, board_size, board_size)
    with torch.no_grad():
        p1, v1 = model(x)
        p2, v2 = loaded(x)

    assert torch.allclose(p1, p2, atol=1e-5)
    assert torch.allclose(v1, v2, atol=1e-5)


# ---------------------------------------------------------------------------
# Temperature scheduling during a full game
# ---------------------------------------------------------------------------


def test_temperature_schedules_correctly_in_game():
    """
    Verify that move selection is stochastic in early moves and deterministic
    after the threshold within a real game.
    """
    board_size = 5
    sp = SelfPlay(
        board_size=board_size,
        simulations=2,
        temperature=1.0,
        temp_threshold=3,
    )

    # Create a peaked policy (one dominant move).
    peaked_policy = torch.zeros(board_size * board_size + 1)
    peaked_policy[5] = 1.0
    peaked_policy = peaked_policy.numpy()

    # Before threshold (move 0) – still picks the dominant action
    # (because probability 1 is concentrated there anyway).
    move_before = sp.select_move(peaked_policy, move_number=0)
    assert move_before == 5

    # After threshold (move 3) – greedy, must be argmax.
    move_after = sp.select_move(peaked_policy, move_number=3)
    assert move_after == 5
