"""
Day 26 – Automatic pipeline integration test.

Runs one iteration of the complete self-play → train → evaluate → promote
loop using a 5×5 board and minimal settings so the test completes quickly.
"""

import os
import sys
import tempfile

import pytest
import torch

# Make iteration/ importable
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def test_run_pipeline_one_iteration(tmp_path, monkeypatch):
    """
    Run one iteration of the automatic pipeline end-to-end.

    Uses a temporary directory for all outputs so the real project files
    (models/, data/, training/checkpoints/) are not affected.
    """
    import main as rip

    models_dir = str(tmp_path / "models")
    data_dir = str(tmp_path / "data")
    ckpt_dir = str(tmp_path / "checkpoints")
    os.makedirs(models_dir)
    os.makedirs(data_dir)
    os.makedirs(ckpt_dir)

    # Bootstrap the expected latest_model.pt for board_size=5
    from network.network import GoNetwork
    bootstrap = GoNetwork(board_size=5)
    latest_path = os.path.join(models_dir, "latest_model.pt")
    torch.save(bootstrap.state_dict(), latest_path)

    monkeypatch.setattr(rip, "MODELS_DIR", models_dir)
    monkeypatch.setattr(rip, "DATA_DIR", data_dir)
    monkeypatch.setattr(rip, "CHECKPOINT_DIR", ckpt_dir)
    monkeypatch.setattr(rip, "LATEST_MODEL", latest_path)
    monkeypatch.setattr(rip, "LATEST_MODEL_PATH", latest_path)

    rip.run_pipeline(
        num_iterations=1,
        board_size=5,
        self_play_games=1,
        simulations=5,
        epochs=1,
        batch_size=8,
        eval_games=2,
        promotion_threshold=0.0,    # always promote
        learning_rate=0.001,
        force_fresh=True,
        start_iteration=1,
    )

    # After one iteration a candidate model must exist.
    candidate = os.path.join(models_dir, "model_iteration_1.pt")
    assert os.path.exists(candidate), "model_iteration_1.pt was not created"

    # latest_model.pt must be loadable as a 5×5 GoNetwork.
    from network.model_loader import load_model
    model = load_model(latest_path, board_size=5)
    assert model is not None
