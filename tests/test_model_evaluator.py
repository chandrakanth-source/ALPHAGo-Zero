"""
Test for ModelEvaluator.compare_models – self-contained version.

Uses tmp_path to create temporary model files so it does not depend on
the on-disk state of models/model_iteration_N.pt.
"""

import os

import pytest
import torch

from evaluation.model_evaluator import ModelEvaluator
from network.network import GoNetwork


def test_model_evaluator(tmp_path):
    board_size = 12

    # Create two temporary 12×12 models.
    model_a = GoNetwork(board_size=board_size)
    model_b = GoNetwork(board_size=board_size)

    old_path = str(tmp_path / "model_old.pt")
    new_path = str(tmp_path / "model_new.pt")

    torch.save(model_a.state_dict(), old_path)
    torch.save(model_b.state_dict(), new_path)

    evaluator = ModelEvaluator(board_size=board_size)
    result = evaluator.compare_models(old_path, new_path)

    assert result is not None
    assert result["old_parameters"] > 0
    assert result["new_parameters"] > 0