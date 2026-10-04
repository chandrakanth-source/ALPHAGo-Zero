"""
Train the network for one AlphaGo Zero iteration.

Loads a self-play dataset, trains a (possibly warm-started) GoNetwork,
saves epoch-level checkpoints for resumability, and writes the final
candidate model to ``models/model_iteration_{iteration}.pt``.
"""

import os
import torch

from torch.utils.data import DataLoader

from network.network import GoNetwork
from network.model_loader import load_model
from training.dataset import SelfPlayDataset
from training.trainer import Trainer


def train_iteration(
    data_path,
    iteration,
    board_size=12,
    epochs=5,
    batch_size=32,
    model_path=None,
    learning_rate=0.001,
    checkpoint_dir="training/checkpoints",
    resume=True,
):
    """
    Train the network on self-play data for one AlphaGo Zero iteration.

    Args:
        data_path:      Path to the ``.pt`` self-play dataset.
        iteration:      Outer-loop iteration number (used for filenames).
        board_size:     Board size (must match the dataset).
        epochs:         Number of training epochs.
        batch_size:     Mini-batch size.
        model_path:     If given, warm-start from this checkpoint / weights
                        file instead of random initialisation.
        learning_rate:  SGD learning rate.
        checkpoint_dir: Directory for epoch-level checkpoints.
        resume:         If *True*, check *checkpoint_dir* for an existing
                        checkpoint for this *iteration* and resume from it.

    Returns:
        str: Path of the saved candidate model file.
    """
    print()
    print("=" * 60)
    print(f"TRAINING ITERATION {iteration}")
    print("=" * 60)

    # ---- Dataset ----
    examples = torch.load(
        data_path,
        map_location="cpu",
        weights_only=False,
    )
    print(f"Loaded {len(examples)} examples")

    dataset = SelfPlayDataset(examples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ---- Model initialisation ----
    if model_path is not None and os.path.exists(model_path):
        model = load_model(model_path, board_size=board_size)
        print(f"Warm-started from: {model_path}")
    else:
        model = GoNetwork(board_size=board_size)
        print("Initialised fresh model")

    # ---- Trainer ----
    trainer = Trainer(
        model,
        learning_rate=learning_rate,
        checkpoint_dir=checkpoint_dir,
    )

    # ---- Optional resume ----
    start_epoch = 0
    if resume:
        from training.checkpoint_manager import CheckpointManager
        ckpt_manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
        if ckpt_manager.checkpoint_exists():
            try:
                payload = ckpt_manager.load_latest(
                    model=model,
                    optimizer=trainer.optimizer,
                    device=str(trainer.device),
                )
                if payload is not None and payload.get("iteration") == iteration:
                    start_epoch = int(payload.get("epoch", 0)) + 1
                    print(f"Resumed from checkpoint at epoch {start_epoch}")
            except RuntimeError:
                print("WARNING: Checkpoint board-size mismatch - starting from epoch 0")

    # ---- Training loop ----
    for epoch in range(start_epoch, epochs):
        metrics = trainer.train_epoch(dataloader)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {metrics['loss']:.4f} | "
            f"Policy: {metrics['policy_loss']:.4f} | "
            f"Value: {metrics['value_loss']:.4f}"
        )

        # Save epoch checkpoint.
        ckpt_path = os.path.join(
            checkpoint_dir,
            f"checkpoint_iter{iteration:04d}_epoch{epoch:04d}.pt",
        )
        trainer.save_checkpoint(
            path=ckpt_path,
            epoch=epoch,
            iteration=iteration,
            metrics=metrics,
        )

    # ---- Save candidate model ----
    os.makedirs("models", exist_ok=True)

    candidate_path = f"models/model_iteration_{iteration}.pt"

    torch.save(model.state_dict(), candidate_path)

    print()
    print(f"Model saved to: {candidate_path}")

    return candidate_path