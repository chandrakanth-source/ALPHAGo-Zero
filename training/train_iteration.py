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
    model_path=None
):

    print()
    print("=" * 60)
    print(
        f"TRAINING ITERATION {iteration}"
    )
    print("=" * 60)

    # ---------------------------------
    # Load data
    # ---------------------------------

    examples = torch.load(
        data_path,
        weights_only=False
    )

    print(
        f"Loaded {len(examples)} examples"
    )

    # ---------------------------------
    # Dataset
    # ---------------------------------

    dataset = SelfPlayDataset(
        examples
    )

    # ---------------------------------
    # DataLoader
    # ---------------------------------

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # ---------------------------------
    # Network
    # ---------------------------------

    if model_path is None:
        model = GoNetwork(board_size=board_size)
    else:
        model = load_model(model_path, board_size=board_size)

    # ---------------------------------
    # Trainer
    # ---------------------------------

    trainer = Trainer(
        model,
        learning_rate=0.001
    )

    # ---------------------------------
    # Training
    # ---------------------------------

    for epoch in range(epochs):

        metrics = trainer.train_epoch(
            dataloader
        )

        print(
            f"Epoch "
            f"{epoch + 1}/{epochs} | "
            f"Loss: "
            f"{metrics['loss']:.4f} | "
            f"Policy: "
            f"{metrics['policy_loss']:.4f} | "
            f"Value: "
            f"{metrics['value_loss']:.4f}"
        )

    # ---------------------------------
    # Save model
    # ---------------------------------

    os.makedirs(
        "models",
        exist_ok=True
    )

    model_path = (
        f"models/"
        f"model_iteration_"
        f"{iteration}.pt"
    )

    torch.save(
        model.state_dict(),
        model_path
    )

    print()
    print(
        f"Model saved to: {model_path}"
    )

    return model_path