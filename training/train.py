import torch
from torch.utils.data import DataLoader

from network.network import GoNetwork
from training.dataset import SelfPlayDataset
from training.trainer import Trainer


def main():

    print("=" * 50)
    print("ALPHAGO ZERO TRAINING")
    print("=" * 50)

    # ---------------------------------
    # Load self-play data
    # ---------------------------------

    examples = torch.load(
        "data/self_play_data.pt",
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
        batch_size=32,
        shuffle=True
    )

    # ---------------------------------
    # Network
    # ---------------------------------

    model = GoNetwork(
        board_size=12
    )

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

    epochs = 5

    for epoch in range(epochs):

        metrics = trainer.train_epoch(
            dataloader
        )

        print()
        print(
            f"Epoch "
            f"{epoch + 1}/{epochs}"
        )

        print(
            f"Total Loss: "
            f"{metrics['loss']:.4f}"
        )

        print(
            f"Policy Loss: "
            f"{metrics['policy_loss']:.4f}"
        )

        print(
            f"Value Loss: "
            f"{metrics['value_loss']:.4f}"
        )

    # ---------------------------------
    # Save model
    # ---------------------------------

    torch.save(
        model.state_dict(),
        "models/latest_model.pt"
    )

    print()
    print(
        "Model saved to:"
    )

    print(
        "models/latest_model.pt"
    )


if __name__ == "__main__":
    main()