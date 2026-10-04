import os
import torch

from network.network import GoNetwork


def load_trained_network(
    board_size=12,
    path="models/latest_model.pt",
    num_res_blocks=10
):
    model = GoNetwork(
        board_size=board_size,
        num_res_blocks=num_res_blocks
    )

    if not os.path.exists(path):
        model.eval()
        return model

    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False
        )
    except Exception:
        model.eval()
        return model

    # Support either a raw state_dict or a checkpoint dictionary.
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

    try:
        model.load_state_dict(checkpoint)
    except Exception:
        # Fallback for state_dict shape mismatches across architecture iterations
        try:
            model.load_state_dict(checkpoint, strict=False)
        except Exception:
            pass

    model.eval()
    return model


def load_model(
    path="models/latest_model.pt",
    board_size=12
):
    return load_trained_network(
        board_size=board_size,
        path=path
    )