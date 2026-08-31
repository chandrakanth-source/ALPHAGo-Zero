import torch

from network.network import GoNetwork


def test_saved_model():

    model = GoNetwork(
        board_size=12
    )

    state = torch.load(
        "models/latest_model.pt",
        map_location="cpu"
    )

    model.load_state_dict(
        state
    )

    model.eval()

    assert model is not None