from network.model_loader import load_model
import torch
def test_model_has_parameters():

    model = load_model(
        "models/latest_model.pt"
    )

    parameters = list(
        model.parameters()
    )

    assert len(parameters) > 0

    total_parameters = sum(
        p.numel()
        for p in parameters
    )

    assert total_parameters > 0

    print(
        "Total parameters:",
        total_parameters
    )
def test_model_loader():

    model = load_model(
        "models/latest_model.pt"
    )

    assert model is not None

    model.eval()

    print(
        "Trained model loaded successfully."
    )
def test_loaded_model_prediction():

    model = load_model(
        "models/latest_model.pt"
    )

    model.eval()

    board_size = 12

    state = torch.zeros(
        1,
        3,
        board_size,
        board_size,
        dtype=torch.float32
    )

    with torch.no_grad():

        policy, value = model(state)

    assert policy is not None
    assert value is not None

    assert policy.shape[0] == 1
    assert value.shape[0] == 1