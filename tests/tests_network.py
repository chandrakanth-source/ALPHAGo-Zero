import torch

from network.network import GoNetwork


def test_network_output():

    board_size = 12

    model = GoNetwork(board_size)

    # Fake Go state
    x = torch.randn(
        1,
        3,
        board_size,
        board_size
    )

    policy, value = model(x)

    print("Policy shape:", policy.shape)
    print("Value shape:", value.shape)

    assert policy.shape == (1, 145)

    assert value.shape == (1, 1)
    probability_sum = policy.sum().item()

    print("Probability sum:", probability_sum)

    assert abs(probability_sum - 1.0) < 1e-5