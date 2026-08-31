import torch


def main():

    data = torch.load(
        "data/self_play_data.pt",
        weights_only=False
    )

    print("Number of examples:", len(data))

    if len(data) == 0:
        print("Dataset is empty.")
        return

    state, policy, value = data[0]

    print("State type:", type(state))
    print("Policy type:", type(policy))
    print("Value:", value)

    print("State shape:", state.shape)
    print("Policy shape:", policy.shape)


if __name__ == "__main__":
    main()