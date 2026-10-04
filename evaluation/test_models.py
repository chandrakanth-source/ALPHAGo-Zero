from network.model_loader import load_model


def main():

    old_path = (
        "models/model_iteration_1.pt"
    )

    new_path = (
        "models/model_iteration_2.pt"
    )

    print("=" * 60)
    print("DAY 15 MODEL CHECK")
    print("=" * 60)

    old_model = load_model(
        old_path,
        board_size=12
    )

    print(
        "Model 1 loaded successfully."
    )

    new_model = load_model(
        new_path,
        board_size=12
    )

    print(
        "Model 2 loaded successfully."
    )

    print()
    print(
        "Both models are ready."
    )

    print("=" * 60)


if __name__ == "__main__":

    main()