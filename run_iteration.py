from iterations.iteration import (
    AlphaGoZeroIteration
)


def main():

    iteration_runner = (
        AlphaGoZeroIteration(
            board_size=12,
            simulations=5
        )
    )

    model_path = (
        iteration_runner
        .run_iteration(
            model_path=
            "models/latest_model.pt",

            iteration=1,

            num_games=5,
            epochs=5
        )
    )

    print()
    print(
        "Iteration completed."
    )

    print(
        f"Model: {model_path}"
    )


if __name__ == "__main__":
    main()