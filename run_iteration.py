from iterations.iteration import (
    AlphaGoZeroIteration
)
from training.train_iteration import (
    train_iteration
)

def main():
    iteration_number=2
    iteration_runner = (
        AlphaGoZeroIteration(
            board_size=12,
            simulations=5
        )
    )

    data_path = (
    iteration_runner
    .run_self_play_iteration(
        model_path=
        "models/model_iteration_1.pt",

        iteration=iteration_number,

        num_games=5
    )
   )
    model_path = train_iteration(
    data_path=data_path,

    iteration=iteration_number,

    board_size=12,

    epochs=5,

    batch_size=32
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