import os
from config.training_config import (
    BOARD_SIZE,
    SELF_PLAY_GAMES,
    MCTS_SIMULATIONS,
    TRAINING_EPOCHS,
    BATCH_SIZE,
    EVALUATION_GAMES,
    PROMOTION_THRESHOLD
)
from training.train_iteration import (
    train_iteration
)
from iterations.iteration import (
    AlphaGoZeroIteration
)
from network.model_loader import (
    load_model
)
def print_pipeline_status(
    iteration
):

    print()
    print("=" * 70)
    print("PIPELINE STATUS")
    print("=" * 70)

    print(
        f"Current iteration: {iteration}"
    )

    print()
    print("Pipeline:")
    print()
    print(
        "1. Load current model       [READY]"
    )

    print(
        "2. Generate self-play data  [NEXT]"
    )

    print(
        "3. Train new model          [PENDING]"
    )

    print(
        "4. Evaluate new model       [PENDING]"
    )

    print(
        "5. Promote/reject model     [PENDING]"
    )

    print("=" * 70)

def check_model(
    model_path,
    board_size=12
):

    if not os.path.exists(model_path):

        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    model = load_model(
        model_path,
        board_size=board_size
    )

    model.eval()

    print(
        f"Model verified: {model_path}"
    )

    return model
def get_latest_iteration(
    models_dir="models"
):

    iteration = 1

    while True:

        model_path = (
            os.path.join(
                models_dir,
                f"model_iteration_{iteration}.pt"
            )
        )

        if not os.path.exists(model_path):

            break

        iteration += 1

    return iteration - 1
def generate_self_play_data(
    model_path,
    iteration
):

    print()
    print("=" * 70)
    print(
        f"GENERATING SELF-PLAY DATA "
        f"FOR ITERATION {iteration}"
    )
    print("=" * 70)

    print()
    print(
        f"Using model: {model_path}"
    )

    print(
        f"Games: {SELF_PLAY_GAMES}"
    )

    print(
        f"MCTS simulations: "
        f"{MCTS_SIMULATIONS}"
    )

    runner = AlphaGoZeroIteration(
        board_size=BOARD_SIZE,
        simulations=MCTS_SIMULATIONS
    )

    data_path = (
        runner.run_self_play_iteration(
            model_path=model_path,
            iteration=iteration,
            num_games=SELF_PLAY_GAMES
        )
    )

    print()
    print(
        "Self-play completed."
    )

    print(
        f"Dataset: {data_path}"
    )

    return data_path
def main():

    print("=" * 70)
    print("ALPHAGO ZERO - DAY 16")
    print("ITERATIVE TRAINING PIPELINE")
    print("=" * 70)

    board_size = BOARD_SIZE
    print()
    print("TRAINING CONFIGURATION")
    print("-" * 40)

    print(
    f"Board size: {BOARD_SIZE}"
)

    print(
    f"Self-play games: "
    f"{SELF_PLAY_GAMES}"
)

    print(
    f"MCTS simulations: "
    f"{MCTS_SIMULATIONS}"
)

    print(
    f"Training epochs: "
    f"{TRAINING_EPOCHS}"
)

    print(
    f"Batch size: "
    f"{BATCH_SIZE}"
)

    print(
    f"Evaluation games: "
    f"{EVALUATION_GAMES}"
)

    print(
    f"Promotion threshold: "
    f"{PROMOTION_THRESHOLD:.0%}"
)
    latest_iteration = (
        get_latest_iteration(
            "models"
        )
    )

    if latest_iteration is None:

        raise RuntimeError(
            "No trained model found."
        )

    current_model = (
        f"models/model_iteration_"
        f"{latest_iteration}.pt"
    )

    next_iteration = (
        latest_iteration + 1
    )

    print()
    print(
        f"Latest iteration: "
        f"{latest_iteration}"
    )

    print(
        f"Next iteration: "
        f"{next_iteration}"
    )

    print()
    print(
        f"Current model: "
        f"{current_model}"
    )

    print()
    print("Checking current model...")

    check_model(
        current_model,
        board_size
    )

    print()
    print(
        "Current model is ready for "
        f"iteration {next_iteration}."
    )

    print()
    print("=" * 70)
    print("DAY 16 INITIALIZATION COMPLETE")
    print("=" * 70)
    print_pipeline_status(
    latest_iteration
)
if __name__ == "__main__":

    current_model = (
        "models/model_iteration_2.pt"
    )

    data_path = generate_self_play_data(
        current_model,
        3
    )

    print()
    print("=" * 70)
    print("SELF-PLAY TEST COMPLETE")
    print("=" * 70)

    print(
        f"Generated: {data_path}"
    )   