from network.model_loader import load_model
from mcts.mcts import MCTS


MODEL_PATH = (
    "models/model_iteration_1.pt"
)


def main():

    print("=" * 60)
    print("MODEL 1 → MCTS")
    print("=" * 60)

    model = load_model(
        MODEL_PATH,
        board_size=12
    )

    print(
        "Model 1 loaded."
    )

    mcts = MCTS(
        model=model,
        board_size=12,
        simulations=10
    )

    print(
        "MCTS created."
    )

    print(
        "Model 1 → MCTS connection OK."
    )

    print("=" * 60)


if __name__ == "__main__":

    main()