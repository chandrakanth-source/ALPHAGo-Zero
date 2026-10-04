from network.model_loader import load_model
from mcts.mcts import MCTS


MODEL_PATH = (
    "models/model_iteration_2.pt"
)


def main():

    print("=" * 60)
    print("MODEL 2 → MCTS")
    print("=" * 60)

    model = load_model(
        MODEL_PATH,
        board_size=12
    )

    print(
        "Model 2 loaded."
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
        "Model 2 → MCTS connection OK."
    )

    print("=" * 60)


if __name__ == "__main__":

    main()