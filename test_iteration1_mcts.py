from network.model_loader import load_model
from mcts.network_Evaluator import NetworkEvaluator


MODEL_PATH = "models/model_iteration_1.pt"


def main():

    print("=" * 60)
    print("ITERATION 1 MODEL → MCTS TEST")
    print("=" * 60)

    model = load_model(
        MODEL_PATH,
        board_size=12
    )

    print("Model loaded.")

    evaluator = NetworkEvaluator(
        model
    )

    print("Network evaluator created.")

    print(
        "Iteration 1 model is ready for MCTS."
    )

    print("=" * 60)


if __name__ == "__main__":
    main()