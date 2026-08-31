from network.model_loader import load_model
from evaluation.evaluator import ModelEvaluator


def main():

    board_size = 12

    old_model = load_model(
        "models/previous_model.pt",
        board_size
    )

    new_model = load_model(
        "models/latest_model.pt",
        board_size
    )

    evaluator = ModelEvaluator(
        num_games=10
    )

    print("=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)

    print()
    print("Old model:")
    print("models/previous_model.pt")

    print()
    print("New model:")
    print("models/latest_model.pt")

    # Arena integration will provide
    # actual game results.

    print()
    print("Evaluation system ready.")


if __name__ == "__main__":

    main()