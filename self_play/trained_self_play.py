from network.model_loader import load_model
from mcts.network_Evaluator import NetworkEvaluator
from self_play.self_play import SelfPlay
def main():

    print("=" * 60)
    print("TRAINED NETWORK SELF-PLAY")
    print("=" * 60)

    # -------------------------------------
    # Load trained model
    # -------------------------------------

    model = load_model(
        "models/latest_model.pt"
    )

    print(
        "Trained model loaded."
    )

    # -------------------------------------
    # Create evaluator
    # -------------------------------------

    evaluator = NetworkEvaluator(
        model
    )

    print(
        "Network evaluator created."
    )

    # -------------------------------------
    # Create self-play
    # -------------------------------------

    self_play = SelfPlay(
        board_size=12,
        simulations=10,
        evaluator=evaluator
    )

    print(
        "Trained network connected to self-play."
    )

    # -------------------------------------
    # Generate game
    # -------------------------------------

    examples = self_play.generate_game()

    print()
    print(
        "Self-play completed."
    )

    print(
        "Training examples:",
        len(examples)
    )


def create_trained_self_play():

    model = load_model(
        "models/latest_model.pt"
    )

    evaluator = NetworkEvaluator(
        model
    )

    self_play = SelfPlay(
        board_size=12,
        simulations=10
    )

    # Attach the trained evaluator
    self_play.evaluator = evaluator

    return self_play


if __name__ == "__main__":

    self_play = create_trained_self_play()

    print(
        "Trained network connected to self-play."
    )