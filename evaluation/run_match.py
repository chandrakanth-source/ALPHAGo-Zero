from environment.go_game import GoGame

from network.model_loader import (
    load_trained_network
)

from evaluation.model_match import (
    ModelPlayer,
    play_game
)


BOARD_SIZE = 12

SIMULATIONS = 10

NUM_GAMES = 4


MODEL_2_PATH = (
    "models/model_iteration_2.pt"
)

MODEL_3_PATH = (
    "models/model_iteration_3.pt"
)


def main():

    print("=" * 70)
    print("ALPHAGO ZERO - MODEL 2 VS MODEL 3")
    print("=" * 70)

    print()

    print(
        "Loading Model 2..."
    )

    model_2 = load_trained_network(
        board_size=BOARD_SIZE,
        path=MODEL_2_PATH
    )

    print(
        "Model 2 loaded successfully."
    )

    print()

    print(
        "Loading Model 3..."
    )

    model_3 = load_trained_network(
        board_size=BOARD_SIZE,
        path=MODEL_3_PATH
    )

    print(
        "Model 3 loaded successfully."
    )

    player_2 = ModelPlayer(
        model=model_2,
        board_size=BOARD_SIZE,
        simulations=SIMULATIONS
    )

    player_3 = ModelPlayer(
        model=model_3,
        board_size=BOARD_SIZE,
        simulations=SIMULATIONS
    )

    model_2_wins = 0
    model_3_wins = 0
    draws = 0

    print()
    print("=" * 70)
    print("STARTING MATCHES")
    print("=" * 70)

    for game_number in range(
        1,
        NUM_GAMES + 1
    ):

        print()
        print(
            f"Game {game_number}/{NUM_GAMES}"
        )

        game = GoGame(
            board_size=BOARD_SIZE
        )

        # Alternate who plays Black
        if game_number % 2 == 1:

            black_player = player_2
            white_player = player_3

            model_2_color = "Black"

        else:

            black_player = player_3
            white_player = player_2

            model_2_color = "White"

        print(
            f"Model 2: {model_2_color}"
        )

        print(
            f"Model 3: "
            f"{'White' if model_2_color == 'Black' else 'Black'}"
        )

        winner = play_game(
            model_black=black_player,
            model_white=white_player,
            game=game
        )

        result = game.get_result()

        print(
            f"Black score: "
            f"{result['black_score']}"
        )

        print(
            f"White score: "
            f"{result['white_score']}"
        )

        if winner == 1:

            winner_name = (
                "Model 2"
                if model_2_color == "Black"
                else "Model 3"
            )

        elif winner == -1:

            winner_name = (
                "Model 3"
                if model_2_color == "Black"
                else "Model 2"
            )

        else:

            winner_name = "Draw"

        print(
            f"Winner: {winner_name}"
        )

        if winner_name == "Model 2":

            model_2_wins += 1

        elif winner_name == "Model 3":

            model_3_wins += 1

        else:

            draws += 1

    print()
    print("=" * 70)
    print("FINAL MATCH RESULT")
    print("=" * 70)

    print(
        f"Model 2 wins : {model_2_wins}"
    )

    print(
        f"Model 3 wins : {model_3_wins}"
    )

    print(
        f"Draws        : {draws}"
    )

    print(
        f"Total games  : {NUM_GAMES}"
    )

    print()

    if NUM_GAMES > 0:

        model_3_win_rate = (
            model_3_wins / NUM_GAMES
        )

        print(
            f"Model 3 win rate: "
            f"{model_3_win_rate:.2%}"
        )

    print("=" * 70)


if __name__ == "__main__":

    main()