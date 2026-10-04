from environment.go_game import GoGame

from network.model_loader import (
    load_trained_network
)

from mcts.mcts import MCTS


def main():

    print("=" * 70)
    print("PUCT MCTS TEST")
    print("=" * 70)

    model = load_trained_network(
        board_size=12,
        path="models/model_iteration_3.pt"
    )

    game = GoGame(
        board_size=12
    )

    search = MCTS(
        model=model,
        game=game,
        board_size=12,
        simulations=10,
        c_puct=1.5
    )

    move = search.search(
        game
    )

    print()

    print(
        f"Selected move: {move}"
    )

    print(
        f"Root visits: "
        f"{search.root.visit_count}"
    )

    print(
        f"Root children: "
        f"{len(search.root.children)}"
    )

    print()

    print(
        "PUCT MCTS test completed."
    )

    print("=" * 70)


if __name__ == "__main__":

    main()