from mcts.mcts import MCTS


class ModelPlayer:

    def __init__(
        self,
        model,
        board_size=12,
        simulations=10
    ):

        self.model = model
        self.board_size = board_size
        self.simulations = simulations

    def select_move(self, game):

        searcher = MCTS(
            model=self.model,
            game=game,
            board_size=self.board_size,
            simulations=self.simulations
        )

        move = searcher.search(game)

        return move


def play_game(
    model_black,
    model_white,
    game,
    max_moves=200
):

    players = {
        1: model_black,
        -1: model_white
    }

    move_number = 0

    while not game.is_terminal():

        current_player = game.current_player

        player = players[current_player]

        move = player.select_move(game)

        if move == game.get_pass_action():

            print(
                f"Move {move_number + 1}: "
                f"Player {current_player} -> PASS"
            )

        elif isinstance(move, int):

            row = move // game.board_size
            col = move % game.board_size

            print(
                f"Move {move_number + 1}: "
                f"Player {current_player} -> "
                f"({row}, {col})"
            )

        if not game.is_legal(move):

            print(
                f"Illegal move {move}. "
                f"Forcing PASS."
            )

            move = game.get_pass_action()

        success = game.play(move)

        if not success:

            print(
                f"Move {move} failed. "
                f"Forcing PASS."
            )

            game.play(
                game.get_pass_action()
            )

        move_number += 1

        if move_number >= max_moves:

            print(
                "Maximum move limit reached."
            )

            break

    return game.get_winner()