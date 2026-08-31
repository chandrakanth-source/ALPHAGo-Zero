class Arena:

    def __init__(
        self,
        model_a,
        model_b,
        game_factory,
        mcts_factory,
        num_games=10
    ):

        self.model_a = model_a
        self.model_b = model_b

        self.game_factory = game_factory
        self.mcts_factory = mcts_factory

        self.num_games = num_games

    def play_game(
        self,
        model_a,
        model_b
    ):
        """
        Play one complete game.

        The exact MCTS/GoGame calls should use
        the existing implementations in this project.
        """

        game = self.game_factory()

        player_models = {
            1: model_a,
            -1: model_b
        }

        while not game.is_terminal():

            current_player = game.current_player

            current_model = player_models[
                current_player
            ]

            mcts = self.mcts_factory(
                game,
                current_model
            )

            move = mcts.best_move()

            game.play_move(move)

        return game.get_result()

    def evaluate(self):

        wins = 0
        losses = 0
        draws = 0

        for game_number in range(
            self.num_games
        ):

            print(
                f"Evaluation game "
                f"{game_number + 1}/"
                f"{self.num_games}"
            )

            result = self.play_game(
                self.model_a,
                self.model_b
            )

            if result == 1:

                wins += 1

            elif result == -1:

                losses += 1

            else:

                draws += 1

        return (
            wins,
            losses,
            draws
        )