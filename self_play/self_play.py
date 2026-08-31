import numpy as np
import torch
from environment.go_game import GoGame
from mcts.mcts import MCTS
class SelfPlay:

    def __init__(
        self,
        board_size=12,
        simulations=100,
        evaluator=None
    ):
        self.evaluator=evaluator
        self.board_size = board_size
        self.simulations = simulations
    def save_examples(self, examples, path):
        torch.save(examples, path)
    def load_examples(self, path):
        return torch.load(
            path,
            weights_only=False
        )
    def generate_dataset(
        self,
        num_games=5
    ):

        all_examples = []

        for game_number in range(num_games):

            print(
                f"Generating game "
                f"{game_number + 1}/{num_games}"
            )

            examples = self.generate_game()

            all_examples.extend(
                examples
            )

            print(
                f"Examples collected: "
                f"{len(all_examples)}"
            )

        return all_examples
    def create_game(self):
        return GoGame(self.board_size)
    def generate_games(self, num_games=5):

        all_data = []

        for game_number in range(num_games):

            print(
                f"Playing game "
                f"{game_number + 1}/{num_games}"
            )

            game_data = self.generate_game()

            all_data.extend(game_data)

        return all_data
    def generate_game(self):

        history = self.play_game()

        game = self.create_game()

        # The game result must come from the
        # actual game played.
        #
        # We will use the final result returned
        # by the environment.

        result = self.get_game_result(game)

        training_data = self.create_training_data(
            history,
            result
        )

        return training_data
    def get_mcts_policy(self, mcts):
        
        root = mcts.root

        policy = np.zeros(
            self.board_size * self.board_size + 1,
            dtype=np.float32
        )

        total_visits = 0

        for move, child in root.children.items():

            visits = child.visit_count

            policy[move] = visits

            total_visits += visits

        if total_visits > 0:
            policy /= total_visits

        return policy
    def get_state(self, game):
        board = game.board
        player = game.current_player

        return np.stack(
            (
                board == player,
                board == -player,
                np.full(board.shape, player == 1),
            )
        ).astype(np.float32)
    def play_game(self,max_moves=500):

        game = self.create_game()

        history = []

        move_count=0
        while not game.is_terminal() and move_count < max_moves:

            mcts = MCTS(
                game,
                self.evaluator,
                simulations=self.simulations
            )

            move = mcts.search()

            policy = self.get_mcts_policy(mcts)

            state = self.get_state(game)

            player = game.current_player

            history.append(
                (
                    state,
                    policy,
                    player
                )
            )

            if move == game.get_pass_action():

                game.pass_move()

            else:

                game.make_move(move)
            move_count += 1
        result = self.get_game_result(game)

        return history, result
    def generate_game(self):

        history, result = self.play_game()

        training_data = self.create_training_data(
            history,
            result
        )

        return training_data
    def get_game_result(self, game):

        winner = game.get_winner()

        if winner == 0:
            return 0.0

        return float(winner)
    def create_training_data(
        self,
        history,
        result
    ):

        training_data = []

        for state, policy, player in history:

            value = result * player

            training_data.append(
                (
                    state,
                    policy,
                    value
                )
            )

        return training_data