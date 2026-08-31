import os
import torch

from network.model_loader import load_model
from mcts.network_Evaluator import NetworkEvaluator
from self_play.self_play import SelfPlay
from training.train_iteration import train_iteration


class AlphaGoZeroIteration:

    def __init__(
        self,
        board_size=12,
        simulations=10
    ):

        self.board_size = board_size
        self.simulations = simulations
    def create_evaluator(
        self,
        model
    ):

        evaluator = NetworkEvaluator(
            model
        )

        return evaluator
    def load_current_model(
        self,
        model_path
    ):

        print(
            f"Loading model: {model_path}"
        )

        model = load_model(
            model_path,
            board_size=self.board_size
        )

        return model
    def create_self_play(
        self,
        evaluator
    ):

        self_play = SelfPlay(
            board_size=self.board_size,
            simulations=self.simulations,
            evaluator=evaluator
        )

        return self_play
    def create_self_play(
        self,
        evaluator
    ):

        self_play = SelfPlay(
            board_size=self.board_size,
            simulations=self.simulations,
            evaluator=evaluator
        )

        return self_play
    def generate_self_play_data(
        self,
        self_play,
        num_games=5
    ):

        print()
        print(
            "Generating self-play data..."
        )

        all_examples = []

        for game_number in range(
            num_games
        ):

            print(
                f"Game "
                f"{game_number + 1}/"
                f"{num_games}"
            )

            examples = (
                self_play.generate_game()
            )

            all_examples.extend(
                examples
            )

            print(
                f"Examples collected: "
                f"{len(all_examples)}"
            )

        return all_examples
    def save_data(
        self,
        examples,
        iteration
    ):

        os.makedirs(
            "data",
            exist_ok=True
        )

        path = (
            f"data/"
            f"self_play_iteration_"
            f"{iteration}.pt"
        )

        torch.save(
            examples,
            path
        )

        print()
        print(
            f"Dataset saved to: {path}"
        )

        return path
    def run_self_play_iteration(
        self,
        model_path,
        iteration,
        num_games=5
    ):

        print()
        print("=" * 60)

        print(
            f"ALPHAGO ZERO "
            f"ITERATION {iteration}"
        )

        print("=" * 60)

        # -----------------------------
        # Load current model
        # -----------------------------

        model = self.load_current_model(
            model_path
        )

        # -----------------------------
        # Create evaluator
        # -----------------------------

        evaluator = self.create_evaluator(
            model
        )

        # -----------------------------
        # Create self-play
        # -----------------------------

        self_play = self.create_self_play(
            evaluator
        )

        # -----------------------------
        # Generate data
        # -----------------------------

        examples = (
            self.generate_self_play_data(
                self_play,
                num_games=num_games
            )
        )

        # -----------------------------
        # Save data
        # -----------------------------

        data_path = self.save_data(
            examples,
            iteration
        )

        return data_path
    def run_self_play_iteration(
        self,
        model_path,
        iteration,
        num_games=5
    ):

        print()
        print("=" * 60)

        print(
            f"ALPHAGO ZERO "
            f"ITERATION {iteration}"
        )

        print("=" * 60)

        # -----------------------------
        # Load current model
        # -----------------------------

        model = self.load_current_model(
            model_path
        )

        # -----------------------------
        # Create evaluator
        # -----------------------------

        evaluator = self.create_evaluator(
            model
        )

        # -----------------------------
        # Create self-play
        # -----------------------------

        self_play = self.create_self_play(
            evaluator
        )

        # -----------------------------
        # Generate data
        # -----------------------------

        examples = (
            self.generate_self_play_data(
                self_play,
                num_games=num_games
            )
        )

        # -----------------------------
        # Save data
        # -----------------------------

        data_path = self.save_data(
            examples,
            iteration
        )

        return data_path

    def run_iteration(
        self,
        model_path,
        iteration,
        num_games=5,
        epochs=5,
        batch_size=32
    ):
        data_path = self.run_self_play_iteration(
            model_path=model_path,
            iteration=iteration,
            num_games=num_games
        )

        return train_iteration(
            data_path=data_path,
            iteration=iteration,
            board_size=self.board_size,
            epochs=epochs,
            batch_size=batch_size,
            model_path=model_path
        )