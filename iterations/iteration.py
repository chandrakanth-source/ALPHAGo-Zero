"""
AlphaGo Zero single-iteration runner.

Ties together self-play, training, and (optionally) model saving into one
logical unit that the full pipeline calls in a loop.
"""

import os
import torch

from network.model_loader import load_model
from mcts.network_Evaluator import NetworkEvaluator
from self_play.self_play import SelfPlay
from training.train_iteration import train_iteration


class AlphaGoZeroIteration:
    """
    Encapsulates one AlphaGo Zero self-play → train iteration.

    Args:
        board_size:   Go board side length.
        simulations:  MCTS simulations per move.
    """

    def __init__(
        self,
        board_size=12,
        simulations=10,
    ):
        self.board_size = board_size
        self.simulations = simulations

    # ------------------------------------------------------------------
    # Sub-component constructors
    # ------------------------------------------------------------------

    def create_evaluator(self, model):
        """Wrap *model* in a :class:`NetworkEvaluator`."""
        return NetworkEvaluator(model)

    def load_current_model(self, model_path):
        """Load and return the model at *model_path*."""
        print(f"Loading model: {model_path}")
        return load_model(model_path, board_size=self.board_size)

    def create_self_play(self, evaluator):
        """Create a :class:`SelfPlay` connected to *evaluator*."""
        return SelfPlay(
            board_size=self.board_size,
            simulations=self.simulations,
            evaluator=evaluator,
        )

    # ------------------------------------------------------------------
    # Data generation
    # ------------------------------------------------------------------

    def generate_self_play_data(self, self_play, num_games=5):
        """
        Play *num_games* games and return all training examples.

        Args:
            self_play:  A configured :class:`SelfPlay` instance.
            num_games:  Number of games to generate.

        Returns:
            list: All ``(state, policy, value)`` training triples.
        """
        print()
        print("Generating self-play data...")

        all_examples = []

        for game_number in range(num_games):
            print(f"Game {game_number + 1}/{num_games}")

            examples = self_play.generate_game()
            all_examples.extend(examples)

            print(f"Examples collected: {len(all_examples)}")

        return all_examples

    # ------------------------------------------------------------------
    # Dataset persistence
    # ------------------------------------------------------------------

    def save_data(self, examples, iteration):
        """
        Save *examples* to ``data/self_play_iteration_{iteration}.pt``.

        Returns:
            str: Path of the saved file.
        """
        os.makedirs("data", exist_ok=True)

        path = f"data/self_play_iteration_{iteration}.pt"

        torch.save(examples, path)

        print()
        print(f"Dataset saved to: {path}")

        return path

    # ------------------------------------------------------------------
    # High-level entry points
    # ------------------------------------------------------------------

    def run_self_play_iteration(
        self,
        model_path,
        iteration,
        num_games=5,
    ):
        """
        Run self-play for one iteration and persist the dataset.

        Args:
            model_path: Path to the current model weights.
            iteration:  Iteration number (used for the output filename).
            num_games:  Number of self-play games.

        Returns:
            str: Path to the saved dataset file.
        """
        print()
        print("=" * 60)
        print(f"ALPHAGO ZERO ITERATION {iteration}")
        print("=" * 60)

        model = self.load_current_model(model_path)
        evaluator = self.create_evaluator(model)
        self_play = self.create_self_play(evaluator)

        examples = self.generate_self_play_data(
            self_play, num_games=num_games
        )

        return self.save_data(examples, iteration)

    def run_iteration(
        self,
        model_path,
        iteration,
        num_games=5,
        epochs=5,
        batch_size=32,
    ):
        """
        Run a complete self-play → train iteration.

        Args:
            model_path:  Path to the current best model.
            iteration:   Iteration number.
            num_games:   Self-play games.
            epochs:      Training epochs.
            batch_size:  Mini-batch size.

        Returns:
            str: Path of the newly trained model.
        """
        data_path = self.run_self_play_iteration(
            model_path=model_path,
            iteration=iteration,
            num_games=num_games,
        )

        return train_iteration(
            data_path=data_path,
            iteration=iteration,
            board_size=self.board_size,
            epochs=epochs,
            batch_size=batch_size,
            model_path=model_path,
        )