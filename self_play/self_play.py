"""
Self-play module for AlphaGo Zero.

Plays games against itself using MCTS guided by the trained neural network,
collecting (state, policy, value) training examples.
"""

import numpy as np
import torch

from environment.go_game import GoGame
from mcts.mcts import MCTS
from mcts.network_Evaluator import NetworkEvaluator


class SelfPlay:
    """
    Runs self-play games using MCTS + neural network.

    Args:
        board_size:   Side length of the Go board.
        simulations:  Number of MCTS simulations per move.
        evaluator:    Pre-built NetworkEvaluator (optional). If *None* and
                      *model* is also *None* a fresh, untrained GoNetwork is
                      used automatically.
        model:        Raw GoNetwork (optional).  Wrapped in a NetworkEvaluator
                      when supplied.
        temperature:  Move-selection temperature (see `select_move`).
        temp_threshold: After this many moves per game the temperature is set
                        to 0 (greedy / best-move selection).  Set to 0 to
                        always use `temperature`.  Default is 30.
    """

    def __init__(
        self,
        board_size=12,
        simulations=100,
        evaluator=None,
        model=None,
        temperature=1.0,
        temp_threshold=30,
        dirichlet_alpha=None,
    ):
        self.board_size = board_size
        # AlphaGo Zero uses alpha ~ 10 / (typical legal moves); 0.03 on 19x19.
        self.dirichlet_alpha = dirichlet_alpha
        self.simulations = simulations
        self.temperature = temperature
        self.temp_threshold = temp_threshold

        # Resolve evaluator: accept a pre-built evaluator, a raw model, or
        # fall back to a random (untrained) GoNetwork.
        if evaluator is not None:
            self.evaluator = evaluator
            # Expose the underlying model for callers that inspect it.
            self.model = getattr(evaluator, "model", None)
        elif model is not None:
            self.evaluator = NetworkEvaluator(model)
            self.model = model
        else:
            # Lazy import to avoid circular dependency at module level.
            from network.network import GoNetwork

            _model = GoNetwork(board_size=board_size)
            self.evaluator = NetworkEvaluator(_model)
            self.model = _model

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save_examples(self, examples, path):
        """Serialize training examples to disk with torch.save."""
        torch.save(examples, path)

    def load_examples(self, path):
        """Load training examples that were saved with :meth:`save_examples`."""
        return torch.load(path, weights_only=False)

    # ------------------------------------------------------------------
    # Game helpers
    # ------------------------------------------------------------------

    def create_game(self):
        """Return a fresh GoGame instance."""
        return GoGame(self.board_size)

    def get_state(self, game):
        """
        Encode *game* into the 3-plane float32 tensor expected by GoNetwork.

        Planes:
          0 – current player's stones
          1 – opponent's stones
          2 – empty intersections
        """
        board = game.board
        player = game.current_player
        return np.stack(
            (
                board == player,
                board == -player,
                board == 0,
            )
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # MCTS policy extraction
    # ------------------------------------------------------------------

    def get_mcts_policy(self, mcts):
        """
        Convert MCTS visit counts into a policy vector.

        Returns a normalised probability distribution over
        ``board_size * board_size + 1`` actions (last entry = PASS).
        """
        root = mcts.root

        policy = np.zeros(
            self.board_size * self.board_size + 1,
            dtype=np.float32,
        )

        total_visits = 0

        for move, child in root.children.items():
            visits = child.visit_count
            policy[move] = visits
            total_visits += visits

        if total_visits > 0:
            policy /= total_visits

        return policy

    # ------------------------------------------------------------------
    # Temperature-based move selection  (Task 24)
    # ------------------------------------------------------------------

    def select_move(self, policy, move_number=0):
        """
        Sample a move from *policy* using temperature-based selection.

        Temperature schedule
        --------------------
        * For ``move_number < temp_threshold`` the configured *temperature*
          (typically 1.0) is used so that the agent explores diverse moves.
        * For ``move_number >= temp_threshold`` temperature is set to 0,
          giving greedy (argmax) selection.

        Args:
            policy:      1-D numpy array of visit-count probabilities.
            move_number: Zero-indexed move counter within the current game.
                         Used to switch from exploratory to greedy selection.

        Returns:
            int: Selected action index.
        """
        # Apply temperature threshold scheduling.
        if self.temp_threshold > 0 and move_number >= self.temp_threshold:
            effective_temperature = 0.0
        else:
            effective_temperature = self.temperature

        # Greedy (temperature == 0) → deterministic argmax.
        if effective_temperature <= 0:
            return int(np.argmax(policy))

        # Raise probabilities to the power 1/T, then renormalise.
        powered = np.power(
            policy.astype(np.float64),
            1.0 / effective_temperature,
        )

        total = powered.sum()

        if total <= 0:
            # Fall back to argmax if all probabilities are zero.
            return int(np.argmax(policy))

        probabilities = powered / total

        return int(
            np.random.choice(
                len(probabilities),
                p=probabilities,
            )
        )

    # ------------------------------------------------------------------
    # Game result
    # ------------------------------------------------------------------

    def get_game_result(self, game):
        """
        Return the winner as a float: +1 (Black), -1 (White), 0 (draw).
        """
        winner = game.get_winner()

        if winner == 0:
            return 0.0

        return float(winner)

    # ------------------------------------------------------------------
    # Training-data assembly
    # ------------------------------------------------------------------

    def create_training_data(self, history, result):
        """
        Convert a game history into labelled training examples.

        Args:
            history: List of (state, policy, player) tuples.
            result:  Game result from :meth:`get_game_result`.

        Returns:
            List of (state, policy, value) tuples ready for the dataset.
        """
        training_data = []

        for state, policy, player in history:
            # The value is from *this player's* perspective: if the player
            # won, value = +1; if the opponent won, value = -1.
            value = result * player
            training_data.append((state, policy, value))

        return training_data

    # ------------------------------------------------------------------
    # Core game loop  (Task 3 – trained model connected to MCTS self-play)
    # ------------------------------------------------------------------

    def play_game(self, max_moves=500):
        """
        Play one complete self-play game.

        Each move: run MCTS from the current position using the trained
        evaluator, extract the visit-count policy, select a move with
        temperature, and record the transition.

        Returns:
            tuple: (history, result) where *history* is a list of
                   (state, policy, player) tuples and *result* is the
                   game outcome (+1 / -1 / 0).
        """
        game = self.create_game()
        history = []
        move_count = 0

        while not game.is_terminal() and move_count < max_moves:

            # Build an MCTS tree rooted at the current position and backed
            # by the trained network evaluator.
            mcts = MCTS(
                model=self.model,
                game=game,
                board_size=self.board_size,
                simulations=self.simulations,
                dirichlet_alpha=self.dirichlet_alpha,
            )

            # Override the default NetworkEvaluator with our shared one so
            # that the same (possibly already-loaded) model is used.
            mcts.evaluator = self.evaluator

            mcts.search(game)

            # Derive MCTS policy from root visit counts.
            policy = self.get_mcts_policy(mcts)

            # Temperature-scheduled move selection.
            move = self.select_move(policy, move_number=move_count)

            # Record the pre-move state for training.
            state = self.get_state(game)
            player = game.current_player

            history.append((state, policy, player))

            # Apply move to the game.
            if move == game.get_pass_action():
                game.pass_move()
            else:
                success = game.make_move(move)
                if not success:
                    # Illegal move returned by MCTS – fall back to pass.
                    game.pass_move()

            move_count += 1

        result = self.get_game_result(game)
        return history, result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_game(self):
        """Play one game and return training examples."""
        history, result = self.play_game()
        return self.create_training_data(history, result)

    def generate_games(self, num_games=5):
        """Play *num_games* games and return all training examples."""
        all_data = []

        for game_number in range(num_games):
            print(
                f"Playing game {game_number + 1}/{num_games}"
            )

            game_data = self.generate_game()
            all_data.extend(game_data)

        return all_data

    def generate_dataset(self, num_games=5):
        """Alias for :meth:`generate_games`."""
        return self.generate_games(num_games)
