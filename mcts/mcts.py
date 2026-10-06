import math
import numpy as np

from mcts.node import MCTSNode
from mcts.network_Evaluator import (
    NetworkEvaluator
)


def puct_score(
    parent=None,
    child=None,
    c_puct=1.5,
    *,
    prior=None,
    value=None,
    parent_visits=None,
    child_visits=None
):

    if parent is not None and child is not None:
        exploitation = child.value()
        child_prior = child.prior
        visits = parent.visit_count
        child_visit_count = child.visit_count
    else:
        if None in (prior, value, parent_visits, child_visits):
            raise TypeError(
                "puct_score requires parent and child or "
                "prior, value, parent_visits, and child_visits"
            )

        exploitation = value
        child_prior = prior
        visits = parent_visits
        child_visit_count = child_visits

    exploration = (
        c_puct
        * child_prior
        * math.sqrt(visits)
        / (1 + child_visit_count)
    )

    return float(exploitation + exploration)


class MCTS:

    def __init__(
        self,
        model=None,
        game=None,
        board_size=None,
        simulations=100,
        c_puct=1.5
    ):

        # Support the legacy MCTS(game, evaluator) call shape.
        if hasattr(model, "board"):
            legacy_game = model
            legacy_evaluator = game
            game = legacy_game
            model = (
                legacy_evaluator.model
                if hasattr(legacy_evaluator, "model")
                else None
            )

        if game is not None and board_size is None:
            board_size = game.board_size

        if model is None:
            from network.network import GoNetwork

            model = GoNetwork(
                board_size=board_size or 12
            )

        self.model = model

        self.game = game

        self.board_size = board_size

        self.simulations = simulations

        self.c_puct = c_puct

        if (
            'legacy_evaluator' in locals()
            and legacy_evaluator is not None
        ):
            self.evaluator = legacy_evaluator
        else:
            self.evaluator = NetworkEvaluator(
                self.model
            )

        self.root = None

    # --------------------------------------------------
    # PUCT
    # --------------------------------------------------

    def puct_score(self, parent, child):
        return puct_score(
            parent,
            child,
            self.c_puct
        )

    # --------------------------------------------------
    # SELECT
    # --------------------------------------------------

    def select_child(self, node):

        best_child = None

        best_score = float("-inf")

        for child in node.children.values():

            score = self.puct_score(
                node,
                child
            )

            if score > best_score:

                best_score = score

                best_child = child

        return best_child

    # --------------------------------------------------
    # EXPAND
    # --------------------------------------------------

    def expand(
        self,
        node,
        policy
    ):

        state = node.state

        legal_moves = (
            state.get_legal_moves()
        )

        pass_action = (
            state.get_pass_action()
        )

        # Convert tuple moves to integer
        # actions.

        converted_moves = []

        for move in legal_moves:

            if move == "PASS":

                converted_moves.append(
                    pass_action
                )

            elif isinstance(
                move,
                tuple
            ):

                row, col = move

                action = (
                    row
                    * state.board_size
                    + col
                )

                converted_moves.append(
                    action
                )

            else:

                converted_moves.append(
                    move
                )

        if not converted_moves:

            return

        priors = np.asarray(
            [
                policy[action]
                for action
                in converted_moves
            ],
            dtype=np.float32
        )

        # Keep only positive probability
        # values.

        priors = np.maximum(
            priors,
            0.0
        )

        total = priors.sum()

        if total <= 0:

            priors = (
                np.ones_like(priors)
                / len(priors)
            )

        else:

            priors = (
                priors
                / total
            )

        for action, prior in zip(
            converted_moves,
            priors
        ):

            if action in node.children:

                continue

            node.add_child(
                move=action,
                state=None,
                prior=float(prior)
            )

    # --------------------------------------------------
    # APPLY MOVE TO STATE
    # --------------------------------------------------

    def make_child_state(
        self,
        node,
        child
    ):

        if child.state is not None:
            return

        state = node.state

        # Create a fresh GoGame state.

        from environment.go_game import (
            GoGame
        )

        new_state = GoGame(
            board_size=state.board_size
        )

        new_state.set_state(
            state.get_state()
        )

        new_state.play(
            child.move
        )

        child.state = new_state

    # --------------------------------------------------
    # BACKUP
    # --------------------------------------------------

    def backup(
        self,
        node,
        value
    ):

        current = node

        # *value* is from the perspective of the player to move at *node*.
        # Each node stores value from the perspective of the player who made
        # the move into it (its parent's mover), so select_child, which
        # maximises child.value(), picks moves good for the player choosing.
        value = -value

        while current is not None:

            current.visit_count += 1

            current.value_sum += value

            # Switch perspective because
            # players alternate.

            value = -value

            current = current.parent

    # --------------------------------------------------
    # SEARCH
    # --------------------------------------------------

    def search(
        self,
        state=None
    ):

        if state is None:
            state = self.game

        if state is None:
            raise ValueError(
                "search requires a game state"
            )

        self.root = MCTSNode(
            state=state
        )

        # Initial network evaluation.

        policy, value = (
            self.evaluator.evaluate(
                state
            )
        )

        self.expand(
            self.root,
            policy
        )

        if not self.root.children:

            return state.get_pass_action()

        # Run simulations.

        for _ in range(
            self.simulations
        ):

            node = self.root

            # ----------------------------
            # Selection
            # ----------------------------

            while (
                node.is_expanded()
                and node.children
            ):

                child = (
                    self.select_child(
                        node
                    )
                )

                self.make_child_state(
                    node,
                    child
                )

                node = child

                if node.state.is_terminal():

                    break

            # ----------------------------
            # Evaluation / Expansion
            # ----------------------------

            if node.state.is_terminal():

                value = (
                    self.terminal_value(
                        node.state
                    )
                )

            else:

                policy, value = (
                    self.evaluator.evaluate(
                        node.state
                    )
                )

                self.expand(
                    node,
                    policy
                )

            # ----------------------------
            # Backup
            # ----------------------------

            self.backup(
                node,
                value
            )

        # Choose highest visit count.

        best_child = max(
            self.root.children.values(),
            key=lambda child:
                child.visit_count
        )

        return best_child.move

    # --------------------------------------------------
    # TERMINAL VALUE
    # --------------------------------------------------

    def terminal_value(
        self,
        state
    ):

        winner = state.get_winner()

        current_player = (
            state.current_player
        )

        if winner == 0:

            return 0.0

        if winner == current_player:

            return 1.0

        return -1.0