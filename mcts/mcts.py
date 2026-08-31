import numpy as np
import math
from mcts.network_Evaluator import NetworkEvaluator
from network.network import GoNetwork
from mcts.node import MCTSNode

class MCTS:

    def __init__(self, model=None,game=None, board_size=None, simulations=100):

        initial_state = None
        evaluator = None
        if board_size is None and hasattr(model, "board"):
            board = model
            board_size = board.board_size
            initial_state = board
            if isinstance(game, NetworkEvaluator):
                evaluator = game
                model = evaluator.model
            else:
                model = GoNetwork(board_size)
        self.game=game
        self.model = model
        self.simulations = simulations
        self.board_size = board_size
        self.root = MCTSNode(state=initial_state)

        self.evaluator = evaluator or NetworkEvaluator(self.model)
    def evaluate_position(self, state):

         policy, value = self.evaluator.evaluate(
        state
        )

         return policy, value
    def search(self, state=None):
        if state is not None:
            self.root = MCTSNode(state=state)
        elif self.root.state is None:
            raise ValueError("search requires a game state")
        else:
            state = self.root.state

        policy, value = self.evaluator.evaluate(state)

        legal_moves = [
            i for i in range(len(policy))
            if state.is_legal(i)
        ]

        if not legal_moves:
            return state.get_pass_action()

        board_moves = [
            move for move in legal_moves
            if move != state.get_pass_action()
        ]
        if board_moves:
            legal_moves = board_moves

        for move in legal_moves:
            self.root.add_child(
                move=move,
                state=state,
                prior=float(policy[move])
            ).visit_count = 1

        best_move = legal_moves[np.argmax(policy[legal_moves])]
        return best_move


def puct_score(
    prior,
    value,
    parent_visits,
    child_visits,
    c_puct=1.5
):

    exploration = (
        c_puct
        * prior
        * math.sqrt(parent_visits)
        / (1 + child_visits)
    )

    return value + exploration

def add_dirichlet_noise(policy, alpha=0.3, epsilon=0.25):

    noise = np.random.dirichlet(
        [alpha] * len(policy)
    )

    policy = (
        (1 - epsilon) * policy
        +
        epsilon * noise
    )

    return policy