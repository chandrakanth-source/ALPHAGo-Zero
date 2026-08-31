from evaluation.results import EvaluationResult
import torch

from network.network import GoNetwork


def load_model(
    model_path="models/latest_model.pt",
    board_size=12
):

    model = GoNetwork(
        board_size=board_size
    )

    checkpoint = torch.load(
        model_path,
        map_location="cpu"
    )

    # If you saved only model.state_dict()
    model.load_state_dict(
        checkpoint
    )

    model.eval()

    return model

class ModelEvaluator:

    def __init__(
        self,
        num_games=10
    ):

        self.num_games = num_games

    def evaluate_results(
        self,
        wins,
        losses,
        draws
    ):

        return EvaluationResult(
            wins=wins,
            losses=losses,
            draws=draws
        )

    def is_better(
        self,
        result,
        threshold=0.55
    ):

        return result.score >= threshold