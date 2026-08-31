from evaluation.results import EvaluationResult


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