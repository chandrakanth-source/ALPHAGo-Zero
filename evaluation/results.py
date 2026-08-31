class EvaluationResult:

    def __init__(
        self,
        wins,
        losses,
        draws
    ):

        self.wins = wins
        self.losses = losses
        self.draws = draws

        self.total_games = (
            wins + losses + draws
        )

    @property
    def win_rate(self):

        if self.total_games == 0:
            return 0.0

        return self.wins / self.total_games

    @property
    def score(self):

        if self.total_games == 0:
            return 0.0

        return (
            self.wins
            + 0.5 * self.draws
        ) / self.total_games

    def summary(self):

        return {
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "games": self.total_games,
            "win_rate": self.win_rate,
            "score": self.score
        }