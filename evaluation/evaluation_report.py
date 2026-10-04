class EvaluationReport:

    def __init__(
        self,
        old_model,
        new_model,
        old_wins,
        new_wins,
        draws
    ):

        self.old_model = old_model
        self.new_model = new_model

        self.old_wins = old_wins
        self.new_wins = new_wins
        self.draws = draws

    @property
    def total_games(self):

        return (
            self.old_wins
            + self.new_wins
            + self.draws
        )

    @property
    def new_win_rate(self):

        if self.total_games == 0:
            return 0.0

        return (
            self.new_wins
            / self.total_games
        )

    def display(self):

        print()
        print("=" * 60)
        print("MODEL EVALUATION REPORT")
        print("=" * 60)

        print(
            f"Old model: {self.old_model}"
        )

        print(
            f"New model: {self.new_model}"
        )

        print(
            f"Old wins: {self.old_wins}"
        )

        print(
            f"New wins: {self.new_wins}"
        )

        print(
            f"Draws: {self.draws}"
        )

        print(
            f"Total games: {self.total_games}"
        )

        print(
            f"New model win rate: "
            f"{self.new_win_rate:.2%}"
        )

        print("=" * 60)