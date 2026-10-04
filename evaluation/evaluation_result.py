class EvaluationResult:

    def __init__(
        self,
        candidate_wins,
        current_wins,
        draws=0
    ):

        self.candidate_wins = (
            candidate_wins
        )

        self.current_wins = (
            current_wins
        )

        self.draws = draws

        self.total_games = (
            candidate_wins
            + current_wins
            + draws
        )

    @property
    def candidate_win_rate(self):

        if self.total_games == 0:

            return 0.0

        return (
            self.candidate_wins
            / self.total_games
        )

    @property
    def current_win_rate(self):

        if self.total_games == 0:

            return 0.0

        return (
            self.current_wins
            / self.total_games
        )

    def print_summary(self):

        print()
        print("=" * 60)
        print("EVALUATION RESULT")
        print("=" * 60)

        print(
            f"Candidate wins: "
            f"{self.candidate_wins}"
        )

        print(
            f"Current wins: "
            f"{self.current_wins}"
        )

        print(
            f"Draws: "
            f"{self.draws}"
        )

        print(
            f"Total games: "
            f"{self.total_games}"
        )

        print(
            f"Candidate win rate: "
            f"{self.candidate_win_rate:.2%}"
        )

        print(
            f"Current win rate: "
            f"{self.current_win_rate:.2%}"
        )

        print("=" * 60)