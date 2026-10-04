from evaluation.evaluation_result import (
    EvaluationResult
)

from evaluation.promotion import (
    PromotionManager
)


PROMOTION_THRESHOLD = 0.55


def decide_promotion(
    candidate_path,
    result
):

    print()
    print("=" * 60)
    print("PROMOTION DECISION")
    print("=" * 60)

    print(
        f"Required win rate: "
        f"{PROMOTION_THRESHOLD:.0%}"
    )

    print(
        f"Candidate win rate: "
        f"{result.candidate_win_rate:.2%}"
    )

    if (
        result.candidate_win_rate
        >= PROMOTION_THRESHOLD
    ):

        print()
        print(
            "Candidate is STRONGER."
        )

        manager = PromotionManager()

        manager.promote(
            candidate_path
        )

        return True

    else:

        print()
        print(
            "Candidate did NOT reach "
            "the promotion threshold."
        )

        manager = PromotionManager()

        manager.reject(
            candidate_path
        )

        return False


def main():

    candidate_path = (
        "models/model_iteration_3.pt"
    )

    # Temporary test result.
    # We will replace this with
    # real Model 2 vs Model 3 games.

    result = EvaluationResult(
        candidate_wins=12,
        current_wins=7,
        draws=1
    )

    result.print_summary()

    decide_promotion(
        candidate_path,
        result
    )


if __name__ == "__main__":

    main()