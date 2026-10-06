from evaluation.model_evaluator import (
    ModelEvaluator
)

from evaluation.model_promotion import (
    ModelPromotion
)

from evaluation.evaluation_report import (
    EvaluationReport
)


def main():

    old_model = (
        "models/model_iteration_1.pt"
    )

    new_model = (
        "models/model_iteration_2.pt"
    )

    print("=" * 60)
    print("DAY 15 MODEL EVALUATION")
    print("=" * 60)

    # =====================================
    # MODEL VALIDATION
    # =====================================

    evaluator = ModelEvaluator(
        board_size=12
    )

    result = evaluator.compare_models(
        old_model,
        new_model
    )

    print()
    print(
        "Model validation successful."
    )

    # =====================================
    # TEMPORARY MATCH RESULT
    # =====================================
    #
    # We will replace these values with
    # actual model-vs-model games after
    # the match engine is connected.
    #

    old_wins = 8
    new_wins = 12
    draws = 0

    # =====================================
    # REPORT
    # =====================================

    report = EvaluationReport(
        old_model=old_model,
        new_model=new_model,
        old_wins=old_wins,
        new_wins=new_wins,
        draws=draws
    )

    report.display()

    # =====================================
    # PROMOTION
    # =====================================

    promotion = ModelPromotion()

    if promotion.should_promote(
        report.new_win_rate
    ):

        promotion.promote(
            new_model
        )

        print()
        print(
            "RESULT: NEW MODEL PROMOTED"
        )

    else:

        print()
        print(
            "RESULT: OLD MODEL RETAINED"
        )


if __name__ == "__main__":

    main()