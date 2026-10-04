from evaluation.evaluation_report import (
    EvaluationReport
)


def test_evaluation_report():

    report = EvaluationReport(
        old_model="model_iteration_1.pt",
        new_model="model_iteration_2.pt",
        old_wins=8,
        new_wins=12,
        draws=0
    )

    assert report.total_games == 20

    assert (
        report.new_win_rate == 0.60
    )