from evaluation.model_promotion import (
    ModelPromotion
)


def test_model_should_promote():

    promotion = ModelPromotion()

    assert (
        promotion.should_promote(
            0.60
        )
        is True
    )


def test_model_should_not_promote():

    promotion = ModelPromotion()

    assert (
        promotion.should_promote(
            0.40
        )
        is False
    )


def test_exact_threshold():

    promotion = ModelPromotion()

    assert (
        promotion.should_promote(
            0.55
        )
        is True
    )