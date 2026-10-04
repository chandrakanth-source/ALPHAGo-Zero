import shutil
import os


class ModelPromotion:

    def __init__(
        self,
        models_dir="models"
    ):

        self.models_dir = models_dir

    # =====================================
    # SHOULD PROMOTE?
    # =====================================

    def should_promote(
        self,
        new_win_rate,
        threshold=0.55
    ):

        return (
            new_win_rate >= threshold
        )

    # =====================================
    # PROMOTE MODEL
    # =====================================

    def promote(
        self,
        new_model_path
    ):

        latest_path = os.path.join(
            self.models_dir,
            "latest_model.pt"
        )

        shutil.copy2(
            new_model_path,
            latest_path
        )

        print(
            f"New model promoted:"
            f" {new_model_path}"
        )

        print(
            f"Updated:"
            f" {latest_path}"
        )

        return latest_path