import os
import shutil


class PromotionManager:

    def __init__(
        self,
        models_dir="models"
    ):

        self.models_dir = models_dir

        os.makedirs(
            self.models_dir,
            exist_ok=True
        )

    def promote(
        self,
        model_path
    ):

        latest_path = os.path.join(
            self.models_dir,
            "latest_model.pt"
        )

        shutil.copy2(
            model_path,
            latest_path
        )

        print()
        print("=" * 60)
        print("MODEL PROMOTED")
        print("=" * 60)

        print(
            f"New latest model: "
            f"{model_path}"
        )

        print(
            f"Copied to: "
            f"{latest_path}"
        )

        return latest_path

    def reject(
        self,
        model_path
    ):

        print()
        print("=" * 60)
        print("MODEL REJECTED")
        print("=" * 60)

        print(
            f"Candidate remains: "
            f"{model_path}"
        )

        print(
            "Previous latest model "
            "will remain active."
        )

        return False