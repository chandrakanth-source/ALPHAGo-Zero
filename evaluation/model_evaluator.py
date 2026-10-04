import torch

from network.model_loader import load_model


class ModelEvaluator:

    def __init__(
        self,
        board_size=12,
        device="cpu"
    ):

        self.board_size = board_size

        self.device = torch.device(
            device
        )

    # =====================================
    # LOAD MODEL
    # =====================================

    def load(
        self,
        model_path
    ):

        model = load_model(
            model_path,
            board_size=self.board_size
        )

        model.to(
            self.device
        )

        model.eval()

        return model

    # =====================================
    # GET MODEL PARAMETERS
    # =====================================

    def get_parameter_count(
        self,
        model
    ):

        return sum(
            p.numel()
            for p in model.parameters()
        )

    # =====================================
    # COMPARE MODEL STRUCTURE
    # =====================================

    def compare_models(
        self,
        old_model_path,
        new_model_path
    ):

        old_model = self.load(
            old_model_path
        )

        new_model = self.load(
            new_model_path
        )

        old_parameters = (
            self.get_parameter_count(
                old_model
            )
        )

        new_parameters = (
            self.get_parameter_count(
                new_model
            )
        )

        print(
            f"Old model parameters: "
            f"{old_parameters}"
        )

        print(
            f"New model parameters: "
            f"{new_parameters}"
        )

        return {
            "old_parameters":
                old_parameters,

            "new_parameters":
                new_parameters
        }