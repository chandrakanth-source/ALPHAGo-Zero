import numpy as np
import torch

from network.network import GoNetwork


def load_trained_network(
    board_size=12,
    path="models/latest_model.pt"
):

    model = GoNetwork(
        board_size=board_size
    )

    model.load_state_dict(
        torch.load(
            path,
            map_location="cpu"
        )
    )

    model.eval()

    return model


class NetworkEvaluator:

    def __init__(
        self,
        model,
        device="cpu"
    ):

        self.model = model

        self.device = torch.device(
            device
        )

        self.model.to(
            self.device
        )

        self.model.eval()

    def _encode_state(
        self,
        state
    ):

        if hasattr(state, "board"):

            board = state.board

            current_player = (
                state.current_player
            )

        elif isinstance(state, dict):

            board = state["board"]

            current_player = (
                state.get(
                    "current_player",
                    1
                )
            )

        else:

            board = state

            current_player = 1

        board = np.asarray(board)

        if board.ndim == 2:

            board = np.stack(
                (
                    board == current_player,
                    board == -current_player,
<<<<<<< HEAD
                    np.full(
                        board.shape,
                        current_player == 1
                    ),
=======
                    # Must match the training encoding in
                    # SelfPlay.get_state: plane 2 = empty points.
                    board == 0,
>>>>>>> origin/shafreed
                )
            )

        elif (
            board.ndim != 3
            or board.shape[0] != 3
        ):

            raise ValueError(
                "state must be a board "
                "or a 3-plane tensor"
            )

        return board.astype(
            np.float32,
            copy=False
        )

    def evaluate(
        self,
        state
    ):

        self.model.eval()

        encoded_state = torch.tensor(
            self._encode_state(state),
            dtype=torch.float32,
            device=self.device
        )

        if encoded_state.dim() == 3:

            encoded_state = (
                encoded_state.unsqueeze(0)
            )

        with torch.no_grad():

            policy, value = self.model(
                encoded_state
            )

        policy = policy.squeeze(0)

        value = value.squeeze(0)

        return (
            policy.cpu().numpy(),
            value.item()
        )