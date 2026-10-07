"""
Neural network trainer for AlphaGo Zero.

Trains the dual-head GoNetwork on (state, policy, value) examples produced
by MCTS self-play.  Supports checkpointing and resumable training.
"""

import os

import torch
import torch.nn.functional as F

from training.checkpoint_manager import CheckpointManager


class Trainer:
    """
    Trains a GoNetwork on self-play data.

    Args:
        model:          GoNetwork instance to train.
        learning_rate:  Initial SGD learning rate.
        weight_decay:   L2 regularisation coefficient.
        device:         Torch device string (``"cuda"`` or ``"cpu"``).
                        Auto-detected when *None*.
        checkpoint_dir: Directory used by the embedded
                        :class:`~training.checkpoint_manager.CheckpointManager`.
    """

    def __init__(
        self,
        model,
        learning_rate=0.001,
        weight_decay=1e-4,
        device=None,
        checkpoint_dir="training/checkpoints",
    ):
        self.model = model

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.model.to(self.device)

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
        )

        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir
        )

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def calculate_loss(
        self,
        predicted_policy,
        predicted_value,
        target_policy,
        target_value,
    ):
        """
        Compute the combined AlphaGo Zero loss.

        Loss = cross-entropy(policy) + MSE(value)

        Args:
            predicted_policy: Network logits  (B, action_space).
            predicted_value:  Network value   (B, 1).
            target_policy:    MCTS policy     (B, action_space).
            target_value:     Game outcome    (B,).

        Returns:
            tuple: (total_loss, policy_loss, value_loss)
        """
        # Policy head: cross-entropy against MCTS visit distribution.
        policy_loss = -torch.mean(
            torch.sum(
                target_policy
                * F.log_softmax(predicted_policy, dim=1),
                dim=1,
            )
        )

        # Value head: mean-squared error vs. game outcome.
        value_loss = F.mse_loss(
            predicted_value.squeeze(-1),
            target_value,
        )

        total_loss = policy_loss + value_loss

        return total_loss, policy_loss, value_loss

    # ------------------------------------------------------------------
    # Single epoch of training
    # ------------------------------------------------------------------

    def train_epoch(self, dataloader):
        """
        Train the model for one epoch over *dataloader*.

        Args:
            dataloader: A PyTorch DataLoader yielding
                        (states, policies, values) batches.

        Returns:
            dict: ``{"loss": …, "policy_loss": …, "value_loss": …}``
        """
        self.model.train()

        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        batches = 0

        for states, policies, values in dataloader:

            states = states.to(self.device)
            policies = policies.to(self.device)
            values = values.to(self.device)

            self.optimizer.zero_grad()

            predicted_policy, predicted_value = self.model(states)

            # GoNetwork already applies softmax; calculate_loss applies
            # log_softmax, so feed it log-probabilities (log_softmax of
            # log-probs is the identity) instead of double-softmaxing.
            predicted_policy = torch.log(predicted_policy.clamp_min(1e-8))

            loss, policy_loss, value_loss = self.calculate_loss(
                predicted_policy,
                predicted_value,
                policies,
                values,
            )

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            batches += 1

        if batches == 0:
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}

        return {
            "loss": total_loss / batches,
            "policy_loss": total_policy_loss / batches,
            "value_loss": total_value_loss / batches,
        }

    # ------------------------------------------------------------------
    # Checkpoint save / load  (Task 25)
    # ------------------------------------------------------------------

    def save_checkpoint(self, path, epoch, iteration=0, metrics=None):
        """
        Save a training checkpoint.

        Supports two calling conventions:

        * ``save_checkpoint(path, epoch)`` — legacy; writes a raw PyTorch
          checkpoint dict to *path* directly (backwards-compatible).
        * When *iteration* is supplied the :class:`CheckpointManager` is
          used, which also keeps a rolling ``checkpoint_latest.pt``.

        Args:
            path:      Destination file path.
            epoch:     Current epoch number (0-indexed).
            iteration: Current outer-loop iteration (default 0).
            metrics:   Optional loss dict to embed.
        """
        if iteration > 0:
            # Use the full CheckpointManager.
            self.checkpoint_manager.save(
                model=self.model,
                optimizer=self.optimizer,
                iteration=iteration,
                epoch=epoch,
                metrics=metrics,
            )
        else:
            # Legacy path: raw state-dict checkpoint at explicit *path*.
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            }
            torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        """
        Load a checkpoint from *path*.

        Supports both raw state-dict files (``{"model_state_dict": …}``)
        and CheckpointManager payloads.

        Args:
            path: Path to the ``.pt`` checkpoint file.

        Returns:
            int: The epoch number stored in the checkpoint.
        """
        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=False,
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return checkpoint["epoch"]

    def resume_from_latest(self):
        """
        Attempt to resume from the latest checkpoint.

        Returns:
            tuple: (iteration, epoch) from the checkpoint, or (0, 0) if
                   no checkpoint exists.
        """
        if not self.checkpoint_manager.checkpoint_exists():
            return 0, 0

        payload = self.checkpoint_manager.load_latest(
            model=self.model,
            optimizer=self.optimizer,
            device=str(self.device),
        )

        if payload is None:
            return 0, 0

        iteration = int(payload.get("iteration", 0))
        epoch = int(payload.get("epoch", 0))

        print(
            f"[Trainer] Resumed from checkpoint: "
            f"iteration={iteration}, epoch={epoch}"
        )

        return iteration, epoch