"""
Checkpoint management for AlphaGo Zero training.

Provides save / load helpers that preserve the complete training state
(model weights, optimiser state, epoch, iteration, and loss history) so
that an interrupted training run can be resumed from exactly where it left
off.
"""

import json
import os
import time

import torch


class CheckpointManager:
    """
    Manages saving and loading of training checkpoints.

    Checkpoints are written under *checkpoint_dir* with the naming scheme::

        checkpoint_iter{iteration:04d}_epoch{epoch:04d}.pt
        checkpoint_latest.pt   (symlink / copy of the most recent checkpoint)

    Args:
        checkpoint_dir: Directory to write checkpoints into.
        max_to_keep:    Maximum number of per-iteration checkpoints to
                        retain on disk (oldest are pruned automatically).
                        Set to 0 to keep all.
    """

    LATEST_NAME = "checkpoint_latest.pt"
    META_NAME = "checkpoint_meta.json"

    def __init__(
        self,
        checkpoint_dir="training/checkpoints",
        max_to_keep=5,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.max_to_keep = max_to_keep

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Ordered list of checkpoint filenames (oldest → newest).
        self._history = self._load_meta()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _meta_path(self):
        return os.path.join(self.checkpoint_dir, self.META_NAME)

    def _latest_path(self):
        return os.path.join(self.checkpoint_dir, self.LATEST_NAME)

    def _load_meta(self):
        """Return the persisted checkpoint history list (or empty list)."""
        path = self._meta_path()
        if os.path.exists(path):
            try:
                with open(path, "r") as fh:
                    data = json.load(fh)
                return data.get("history", [])
            except (json.JSONDecodeError, KeyError):
                return []
        return []

    def _save_meta(self):
        """Persist the checkpoint history list to disk."""
        with open(self._meta_path(), "w") as fh:
            json.dump({"history": self._history}, fh, indent=2)

    def _prune_old(self):
        """Delete oldest checkpoints if more than *max_to_keep* exist."""
        if self.max_to_keep <= 0:
            return

        while len(self._history) > self.max_to_keep:
            oldest = self._history.pop(0)
            oldest_path = os.path.join(self.checkpoint_dir, oldest)
            if os.path.exists(oldest_path):
                try:
                    os.remove(oldest_path)
                    print(f"[CheckpointManager] Pruned old checkpoint: {oldest}")
                except OSError:
                    pass  # Non-fatal – file may already be gone.

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        model,
        optimizer,
        iteration,
        epoch,
        metrics=None,
        extra=None,
    ):
        """
        Save a training checkpoint.

        Args:
            model:     The GoNetwork instance.
            optimizer: The current torch optimiser.
            iteration: Current AlphaGo Zero outer-loop iteration number.
            epoch:     Training epoch number within *iteration*.
            metrics:   Optional dict of loss values to embed in the file.
            extra:     Optional dict of any additional data to store.

        Returns:
            str: Absolute path of the written checkpoint file.
        """
        filename = (
            f"checkpoint_iter{iteration:04d}_epoch{epoch:04d}.pt"
        )
        path = os.path.join(self.checkpoint_dir, filename)

        payload = {
            "iteration": iteration,
            "epoch": epoch,
            "timestamp": time.time(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics or {},
            "extra": extra or {},
        }

        torch.save(payload, path)

        # Overwrite the "latest" shortcut.
        torch.save(payload, self._latest_path())

        # Update history and prune.
        self._history.append(filename)
        self._prune_old()
        self._save_meta()

        return path

    def load(self, path, model, optimizer=None, device="cpu"):
        """
        Restore a checkpoint into *model* (and optionally *optimizer*).

        Args:
            path:      Checkpoint file path.
            model:     GoNetwork instance to restore weights into.
            optimizer: If provided, restore optimizer state as well.
            device:    Map-location for ``torch.load``.

        Returns:
            dict: The full checkpoint payload (contains *iteration*,
                  *epoch*, *metrics*, etc.).
        """
        payload = torch.load(
            path,
            map_location=device,
            weights_only=False,
        )

        model.load_state_dict(payload["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])

        return payload

    def load_latest(self, model, optimizer=None, device="cpu"):
        """
        Restore the most recent checkpoint if one exists.

        Returns:
            dict or None: Checkpoint payload, or *None* if no checkpoint
                          exists yet.
        """
        path = self._latest_path()
        if not os.path.exists(path):
            return None

        return self.load(path, model, optimizer=optimizer, device=device)

    def latest_iteration(self):
        """
        Return the iteration number stored in the latest checkpoint,
        or 0 if no checkpoint exists.
        """
        path = self._latest_path()
        if not os.path.exists(path):
            return 0

        try:
            payload = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )
            return int(payload.get("iteration", 0))
        except Exception:
            return 0

    def latest_epoch(self):
        """
        Return the epoch number stored in the latest checkpoint, or 0.
        """
        path = self._latest_path()
        if not os.path.exists(path):
            return 0

        try:
            payload = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )
            return int(payload.get("epoch", 0))
        except Exception:
            return 0

    def list_checkpoints(self):
        """Return a list of known checkpoint filenames (oldest → newest)."""
        return list(self._history)

    def checkpoint_exists(self):
        """Return *True* if at least one checkpoint is available."""
        return os.path.exists(self._latest_path())
