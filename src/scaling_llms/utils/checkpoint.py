"""
CheckpointManager: Centralized checkpoint save/load logic.
"""
import logging
from pathlib import Path
from typing import Any
import torch

from scaling_llms.constants import CHECKPOINT_KEYS
from scaling_llms.utils.loggers import BaseLogger


class CheckpointManager:
    """
    Manages checkpoint save/load for training state.
    
    Stores references to training objects (model, optimizer, scaler, scheduler)
    and handles checkpoint save/load operations.
    
    Handles:
    - Saving model, optimizer, scaler, scheduler, and step counters
    - Loading and resuming from checkpoints
    - Checkpoint naming conventions (latest, best, step_<N>)
    """
    
    def __init__(
        self,
        checkpoint_dir: Path | str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scaler: Any = None,  # GradScaler
        lr_scheduler: Any = None,  # LambdaLR | None
        device: str | torch.device | None = None
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Store references (not copies) to training objects
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.lr_scheduler = lr_scheduler

        self.device = torch.device(device) if device is not None else None

        # Init Logger
        self.logger = BaseLogger(name="CheckpointManager", level=logging.INFO)
    
    # --- Public API ---
    def save(
        self,
        trainer_state: dict[str, Any],
        name: str = "latest.pt",
    ) -> Path:
        """
        Save checkpoint with model, optimizer, and trainer state.
        
        Args:
            trainer_state: Dictionary with trainer state (e.g., step_idx, tokens_seen_total)
            name: Checkpoint filename (e.g., "latest.pt", "best.pt", "step_1000.pt")
        
        Returns:
            Path to the saved checkpoint
        """    
        ckpt_path = self.checkpoint_dir / name
        self.logger.info(f"[save] Saving checkpoint at step {trainer_state['step_idx']} to {ckpt_path}")

        ckpt = {
            CHECKPOINT_KEYS.model: self.model.state_dict(),
            CHECKPOINT_KEYS.optimizer: self.optimizer.state_dict() if self.optimizer is not None else None,
            CHECKPOINT_KEYS.scaler: self.scaler.state_dict() if (self.scaler is not None and self.scaler.is_enabled()) else None,
            CHECKPOINT_KEYS.lr_scheduler: self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
            CHECKPOINT_KEYS.trainer_state: trainer_state,  # Trainer state dict
        }
        
        torch.save(ckpt, ckpt_path)

        return ckpt_path

    def load(
        self,
        ckpt_path: str | Path,
        device: str | torch.device | None = None,
        strict: bool = True,
        weights_only: bool = False,
    ) -> dict[str, Any]:
        """
        Load checkpoint and restore model/optimizer/scaler/scheduler in-place,
        ensuring everything ends up on `device`.

        Returns:
            trainer_state dict
        """
        device = torch.device(device) if device is not None else self.device

        self.logger.info(f"[load_into] Loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")  # always load safely on CPU

        # 1) Model weights
        self.logger.info("[load_into] Loading model state...")
        self.model.load_state_dict(ckpt[CHECKPOINT_KEYS.model], strict=strict)

        # 2) Move model to device (before optimizer state fix)
        if device is not None:
            self.logger.info(f"[load_into] Moving model to {device} ...")
            self.model.to(device)
        
        if weights_only:
            self.logger.info("[load] weights_only=True: skipping optimizer/scaler/scheduler and trainer_state.")
            return {}
    
        # 3) Optimizer
        if self.optimizer is not None and ckpt.get(CHECKPOINT_KEYS.optimizer) is not None:
            self.logger.info("[load_into] Loading optimizer state...")
            self.optimizer.load_state_dict(ckpt[CHECKPOINT_KEYS.optimizer])

            # Optimizer state tensors are often left on CPU after load_state_dict
            if device is not None:
                self.logger.info("[load_into] Moving optimizer state to device...")
                self._ensure_optimizer_state_device(device)

        # 4) Scaler
        if (
            self.scaler is not None
            and getattr(self.scaler, "is_enabled", lambda: False)()
            and ckpt.get(CHECKPOINT_KEYS.scaler) is not None
        ):
            self.logger.info("[load_into] Loading scaler state...")
            self.scaler.load_state_dict(ckpt[CHECKPOINT_KEYS.scaler])
            # GradScaler state is tiny; if it ever contains tensors, this is safe:
            if device is not None:
                try:
                    sd = self.scaler.state_dict()
                    sd = self._move_tree_to_device(sd, device)
                    self.scaler.load_state_dict(sd)
                except Exception:
                    pass

        # 5) LR scheduler
        if self.lr_scheduler is not None and ckpt.get(CHECKPOINT_KEYS.lr_scheduler) is not None:
            self.logger.info("[load_into] Loading lr_scheduler state...")
            self.lr_scheduler.load_state_dict(ckpt[CHECKPOINT_KEYS.lr_scheduler])

        self.logger.info("[load_into] Done. Returning trainer state.")
        
        return ckpt[CHECKPOINT_KEYS.trainer_state]
    
    # --- Internal helper methods ---
    def _move_tree_to_device(self, obj: Any, device: torch.device) -> Any:
        if torch.is_tensor(obj):
            return obj.to(device, non_blocking=True)
        if isinstance(obj, dict):
            return {k: self._move_tree_to_device(v, device) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            constructor = list if isinstance(obj, list) else tuple
            return constructor(self._move_tree_to_device(v, device) for v in obj)
        return obj

    def _ensure_optimizer_state_device(self, device: torch.device) -> None:
        if self.optimizer is None:
            return
        for state in self.optimizer.state.values():
            for k, v in list(state.items()):
                if torch.is_tensor(v):
                    state[k] = v.to(device, non_blocking=True)
                else:
                    state[k] = self._move_tree_to_device(v, device)