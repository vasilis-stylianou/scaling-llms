"""
CheckpointManager: Centralized checkpoint save/load logic.
"""
import logging
from pathlib import Path
from typing import Any
import torch

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
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Store references (not copies) to training objects
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.lr_scheduler = lr_scheduler

        # Init Logger
        self.logger = BaseLogger(name="CheckpointManager", level=logging.INFO)
    
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
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
            "scaler": self.scaler.state_dict() if (self.scaler is not None and self.scaler.is_enabled()) else None,
            "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
            "trainer": trainer_state,  # Trainer state dict
        }
        
        torch.save(ckpt, ckpt_path)

        return ckpt_path
    
    def load(
        self,
        ckpt_path: str | Path,
        strict: bool = True,
    ) -> dict[str, Any]:
        """
        Load checkpoint and restore training state.
        
        Args:
            ckpt_path: Path to checkpoint file
            strict: Whether to strictly enforce state dict matching
        
        Returns:
            Trainer state dictionary with keys like step_idx, tokens_seen_total, etc.
        """
        self.logger.info(f"[load] Loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        
        # Load training objects
        self.logger.info("[load] Loading model state...")
        self.model.load_state_dict(ckpt["model"], strict=strict)

        if (self.optimizer is not None) and (ckpt.get("optimizer") is not None):
            self.logger.info("[load] Loading optimizer state...")
            self.optimizer.load_state_dict(ckpt["optimizer"])
        
        if self.scaler and self.scaler.is_enabled() and (ckpt.get("scaler") is not None):
            self.logger.info("[load] Loading scaler state...")
            self.scaler.load_state_dict(ckpt["scaler"])
        
        if (self.lr_scheduler is not None) and (ckpt.get("lr_scheduler") is not None):
            self.logger.info("[load] Loading lr_scheduler state...")
            self.lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        
        self.logger.info("[load] Returning trainer state...")

        return ckpt["trainer"]
        
    
    def get_checkpoint_path(self, name: str) -> Path:
        """Get the full path to a checkpoint file."""
        return self.checkpoint_dir / name
    
    def checkpoint_exists(self, name: str) -> bool:
        """Check if a checkpoint file exists."""
        return (self.checkpoint_dir / name).exists()
    
    def list_checkpoints(self) -> list[Path]:
        """List all checkpoint files in the directory."""
        return sorted(self.checkpoint_dir.glob("*.pt"))
