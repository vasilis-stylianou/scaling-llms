"""
CheckpointManager: Centralized checkpoint save/load logic.
"""
from pathlib import Path
from typing import Any
import torch


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
        optimizer: torch.optim.Optimizer,
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
        
        ckpt = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler.is_enabled() else None,
            "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
            "trainer": trainer_state,  # Trainer state dict
        }
        
        torch.save(ckpt, ckpt_path)
        return ckpt_path
    
    def load(
        self,
        path: str | Path,
        strict: bool = True,
    ) -> dict[str, Any]:
        """
        Load checkpoint and restore training state.
        
        Args:
            path: Path to checkpoint file
            strict: Whether to strictly enforce state dict matching
        
        Returns:
            Trainer state dictionary with keys like step_idx, tokens_seen_total, etc.
        """
        ckpt = torch.load(path, map_location="cpu")
        
        # Load training objects
        self.model.load_state_dict(ckpt["model"], strict=strict)
        self.optimizer.load_state_dict(ckpt["optimizer"])
        
        if self.scaler and self.scaler.is_enabled() and (ckpt.get("scaler") is not None):
            self.scaler.load_state_dict(ckpt["scaler"])
        
        if (self.lr_scheduler is not None) and (ckpt.get("lr_scheduler") is not None):
            self.lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        
        # Return trainer state (handle both old and new formats for backwards compatibility)
        if "trainer" in ckpt:
            return ckpt["trainer"]
        else:
            # Legacy format: step_idx and tokens_seen_total at top level
            return {
                "step_idx": int(ckpt.get("step_idx", 0)),
                "tokens_seen_total": int(ckpt.get("tokens_seen_total", 0)),
            }
    
    def get_checkpoint_path(self, name: str) -> Path:
        """Get the full path to a checkpoint file."""
        return self.checkpoint_dir / name
    
    def checkpoint_exists(self, name: str) -> bool:
        """Check if a checkpoint file exists."""
        return (self.checkpoint_dir / name).exists()
    
    def list_checkpoints(self) -> list[Path]:
        """List all checkpoint files in the directory."""
        return sorted(self.checkpoint_dir.glob("*.pt"))
