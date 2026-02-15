from __future__ import annotations
from dataclasses import dataclass
import importlib
import logging
import math
from pathlib import Path
from typing import Any
import torch
import json

from scaling_llms.constants import RUN_DIRS, RUN_FILES, METRIC_CATS
from scaling_llms.tracking.registries import RunManager
from scaling_llms.tracking.checkpoint import CheckpointManager
from scaling_llms.utils.loggers import TrainerLogger
from scaling_llms.utils.config import BaseJsonConfig
from scaling_llms.utils.training import (
    create_infinite_loader,
    compute_grad_zero_frac,
    compute_grad_norm,
    compute_param_norm,
    compute_grad_to_param_ratio,
    make_autocast_context,
    make_lr_scheduler,
    make_timer,
)


# -------------------------
# HELPER FUNCTIONS
# -------------------------
def get_model_class_info(model: torch.nn.Module) -> dict[str, str]:
    """Extract model class and config class info for serialization."""
    model_class = type(model)
    config_class = type(model.cfg)
    return {
        "model_module": model_class.__module__,
        "model_class_name": model_class.__name__,
        "config_module": config_class.__module__,
        "config_class_name": config_class.__name__,
    }


def load_model_class(class_info: dict[str, str]) -> type:
    """Dynamically import and return the model class."""
    module = importlib.import_module(class_info["module"])
    return getattr(module, class_info["class_name"])


# -------------------------
# TRAINER CONFIG
# -------------------------
@dataclass
class TrainerConfig(BaseJsonConfig):
    # Optimization
    num_steps: int
    lr: float
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    precision: str = "bf16"
    accum_steps: int = 1
    grad_clip_norm: float | None = 1.0
    device: str = "auto"  # "auto" | "cpu" | "cuda"

    # LR Scheduler
    lr_schedule: str | None = None  # "none", "cosine", "linear"
    warmup_steps: int = 0
    min_lr_ratio: float = 0.0

    # Trackers / Logging
    track_metrics: bool = True
    enable_tb: bool = False
    train_log_freq: int = 1
    net_log_freq: int = 50
    sys_log_freq: int = 100
    eval_log_freq: int = 500
    ckpt_log_freq: int = -1
    best_eval_nll_tol: float = 1e-4

    # Timer
    enable_cuda_timer: bool = False

    # Reproducibility
    seed: int = 1234
    
    # --- FACTORIES ---
    @classmethod
    def _postprocess_loaded_data(cls, data: dict[str, Any]) -> dict[str, Any]:
        # Convert log_dir back to Path if present
        if "log_dir" in data and data["log_dir"] is not None:
            data["log_dir"] = Path(data["log_dir"])
        return data
    
    # --- PRIVATE METHODS ---
    def __post_init__(self) -> None:
        self._configure_device()
        self._validate_lr_scheduler()

    def _configure_device(self) -> None:
        # If CUDA isn't available or user chose CPU, force CPU + FP32.
        if (self.device == "cpu") or (not torch.cuda.is_available()):
            self.device = "cpu"
            self.precision = "fp32"
            self.device_name = "cpu"
            return

        # Otherwise, use CUDA and pick a sensible mixed-precision default.
        self.device = "cuda"
        self.device_name = torch.cuda.get_device_name(0).lower()

        # Heuristic defaults by GPU generation.
        if "t4" in self.device_name:
            self.precision = "fp16"
        elif "a100" in self.device_name:
            self.precision = "bf16"
        elif "v100" in self.device_name:
            self.precision = "fp16"

    def _validate_lr_scheduler(self) -> None:
        if self.num_steps <= 0:
            raise ValueError(f"num_steps must be > 0; got {self.num_steps}")
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0; got {self.lr}")

        allowed = {"none", "cosine", "linear"}
        if self.lr_schedule is None:
            self.lr_schedule = "none"
        if self.lr_schedule not in allowed:
            raise ValueError(f"lr_schedule must be one of {sorted(allowed)}; got {self.lr_schedule}")

        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0; got {self.warmup_steps}")
        if self.warmup_steps > self.num_steps:
            raise ValueError(f"warmup_steps must be <= num_steps (={self.num_steps}); got {self.warmup_steps}")

        if not (0.0 <= self.min_lr_ratio <= 1.0):
            raise ValueError(f"min_lr_ratio must be in [0,1]; got {self.min_lr_ratio}")

        # If schedule is none, min_lr_ratio doesn't matter.
        if self.lr_schedule == "none":
            self.min_lr_ratio = 1.0


# -------------------------
# TRAINER CLASS
# -------------------------
class Trainer:

    def __init__(
        self,
        cfg: TrainerConfig,
        model: torch.nn.Module,
        train_dl=None,
        eval_dl=None,
        run: RunManager | None = None,
    ):
        # Key Attributes
        self.cfg = cfg
        self.model = model
        self.train_dl = train_dl
        self.eval_dl = eval_dl
        self.device = cfg.device
        self.run = run

        # TODO 
        self.train_iter = create_infinite_loader(train_dl)

        # Configure training objects
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=(cfg.precision == "fp16") and (self.device == "cuda")
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay,
        )
        self.lr_scheduler = make_lr_scheduler(self.optimizer, cfg)

        # Init Timers
        self.wall_timer = make_timer("wall")
        self.cuda_timer = make_timer("cuda_sync") if (self.device == "cuda") and cfg.enable_cuda_timer else None
        
        # Init CheckpointManager
        if self.run is not None:
            # Attach ckpt manager to the active run's checkpoint dir
            self.ckpt_manager = self._create_ckpt_manager()
        else:
            self.ckpt_manager = None

        # Init Logger
        self.logger = TrainerLogger(
            name="Trainer",
            file_name=str(RUN_FILES.train_log) if self.run else None,
            log_dir=self.run.get_metadata_dir() if self.run else None,  # writes metadata/train.log
            level=logging.INFO,
        )

        # Training State
        self.step_idx: int = 0
        self.tokens_seen_total: int = 0
        self.best_eval_nll: float = float("inf")
        self.best_step_idx: int = -1

    # --- STATE DICT ---
    def state_dict(self) -> dict[str, Any]:
        """Get trainer state as a dictionary."""
        return {
            "step_idx": int(self.step_idx),
            "tokens_seen_total": int(self.tokens_seen_total),
            "best_eval_nll": float(self.best_eval_nll),
            "best_step_idx": int(self.best_step_idx),
        }
    
    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load trainer state from a dictionary."""
        self.step_idx = int(state.get("step_idx", 0))
        self.tokens_seen_total = int(state.get("tokens_seen_total", 0))
        self.best_eval_nll = float(state.get("best_eval_nll", float("inf")))
        self.best_step_idx = int(state.get("best_step_idx", -1))

    def reset_state(self) -> None:
        """Reset trainer state to initial values."""
        self.step_idx = 0
        self.tokens_seen_total = 0
        self.best_eval_nll = float("inf")
        self.best_step_idx = -1

    # --- FACTORIES --- 
    @classmethod
    def from_checkpoint(
        cls,
        run: RunManager,
        ckpt_name: str,
        model: torch.nn.Module | None = None,
        train_dl=None,
        eval_dl=None,
        reset_state=False,
        strict: bool = True
    ) -> "Trainer":
        """Load trainer from checkpoint.
        
        Args:
            run_path: Path to the run directory
            ckpt_name: Name of checkpoint file (e.g., "latest.pt")
            model: Model instance. If None, will auto-instantiate from saved metadata.
            train_dl: Training dataloader
            eval_dl: Evaluation dataloader  
            strict: Whether to strictly enforce state dict matching
            
        Returns:
            Trainer instance restored from checkpoint
        """
        # Load trainer configs from metadata dir
        cfg = TrainerConfig.from_json(run.get_metadata_path(RUN_FILES.trainer_config))
        
        # Auto-instantiate model if not provided
        if model is None:
            
            # Load model and config class info
            model_class_path = run.get_metadata_path(RUN_FILES.model_class)
            class_info = json.loads(model_class_path.read_text())

            # Dynamically load the model class
            ModelClass = load_model_class({
                "module": class_info["model_module"],
                "class_name": class_info["model_class_name"]
            })
            
            # Dynamically load the config class
            ConfigClass = load_model_class({
                "module": class_info["config_module"],
                "class_name": class_info["config_class_name"]
            })
            
            # Load model config
            model_cfg_path = run.get_metadata_path(RUN_FILES.model_config)
            model_cfg = ConfigClass.from_json(model_cfg_path)
            
            # Instantiate model
            model = ModelClass(model_cfg)

        # Init Trainer
        trainer = cls(cfg=cfg, model=model, train_dl=train_dl, eval_dl=eval_dl, run=run)

        # Configure Trainer's state
        ckpt_path = run.get_checkpoint_path(ckpt_name)
        trainer_state = trainer.ckpt_manager.load(ckpt_path, strict=strict)
        trainer.load_state_dict(trainer_state)
        
        if reset_state:
            trainer.logger.log_checkpoint("Resetting trainer state.")
            trainer.reset_state()

        return trainer


    # --- PUBLIC API ---
    def train(self, max_steps: int | None = None):
        # PRE-TRAINING SETUP
        # Determine target total steps and remaining steps
        target_total = self.cfg.num_steps if max_steps is None else max_steps
        remaining_steps = target_total - self.step_idx
        if remaining_steps <= 0:
            raise ValueError(
                f"Training already complete or beyond target. "
                f"Current step_idx={self.step_idx}, target_total={target_total}. "
                f"To continue training, pass max_steps > {self.step_idx}."
            )
        
        # Log model metadata on first run (step_idx == 0)
        if self.step_idx == 0 and self.run is not None:
            self.run.log_metadata(self.model.cfg, RUN_FILES.model_config, format="json")
            self.run.log_metadata(get_model_class_info(self.model), RUN_FILES.model_class, format="json")
            self.run.log_metadata(self.cfg, RUN_FILES.trainer_config, format="json")
        
        self.logger.log_start(
            model_params=f"{sum(p.numel() for p in self.model.parameters()):,}",
            n_layer=self.model.cfg.n_layer,
            n_embd=self.model.cfg.n_embd,
            vocab_size=f"{self.model.cfg.vocab_size:,}",
            device=self.device,
            precision=self.cfg.precision,
            num_steps=remaining_steps,
            accum_steps=self.cfg.accum_steps,
            lr=self.cfg.lr,
            step_idx=self.step_idx
        )

        # MAIN TRAINING LOOP
        self.model.to(self.device)
        for _ in range(remaining_steps):
            # Optimize
            train_metrics = self.optimizer_step()

            # Evaluate
            if (
                (self.eval_dl is not None) and
                (self.cfg.eval_log_freq > 0) and
                (self.step_idx > 0) and  # Avoid logging eval metrics at step_idx=0
                (self.step_idx % self.cfg.eval_log_freq == 0)
            ):
                eval_metrics = self.evaluate(self.eval_dl)
                self._log_metrics({METRIC_CATS.eval: eval_metrics})

                self.logger.log_eval(
                    step=self.step_idx, 
                    nll=eval_metrics["nll"], 
                    ppl=eval_metrics["ppl"], 
                    tokens=eval_metrics["tokens"]
                )
                
                # Check for new best checkpoint based on eval nll improvement beyond tolerance threshold
                if (
                    (self.ckpt_manager is not None) and
                    (eval_metrics["nll"] < self.best_eval_nll - self.cfg.best_eval_nll_tol)
                ):
                    self.logger.log_checkpoint(f"New best checkpoint at step {self.step_idx}")
                    self.best_eval_nll = eval_metrics["nll"]
                    self.best_step_idx = self.step_idx
                    self.save_checkpoint(RUN_FILES.best_ckpt)

            # Checkpoint
            if (
                (self.ckpt_manager is not None) and 
                (self.cfg.ckpt_log_freq > 0) and 
                (self.step_idx > 0) and  # Avoid saving checkpoint at step_idx=0
                (self.step_idx % self.cfg.ckpt_log_freq == 0)
            ):
                self.save_checkpoint(RUN_FILES.last_ckpt)
                    
            # Update LR
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # Advance global step index
            self.step_idx += 1

        # POST-TRAINING LOGGING
        last_step = self.step_idx - 1  # last completed step index after training loop        
        if last_step <= 0:
            return # No training steps were taken, so nothing to log

        # Ensure last train step is always logged
        if (self.cfg.train_log_freq > 0) and ((last_step % self.cfg.train_log_freq) != 0):
            self._log_metrics(train_metrics, step=last_step)

        # Always save a final checkpoint at the end of training if checkpointing is enabled
        if (self.ckpt_manager is not None) and (self.cfg.ckpt_log_freq > 0):
            self.save_checkpoint(RUN_FILES.last_ckpt)

    @torch.no_grad()
    def evaluate(self, eval_dl) -> dict:
        self.model.to(self.device)
        self.model.eval()

        total_loss = 0
        total_tokens = 0

        for idx,targets in eval_dl:
            idx = idx.to(self.device, non_blocking=True) # apply async CPU→GPU copy if possible (pin_mem + CUDA)
            targets = targets.to(self.device, non_blocking=True)

            out = self.model(idx, targets, loss_reduction="sum")

            total_loss += float(out.loss.item())
            total_tokens += int(targets.numel())

        avg_nll = total_loss / max(1, total_tokens)
        ppl = math.exp(avg_nll) if avg_nll < 20 else float("inf")  # avoid overflow

        return {
            "nll": avg_nll,
            "ppl": ppl,
            "tokens": total_tokens,
        }

    def save_checkpoint(self, name: str) -> Path:
        if self.ckpt_manager is None:
            raise RuntimeError(
                "Cannot save checkpoint when Trainer.ckpt_manager is None."
                "Attach Trainer to a run using trainer.attach_run(run) to enable checkpointing."
            )
        
        return self.ckpt_manager.save(self.state_dict(), name)

    def attach_run(self, run: RunManager) -> None:
        self.run = run
        self.logger.log_dir = run.get_metadata_dir()
        self.ckpt_manager = self._create_ckpt_manager() 

    # --- TRAINING CORE ---
    def train_micro_step(self, idx, targets) -> torch.Tensor:

        # Forward (use AMP if precision = "fp16" or "bf16")
        with make_autocast_context(self.device, self.cfg.precision):
            out = self.model(idx, targets, loss_reduction="sum")
            loss_sum_tokens = out.loss
            
        # Scale loss for backward:
        # - divide by tokens => token-mean gradient
        # - divide by accum_steps => correct grad accumulation equivalence
        loss_for_backward = (loss_sum_tokens / max(1, targets.numel())) / self.cfg.accum_steps

        # Backward (scale only for fp16)
        if self.scaler.is_enabled():
            self.scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()

        return loss_sum_tokens.detach()

    def optimizer_step(self) -> dict[str, Any]:
        
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        # 1) Init counters and data vars
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        # Start timers
        self.wall_timer.start()
        if self.cuda_timer is not None:
            self.cuda_timer.start()

        loss_sum: torch.Tensor = torch.zeros((), device=self.device) # accumulate losses on device
        tokens: int = 0
        cat2metrics: dict[str, dict[str, Any]] = {}

        # 2) Grad Accumulation
        for _ in range(self.cfg.accum_steps):

            ## Sample the next batch of indices
            idx, targets = next(self.train_iter)
            idx, targets = idx.to(self.device), targets.to(self.device)

            ## Forward/Backward Pass
            loss_sum += self.train_micro_step(idx, targets)
            tokens += int(targets.numel()) # assumes no padding

        # 3) Pre-Step
        ## fp16: unscale before grad stats / clipping
        if self.scaler.is_enabled():
            self.scaler.unscale_(self.optimizer)

        ## Optional grad clipping
        if self.cfg.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)

        ## Network diagnostics (using the grads/params used by the optimizer)
        if (
            (self.cfg.net_log_freq > 0) and 
            (self.step_idx > 0) and  # Avoid logging network diagnostics at step_idx=0
            (self.step_idx % self.cfg.net_log_freq == 0)
        ):
            cat2metrics[METRIC_CATS.network] = self._compute_network_diagnostics()

        # 4) Optimizer step
        if self.scaler.is_enabled():
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        # Stop timers
        self.wall_timer.stop()
        if self.cuda_timer is not None:
            self.cuda_timer.stop()

        # 5) Post-step
        self.tokens_seen_total += tokens

        ## System Diagnostics
        if (
            (self.cfg.sys_log_freq > 0) and 
            (self.step_idx > 0) and  # Avoid logging system diagnostics at step_idx=0 
            (self.step_idx % self.cfg.sys_log_freq == 0)
        ):
            cat2metrics[METRIC_CATS.system] = self._compute_system_diagnostics(tokens)

        ## Training Metrics
        if (
            (self.cfg.train_log_freq > 0) and 
            (self.step_idx % self.cfg.train_log_freq == 0)
        ):
            cat2metrics[METRIC_CATS.train] = self._compute_training_metrics(loss_sum, tokens)

        # 6) Logging
        self._log_metrics(cat2metrics)

        m = cat2metrics.get(METRIC_CATS.train)
        s = cat2metrics.get(METRIC_CATS.system)
        if m is not None:
            self.logger.log_train_step(
                step=self.step_idx,
                nll=m.get("nll"),
                ppl=m.get("ppl"),
                lr=m.get("lr"),
                tokens_seen_total=m.get("tokens_seen_total"),
                tokens_per_sec=(s.get("tokens_per_sec") if s else None),
                step_ms=(s.get("step_ms") if s else None),
                peak_alloc_gb=(s.get("peak_alloc_gb") if s else None),
            )

        return cat2metrics.get(METRIC_CATS.train, {})

    # --- INTERNALS ---
    # LOGGING METHODS        
    def _log_metrics(self, cat2metrics, step=None):
        if self.run is None:
            return

        step = step or self.step_idx

        # Always log metrics as JSONL 
        self.run.log_metrics(cat2metrics, step)

        # Log metrics to TensorBoard if enabled
        if self.cfg.enable_tb:
            self.run.log_tb(cat2metrics, step)

    # METRIC_CATS/DIAGNOSTICS 
    def _compute_network_diagnostics(self):
        return {
            "grad_zero_frac": float(compute_grad_zero_frac(self.model)),
            "grad_norm": (gn_pre:=float(compute_grad_norm(self.model))),
            "param_norm": (pn_pre:=float(compute_param_norm(self.model))),
            "grad_to_param_ratio": float(compute_grad_to_param_ratio(gn_pre, pn_pre)),
        }

    def _compute_system_diagnostics(self, tokens):
        # Wall time (always available)
        wall_ms = self.wall_timer.elapsed_ms()
        tokens_per_sec = tokens / (wall_ms / 1e3) if wall_ms > 0 else float("nan")

        metrics = {
            "step_ms": float(wall_ms),
            "tokens_per_sec": float(tokens_per_sec),
            "peak_alloc_gb": float(
                torch.cuda.max_memory_allocated() / 1024**3
                if self.device == "cuda"
                else 0.0
            ),
        }

        # CUDA time (only if enabled and on CUDA)
        if self.cuda_timer is not None:
            cuda_ms = self.cuda_timer.elapsed_ms()
            cuda_tokens_per_sec = tokens / (cuda_ms / 1e3) if cuda_ms > 0 else float("nan")
            metrics["cuda_step_ms"] = float(cuda_ms)
            metrics["cuda_tokens_per_sec"] = float(cuda_tokens_per_sec)

        return metrics

    def _compute_training_metrics(self, loss_sum: torch.Tensor, tokens: int) -> dict[str, Any]:
        nll = float((loss_sum / max(1, tokens)).item())
        ppl = math.exp(nll) if nll < 20 else float("inf")

        return {
            "step": int(self.step_idx),
            "loss_sum": float(loss_sum.item()),
            "nll": nll,
            "ppl": ppl,
            "tokens": int(tokens),
            "tokens_seen_total": int(self.tokens_seen_total),
            "lr": float(self.optimizer.param_groups[0]["lr"]),
        }
    
    def _create_ckpt_manager(self):
        return CheckpointManager(
            self.run.get_checkpoint_dir(), 
            self.model,
            self.optimizer,
            self.scaler,
            self.lr_scheduler,
        )