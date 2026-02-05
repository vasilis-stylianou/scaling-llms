from __future__ import annotations
from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path
from typing import Any
import torch

# Project-local imports
from scaling_llms.tracking.constants import DIRS, FILES, METRICS
from scaling_llms.tracking.managers import RunManager
from scaling_llms.utils.checkpoint import CheckpointManager
from scaling_llms.utils.loggers import TrainerLogger
from scaling_llms.utils.training import (
    infinite_loader,
    compute_grad_zero_frac,
    compute_grad_norm,
    compute_param_norm,
    compute_grad_to_param_ratio,
    make_autocast_context,
    make_lr_scheduler,
    make_timer,
)


# -------------------------
# TRAINER CONFIG
# -------------------------
@dataclass
class TrainerConfig:
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

    # Timer
    timer_mode: str | None = "wall" # "wall" (CPU), "cuda_async", "cuda_sync"

    # Reproducibility
    seed: int = 1234
    

    # --- FACTORIES ---
    @classmethod
    def from_json(cls, path: str | Path) -> "TrainerConfig":
        path = Path(path)
        with path.open("r") as f:
            data = json.load(f)

        # Convert log_dir back to Path if present
        if "log_dir" in data and data["log_dir"] is not None:
            data["log_dir"] = Path(data["log_dir"])

        return cls(**data)
    
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
        self.train_iter = infinite_loader(train_dl)

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

        # Init Timer
        self.timer = make_timer(cfg)
        
        # Init CheckpointManager
        if self.run is not None:
            self.ckpt_manager = CheckpointManager(
                self.run[DIRS.checkpoints],
                self.model,
                self.optimizer,
                self.scaler,
                self.lr_scheduler,
            )
        else:
            self.ckpt_manager = None

        # System Logger
        self.sys_logger = TrainerLogger(
            name="Trainer",
            log_dir=self.run[DIRS.metadata] if self.run else None,  # writes metadata/train.log
            level=logging.INFO,
        )

        # Training State
        self.step_idx: int = 0
        self.tokens_seen_total: int = 0

    # --- STATE DICT ---
    def state_dict(self) -> dict[str, Any]:
        """Get trainer state as a dictionary."""
        return {
            "step_idx": int(self.step_idx),
            "tokens_seen_total": int(self.tokens_seen_total),
        }
    
    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load trainer state from a dictionary."""
        self.step_idx = int(state.get("step_idx", 0))
        self.tokens_seen_total = int(state.get("tokens_seen_total", 0))

    # --- FACTORIES --- 
    @classmethod
    def from_checkpoint(
        cls,
        run_path: str | Path,
        ckpt_name: str,
        model: torch.nn.Module,
        train_dl=None,
        eval_dl=None,
        strict: bool = True,
        restore_rng: bool = True,
    ) -> "Trainer":
        # Resuming run
        run = RunManager(Path(run_path)) 

        # Load trainer configs from metadata dir
        cfg_json_path = run[DIRS.metadata] / FILES.trainer_config
        cfg = TrainerConfig.from_json(cfg_json_path)
        
        # Init Trainer
        trainer = cls(cfg=cfg, model=model, train_dl=train_dl, eval_dl=eval_dl, run=run)

        # Configure Trainer's state
        ckpt_path = run[DIRS.checkpoints] / ckpt_name
        trainer_state = trainer.checkpoint_manager.load(ckpt_path, strict=strict)
        trainer.load_state_dict(trainer_state)

        return trainer


    # --- PUBLIC API ---
    def train(self, num_steps: int | None = None):
        num_steps = num_steps or self.cfg.num_steps

        self.sys_logger.log_start(
            run_root=self.run.root,
            device=self.device,
            precision=self.cfg.precision,
            num_steps=num_steps,
            accum_steps=self.cfg.accum_steps,
            lr=self.cfg.lr,
        )

        # Log configs for fresh runs
        if self.step_idx == 0:
            self._log_configs()

        # MAIN
        self.model.to(self.device)

        for _ in range(num_steps):
            # Optimize
            train_metrics = self.optimizer_step()

            # Evaluate
            if (
                (self.eval_dl is not None) and
                (self.step_idx > 0) and 
                (self.step_idx % self.cfg.eval_log_freq == 0)
            ):
                eval_metrics = self.evaluate(self.eval_dl)
                self._log_metrics({METRICS.eval: eval_metrics})

                self.sys_logger.log_eval(
                    step=self.step_idx, 
                    nll=eval_metrics["nll"], 
                    ppl=eval_metrics["ppl"], 
                    tokens=eval_metrics["tokens"]
                )

            # Checkpoint
            # if (self.step_idx > 0) and (self.step_idx % self.cfg.ckpt_log_freq == 0):
            #     self.save_checkpoint("latest.pt")
            #     self.save_checkpoint(f"step_{self.step_idx}.pt")


            # Update LR
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # Advance global step index
            self.step_idx += 1

        # Ensure last train step is logged
        last_step = self.step_idx - 1
        if (last_step % self.cfg.train_log_freq) != 0:
            self._log_metrics(train_metrics, step=last_step)

        return
    
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

    def save_checkpoint(self, name: str = "latest.pt") -> Path:
        if self.ckpt_manager is None:
            raise RuntimeError("Cannot save checkpoint when Trainer.run is None.")
        
        return self.checkpoint_manager.save(self.state_dict(), name)

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

        if self.timer is not None:
            self.timer.start()

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
        if (self.step_idx > 0) and (self.step_idx % self.cfg.net_log_freq == 0):
            cat2metrics[METRICS.network] = self._compute_network_diagnostics()

        # 4) Optimizer step
        if self.scaler.is_enabled():
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        if self.timer is not None:
            self.timer.stop()

        # 5) Post-step
        self.tokens_seen_total += tokens

        ## System Diagnostics
        if (self.step_idx > 0) and (self.step_idx % self.cfg.sys_log_freq == 0):
            cat2metrics[METRICS.system] = self._compute_system_diagnostics(tokens)

        ## Training Metrics
        if (self.step_idx % self.cfg.train_log_freq == 0):
            cat2metrics[METRICS.train] = self._compute_training_metrics(loss_sum, tokens)

        # 6) Logging
        self._log_metrics(cat2metrics)

        m = cat2metrics.get(METRICS.train)
        s = cat2metrics.get(METRICS.system)
        if m is not None:
            self.sys_logger.log_train_step(
                step=self.step_idx,
                nll=m.get("nll"),
                ppl=m.get("ppl"),
                lr=m.get("lr"),
                tokens_seen_total=m.get("tokens_seen_total"),
                tokens_per_sec=(s.get("tokens_per_sec") if s else None),
                step_ms=(s.get("step_ms") if s else None),
            )

        return cat2metrics.get(METRICS.train, {})

    # --- INTERNALS ---
    # LOGGING METHODS
    def _log_configs(self):
        if self.run is None:
            return
        self.run.log_metadata(self.cfg, FILES.trainer_config, format="json")
        
    def _log_metrics(self, cat2metrics, step=None):
        if self.run is None:
            return

        step = step or self.step_idx

        # Always log metrics as JSONL 
        self.run.log_metrics(cat2metrics, step)

        # Log metrics to TensorBoard if enabled
        if self.cfg.enable_tb:
            self.run.log_tb(cat2metrics, step)

    # METRICS/DIAGNOSTICS 
    def _compute_network_diagnostics(self):
        return {
            "grad_zero_frac": float(compute_grad_zero_frac(self.model)),
            "grad_norm": (gn_pre:=float(compute_grad_norm(self.model))),
            "param_norm": (pn_pre:=float(compute_param_norm(self.model))),
            "grad_to_param_ratio": float(compute_grad_to_param_ratio(gn_pre, pn_pre)),
        }

    def _compute_system_diagnostics(self, tokens):

        if self.timer is not None:
            step_ms = self.timer.elapsed_ms()
            tokens_per_sec = (
                tokens / (step_ms / 1e3) 
                if step_ms > 0 
                else float("nan")
            )
        else:
            step_ms = tokens_per_sec = float("nan")

        return {
            "step_ms": float(step_ms),
            "tokens_per_sec": float(tokens_per_sec),
            "peak_alloc_gb": float(
                torch.cuda.max_memory_allocated() / 1024**3
                if self.device == "cuda"
                else 0.0
            ),
        }

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