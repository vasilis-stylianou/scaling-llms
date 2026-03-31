from __future__ import annotations
from dataclasses import dataclass
import logging
import math
from pathlib import Path
from typing import Any, Literal
import torch

from scaling_llms.checkpointing import (
    get_model_class_info, 
    instantiate_model_from_run,
    CheckpointManager,
)
from scaling_llms.constants import CKPT_FILES, METADATA_FILES, METRIC_CATS
from scaling_llms.tracking import Run
from scaling_llms.utils.config import BaseJsonConfig
from scaling_llms.utils.training import (
    compute_grad_zero_frac,
    compute_grad_norm,
    compute_param_norm,
    compute_grad_to_param_ratio,
    compute_opt_steps_from_token_budget,
    make_autocast_context,
    make_lr_scheduler,
    make_train_iterator,
    make_trainer_logger,
    make_timer,
)


# -------------------------
# TRAINER CONFIG
# -------------------------
@dataclass
class TrainerConfig(BaseJsonConfig):
    
    # Optimization
    num_steps: int
    lr: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    precision: str = "bf16"
    accum_steps: int = 1
    grad_clip_norm: float | None = 1.0
    device: str = "auto"  # "auto" | "cpu" | "cuda"

    # Dataloader 
    iter_mode: Literal["infinite", "single-batch"] = "infinite"

    # LR Scheduler
    lr_schedule: str | None = None  # "none", "cosine", "linear"
    warmup_steps: int = 0
    min_lr_ratio: float = 0.0

    # Trackers / Logging
    enable_tb: bool = False
    net_log_freq: int = 50
    sys_log_freq: int = 100
    eval_log_freq: int = 500
    ckpt_log_freq: int = -1
    keep_last_n: int | None = 3
    best_eval_nll_tol: float = 1e-4

    # Timer
    enable_cuda_timer: bool = False
    
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
        self._derive_steps_from_budget_if_needed()
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

    def _derive_steps_from_budget_if_needed(self) -> None:
        # If num_steps is explicitly provided, we don't touch it.
        if self.num_steps is not None:
            return

        # Otherwise, we require the token budget parameters to be provided 
        # and derive num_steps from the budget.
        self.num_steps = compute_opt_steps_from_token_budget(
            train_tokens_budget=self.train_tokens_budget,
            micro_batch_size=self.micro_batch_size,
            seq_len=self.seq_len,
            accum_steps=self.accum_steps,
        )

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
        run: Run | None = None,
    ):
        # Key Attributes
        self.cfg = cfg
        self.model = model
        self.train_dl = train_dl
        self.eval_dl = eval_dl
        self.device = cfg.device
        self.run = run

        # Configure training data iterator
        self.train_iter = self._init_train_iter(train_dl, cfg.iter_mode)

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
        self.logger = make_trainer_logger(self.run)

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


    # --- FACTORIES --- 
    @classmethod
    def from_checkpoint(
        cls,
        run: Run,
        ckpt_name: str,
        model: torch.nn.Module | None = None,
        train_dl=None,
        eval_dl=None,
        reset_state=False,
        strict: bool = True,
    ) -> "Trainer":
        """Load trainer from checkpoint.
        
        Args:
            run: Run instance
            ckpt_name: Name of checkpoint file (e.g., "latest.pt")
            model: Model instance. If None, will auto-instantiate from saved metadata.
            train_dl: Training dataloader
            eval_dl: Evaluation dataloader  
            strict: Whether to strictly enforce state dict matching
            reset_state: 
                If True, only load model weights and ignore optimizer/scaler/scheduler 
                and trainer state (e.g., step_idx). 
                Useful for fine-tuning or transfer learning from a checkpoint.
            
        Returns:
            Trainer instance restored from checkpoint
        """
        # Load trainer configs from metadata dir
        cfg = TrainerConfig.from_json(run.artifacts_dir.metadata_path(METADATA_FILES.trainer_config))
        
        # Auto-instantiate model if not provided
        model = model or instantiate_model_from_run(run)

        # Init Trainer
        trainer = cls(cfg=cfg, model=model, train_dl=train_dl, eval_dl=eval_dl, run=run)

        # Configure Trainer's state
        ckpt_path = run.artifacts_dir.checkpoint_path(ckpt_name)
        trainer_state = trainer.ckpt_manager.load(
            ckpt_path, 
            strict=strict, 
            device=cfg.device, 
            weights_only=reset_state # If True, skip loading optimizer/scaler/scheduler
        )
        
        if not reset_state:
            trainer.load_state_dict(trainer_state)

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
            self.run.log_metadata(self.model.cfg, METADATA_FILES.model_config, format="json")
            self.run.log_metadata(get_model_class_info(self.model), METADATA_FILES.model_class, format="json")
            self.run.log_metadata(self.cfg, METADATA_FILES.trainer_config, format="json")
        
        self.logger.log_start(
            model_params=f"{sum(p.numel() for p in self.model.parameters()):,}",
            n_layer=self.model.cfg.n_layer,
            n_embd=self.model.cfg.n_embd,
            vocab_size=f"{self.model.cfg.vocab_size:,}",
            device=self.device,
            device_name=self.cfg.device_name,
            precision=self.cfg.precision,
            max_num_steps=target_total,
            remaining_steps=remaining_steps,
            accum_steps=self.cfg.accum_steps,
            lr=self.cfg.lr,
            step_idx=self.step_idx,
            warmup_steps=self.cfg.warmup_steps,
            lr_schedule=self.cfg.lr_schedule or "none"
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
                (
                    (self.step_idx % self.cfg.eval_log_freq == 0)  # regular eval interval
                    or (self.step_idx == target_total - 1)  # always eval at the last step
                )
            ):
                # Compute and log eval metrics
                eval_metrics = self.evaluate(self.eval_dl)
                self._log_metrics({METRIC_CATS.eval: eval_metrics})

                # Report both train and eval metrics to console 
                self.logger.log_train_step(
                    step=self.step_idx,
                    nll=train_metrics["nll"],
                    ppl=train_metrics["ppl"],
                    tokens_seen_total=train_metrics["tokens_seen_total"],
                    tokens_per_sec=train_metrics["tokens_per_sec"],
                    step_ms=train_metrics["step_ms"],
                    lr=train_metrics["lr"],
                    level=logging.INFO
                )
                self.logger.log_eval_step(
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
                    self.save_checkpoint(CKPT_FILES.best_ckpt, offset_step_idx=1) 
                    # NOTE: resuming from this checkpoint should start at the next optimation step

            # Checkpoint
            if (
                (self.ckpt_manager is not None) and 
                (self.cfg.ckpt_log_freq > 0) and 
                (self.step_idx > 0) and  # Avoid saving checkpoint at step_idx=0
                (self.step_idx % self.cfg.ckpt_log_freq == 0)
            ):
                ckpt_name = f"step_{self.step_idx}.pt"
                self.save_checkpoint(ckpt_name, offset_step_idx=1)
                # NOTE: resuming from this checkpoint should start at the next optimization step
                    
            # Update LR
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # Advance global step index
            self.step_idx += 1

        # POST-TRAINING LOGGING
        # Always save a final checkpoint at the end of training if checkpointing is enabled
        if (self.ckpt_manager is not None) and (self.cfg.ckpt_log_freq > 0):
            self.save_checkpoint(CKPT_FILES.last_ckpt)
            # NOTE: no need to offset step_idx for the final checkpoint since it's already been advanced

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

    def save_checkpoint(self, name: str, offset_step_idx: int = 0) -> Path:
        if self.ckpt_manager is None:
            raise RuntimeError(
                "Cannot save checkpoint when Trainer.ckpt_manager is None."
                "Attach Trainer to a run using trainer.attach_run(run) to enable checkpointing."
            )
        
        trainer_state = self.state_dict()
        trainer_state["step_idx"] += offset_step_idx

        return self.ckpt_manager.save(trainer_state, name)

    def attach_run(self, run: Run) -> None:
        self.run = run
        self.logger = make_trainer_logger(run)
        self.ckpt_manager = self._create_ckpt_manager() 

    def attach_dataloaders(self, train_dl=None, eval_dl=None, iter_mode=None) -> None:
        self.train_dl = train_dl
        self.eval_dl = eval_dl
        iter_mode = iter_mode or self.cfg.iter_mode
        self.train_iter = self._init_train_iter(train_dl, iter_mode)

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

        ## Training Metrics (always computed)
        cat2metrics[METRIC_CATS.train] = self._compute_training_metrics(loss_sum, tokens)

        # 6) Logging
        self._log_metrics(cat2metrics)

        train_metrics = cat2metrics.get(METRIC_CATS.train, {})
        if train_metrics is not None:
            self.logger.log_train_step(**train_metrics)

        return train_metrics

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
        # Peak GPU memory allocated (in GB)
        metrics = {
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

        # Wall time (always available)
        wall_ms = self.wall_timer.elapsed_ms()
        tokens_per_sec = tokens / (wall_ms / 1e3) if wall_ms > 0 else float("nan")

        return {
            "step": int(self.step_idx),
            "loss_sum": float(loss_sum.item()),
            "nll": nll,
            "ppl": ppl,
            "tokens": int(tokens),
            "tokens_seen_total": int(self.tokens_seen_total),
            "step_ms": float(wall_ms),
            "tokens_per_sec": float(tokens_per_sec),
            "lr": float(self.optimizer.param_groups[0]["lr"]),
        }
    
    def _create_ckpt_manager(self):
        return CheckpointManager(
            self.run.checkpoints_dir, 
            self.model,
            self.optimizer,
            self.scaler,
            self.lr_scheduler,
            keep_last_n=self.cfg.keep_last_n,
        )
    
    def _init_train_iter(self, train_dl, iter_mode):
        if train_dl is None:
            return None
        return make_train_iterator(train_dl, iter_mode)