from __future__ import annotations
from contextlib import nullcontext
from dataclasses import dataclass
import logging
import math
from pathlib import Path
from typing import Any, Literal
import torch

from scaling_llms.checkpointing import get_model_class_info, CheckpointManager
from scaling_llms.constants import CKPT_FILES, METADATA_FILES, METRIC_CATS
from scaling_llms.distributed import (
    apply_all_reduce_if_distributed,
    is_distributed,
    get_local_rank,
    get_world_size,
    is_ddp_model,
    is_main_process,
)
from scaling_llms.tracking import Run
from scaling_llms.utils.config import BaseJsonConfig
from scaling_llms.utils.training import (
    compute_grad_zero_frac,
    compute_grad_norm,
    compute_param_norm,
    compute_grad_to_param_ratio,
    make_autocast_context,
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
    device: str = "auto"  # requested device: "auto" | "cpu" | "cuda"

    # Multi-GPU
    use_compile: bool = False
    local_rank: int = 0  # set at runtime from LOCAL_RANK env var
    seed: int = 42

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

    # --- PRIVATE METHODS ---
    def __post_init__(self) -> None:
        self._configure_device()
        self._validate_lr_scheduler()

    def _configure_device(self) -> None:
        self.local_rank = get_local_rank()

        # Force CPU mode if requested or CUDA is unavailable.
        if self.device == "cpu" or not torch.cuda.is_available():
            self.device = "cpu"
            self.precision = "fp32"
            self.device_name = "cpu"
            self.use_compile = False
            return

        # In DDP each process owns one GPU identified by local_rank.
        if is_distributed():
            torch.cuda.set_device(self.local_rank)
            device_index = self.local_rank
            self.device = f"cuda:{device_index}"
        else:
            device_index = torch.cuda.current_device()
            self.device = f"cuda:{device_index}"

        self.device_name = torch.cuda.get_device_name(device_index).lower()

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
            raise ValueError(
                f"lr_schedule must be one of {sorted(allowed)}; got {self.lr_schedule}"
            )

        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0; got {self.warmup_steps}")
        if self.warmup_steps > self.num_steps:
            raise ValueError(
                f"warmup_steps must be <= num_steps (={self.num_steps}); got {self.warmup_steps}"
            )

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
        model: torch.nn.Module,  # already wrapped (DDP + compile)
        raw_model: torch.nn.Module,  # unwrapped, used by CheckpointManager
        optimizer: torch.optim.Optimizer,
        scaler: torch.cuda.amp.GradScaler,
        lr_scheduler: Any,
        train_dl=None,
        eval_dl=None,
        run: Run | None = None,
    ):
        # --- Key Attributes ---
        self.cfg = cfg
        self.model = model
        self.raw_model = raw_model
        self.optimizer = optimizer
        self.scaler = scaler
        self.lr_scheduler = lr_scheduler
        self.train_dl = train_dl
        self.eval_dl = eval_dl
        self.device = cfg.device
        self.run = run

        # --- Training State ---
        self.step_idx: int = 0
        self.tokens_seen_total: int = 0
        self.consumed_samples: int = 0
        self.best_eval_nll: float = float("inf")
        self.best_step_idx: int = -1

        # --- Private DDP Attributes ---
        self._is_main = is_main_process()
        self._world_size = get_world_size()
        self._is_ddp = is_ddp_model(self.model)

        # --- Training/Logging Objects & Flags ---
        ## Init training data iterator
        self.train_iter = self._init_train_iter(train_dl, cfg.iter_mode)

        ## Create tracking, logging, and checkpointing flags
        self._tracking_enabled = (run is not None) and self._is_main
        self._console_logging_enabled = self._is_main
        self._checkpointing_enabled = (run is not None) and self._is_main

        ## Init Timers
        self.wall_timer = make_timer("wall") if self._is_main else None
        self.cuda_timer = (
            make_timer("cuda_sync")
            if self._is_main
            and self.device.startswith("cuda")
            and cfg.enable_cuda_timer
            else None
        )

        ## Init CheckpointManager — only on main process
        self.ckpt_manager = (
            self._create_ckpt_manager() if self._checkpointing_enabled else None
        )

        ## Init Logger
        self.logger = make_trainer_logger(self.run)

    # --- STATE DICT ---
    def state_dict(self) -> dict[str, Any]:
        """Get trainer state as a dictionary."""
        return {
            "step_idx": int(self.step_idx),
            "tokens_seen_total": int(self.tokens_seen_total),
            "consumed_samples": int(self.consumed_samples),
            "best_eval_nll": float(self.best_eval_nll),
            "best_step_idx": int(self.best_step_idx),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load trainer state from a dictionary."""
        self.step_idx = int(state.get("step_idx", 0))
        self.tokens_seen_total = int(state.get("tokens_seen_total", 0))
        self.best_eval_nll = float(state.get("best_eval_nll", float("inf")))
        self.best_step_idx = int(state.get("best_step_idx", -1))
        self.consumed_samples = int(state.get("consumed_samples", 0))

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

        # Log model metadata on first run — main process only
        if self._tracking_enabled and (self.step_idx == 0) and (self.run is not None):
            self.run.log_metadata(
                self.raw_model.cfg, METADATA_FILES.model_config, format="json"
            )
            self.run.log_metadata(
                get_model_class_info(self.raw_model),
                METADATA_FILES.model_class,
                format="json",
            )
            self.run.log_metadata(
                self.cfg, METADATA_FILES.trainer_config, format="json"
            )

        if self._console_logging_enabled:
            self.logger.log_start(
                model_params=f"{sum(p.numel() for p in self.raw_model.parameters()):,}",
                n_layer=self.raw_model.cfg.n_layer,
                n_embd=self.raw_model.cfg.n_embd,
                vocab_size=f"{self.raw_model.cfg.vocab_size:,}",
                device=self.device,
                device_name=self.cfg.device_name,
                world_size=self._world_size,
                precision=self.cfg.precision,
                max_num_steps=target_total,
                remaining_steps=remaining_steps,
                accum_steps=self.cfg.accum_steps,
                lr=self.cfg.lr,
                step_idx=self.step_idx,
                warmup_steps=self.cfg.warmup_steps,
                lr_schedule=self.cfg.lr_schedule or "none",
            )

        # MAIN TRAINING LOOP
        for _ in range(remaining_steps):
            # Optimize
            train_metrics = self.optimizer_step()

            if train_metrics and self._console_logging_enabled:
                self.logger.log_train_step(
                    step=self.step_idx,
                    nll=train_metrics["nll"],
                    ppl=train_metrics["ppl"],
                    tokens_seen_total=train_metrics["tokens_seen_total"],
                    tokens_per_sec=train_metrics["tokens_per_sec"],
                    step_ms=train_metrics["step_ms"],
                    lr=train_metrics["lr"],
                    level=logging.INFO,
                )

            # Evaluation 
            if (
                (self.eval_dl is not None)
                and (self.cfg.eval_log_freq > 0)
                and (
                    (
                        self.step_idx % self.cfg.eval_log_freq == 0
                    )  # regular eval interval
                    or (
                        self.step_idx == target_total - 1
                    )  # always eval at the last step
                )
            ):
                # All-reduce eval metrics across GPUs (no-op single GPU)
                eval_metrics = self.evaluate(self.eval_dl)

                # Logging and Checkpointing (main process only)
                ## Log eval metrics to trackers, if enabled
                if self._tracking_enabled:
                    self._log_metrics({METRIC_CATS.eval: eval_metrics})

                ## Report eval metrics to console
                if self._console_logging_enabled:
                    self.logger.log_eval_step(
                        step=self.step_idx,
                        nll=eval_metrics["nll"],
                        ppl=eval_metrics["ppl"],
                        tokens=eval_metrics["tokens"],
                    )

                ## Check for new best checkpoint based on eval nll improvement beyond tolerance threshold
                if (self._checkpointing_enabled) and (
                    eval_metrics["nll"]
                    < self.best_eval_nll - self.cfg.best_eval_nll_tol
                ):
                    if self._console_logging_enabled:
                        self.logger.log_checkpoint(
                            f"New best checkpoint at step {self.step_idx}"
                        )
                    self.best_eval_nll = eval_metrics["nll"]
                    self.best_step_idx = self.step_idx
                    self.save_checkpoint(
                        CKPT_FILES.best_ckpt,
                        offset_step_idx=1,
                        log_step_idx=self.step_idx,
                    )
                    # NOTE: resuming from this checkpoint should start at the next optimization step

            # Checkpoint
            if (
                self._checkpointing_enabled
                and (self.cfg.ckpt_log_freq > 0)
                and (self.step_idx > 0)  # Avoid saving checkpoint at step_idx=0
                and (self.step_idx % self.cfg.ckpt_log_freq == 0)
            ):
                ckpt_name = f"step_{self.step_idx}.pt"
                self.save_checkpoint(
                    ckpt_name, offset_step_idx=1, log_step_idx=self.step_idx
                )
                # NOTE: resuming from this checkpoint should start at the next optimization step

            # Update LR
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # Advance global step index
            self.step_idx += 1

        # POST-TRAINING LOGGING
        # Always save a final checkpoint at the end of training if checkpointing is enabled
        if self._checkpointing_enabled and (self.cfg.ckpt_log_freq > 0):
            self.save_checkpoint(CKPT_FILES.last_ckpt, log_step_idx=self.step_idx - 1)
            # NOTE: no need to offset step_idx for the final checkpoint since it's already been advanced

    @torch.no_grad()
    def evaluate(self, eval_dl) -> dict[str, Any]:
        self.model.eval()

        total_loss = torch.zeros((), device=self.device, dtype=torch.float64)
        total_tokens = torch.zeros((), device=self.device, dtype=torch.long)

        for idx, targets in eval_dl:
            idx = idx.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            out = self.model(idx, targets, loss_reduction="sum")
            total_loss += out.loss.detach().to(torch.float64)
            total_tokens += targets.numel()

        apply_all_reduce_if_distributed(total_loss)
        apply_all_reduce_if_distributed(total_tokens)

        total_loss = float(total_loss.item())
        total_tokens = int(total_tokens.item())

        avg_nll = total_loss / max(1, total_tokens)
        ppl = math.exp(avg_nll) if avg_nll < 20 else float("inf")

        return {
            "nll": avg_nll,
            "ppl": ppl,
            "tokens": total_tokens,
        }

    def save_checkpoint(
        self,
        name: str,
        offset_step_idx: int = 0,
        log_step_idx: int | None = None,
    ) -> Path:
        if self.ckpt_manager is None:
            raise RuntimeError(
                "Cannot save checkpoint when Trainer.ckpt_manager is None."
                "Attach Trainer to a run using trainer.attach_run(run) to enable checkpointing."
            )

        trainer_state = self.state_dict()
        trainer_state["step_idx"] += offset_step_idx

        return self.ckpt_manager.save(trainer_state, name, log_step_idx=log_step_idx)

    def attach_run(self, run: Run) -> None:
        self.run = run
        self._tracking_enabled = (run is not None) and self._is_main
        self._checkpointing_enabled = (run is not None) and self._is_main
        self.logger = make_trainer_logger(self.run)
        if self._checkpointing_enabled:
            self.ckpt_manager = self._create_ckpt_manager()

    def attach_dataloaders(self, train_dl=None, eval_dl=None, iter_mode=None) -> None:
        self.train_dl = train_dl
        self.eval_dl = eval_dl
        iter_mode = iter_mode or self.cfg.iter_mode
        self.train_iter = self._init_train_iter(train_dl, iter_mode)

    def get_runtime_info(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "device_name": self.cfg.device_name,
            "is_distributed": is_distributed(),
            "is_main_process": self._is_main,
            "world_size": self._world_size,
            "local_rank": self.cfg.local_rank,
            "is_ddp_model": self._is_ddp,
        }

    # --- TRAINING CORE ---
    def train_micro_step(self, idx, targets) -> torch.Tensor:

        # Forward (use AMP if precision = "fp16" or "bf16")
        with make_autocast_context(self.device, self.cfg.precision):
            out = self.model(idx, targets, loss_reduction="sum")
            loss_sum_tokens = out.loss

        # Scale loss for backward:
        # - divide by tokens => token-mean gradient
        # - divide by accum_steps => correct grad accumulation equivalence
        loss_for_backward = (
            loss_sum_tokens / max(1, targets.numel())
        ) / self.cfg.accum_steps

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
        if self._is_main and str(self.device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()

        ## Start timers
        if self.wall_timer is not None:
            self.wall_timer.start()
        if self.cuda_timer is not None:
            self.cuda_timer.start()

        loss_sum: torch.Tensor = torch.zeros(
            (), device=self.device, dtype=torch.float64
        )  # accumulate losses on device
        tokens: int = 0
        samples: int = 0
        cat2metrics: dict[str, dict[str, Any]] = {}

        # 2) Grad Accumulation
        for micro_idx in range(self.cfg.accum_steps):
            ## Sample the next batch of indices
            idx, targets = next(self.train_iter)
            idx, targets = idx.to(self.device), targets.to(self.device)

            ## Skip gradient all-reduce on non-final micro-steps (DDP optimisation)
            skip_sync = self._is_ddp and micro_idx < self.cfg.accum_steps - 1
            ctx = self.model.no_sync() if skip_sync else nullcontext()
            with ctx:
                ## Forward/Backward Pass
                loss_sum += self.train_micro_step(idx, targets).to(torch.float64)
                tokens += int(targets.numel())  # assumes no padding
                samples += idx.size(0)

        # NOTE: skip allreduce on non-final micro-steps to avoid
        # redundant cross-GPU gradient syncs during accumulation.
        # DDP normally syncs after every backward(); no_sync() defers
        # that until the final micro-step, reducing communication by
        # a factor of accum_steps.

        # 3) Pre-Step
        ## fp16: unscale before grad stats / clipping
        if self.scaler.is_enabled():
            self.scaler.unscale_(self.optimizer)

        ## Optional grad clipping
        if self.cfg.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.grad_clip_norm
            )

        ## Network diagnostics (rank 0 only)
        if (
            self._tracking_enabled
            and (self.cfg.net_log_freq > 0)
            and (self.step_idx > 0)  # Avoid logging network diagnostics at step_idx=0
            and (self.step_idx % self.cfg.net_log_freq == 0)
        ):
            cat2metrics[METRIC_CATS.network] = self._compute_network_diagnostics()

        # 4) Optimizer step
        if self.scaler.is_enabled():
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        ## Stop timers
        if self.wall_timer is not None:
            self.wall_timer.stop()
        if self.cuda_timer is not None:
            self.cuda_timer.stop()

        # 5) Post-step metrics
        ## All-reduce step metrics across GPUs for consistent logging (no-op single GPU)
        loss_sum, tokens, samples = self._reduce_step_metrics(loss_sum, tokens, samples)

        self.consumed_samples += samples
        self.tokens_seen_total += tokens

        if not self._is_main:
            return {}  # Only the main process computes and logs metrics

        ## Training Metrics (always computed)
        cat2metrics[METRIC_CATS.train] = self._compute_training_metrics(
            loss_sum, tokens
        )

        ## System Diagnostics
        if (
            (self.cfg.sys_log_freq > 0)
            and (self.step_idx > 0)  # Avoid logging system diagnostics at step_idx=0
            and (self.step_idx % self.cfg.sys_log_freq == 0)
        ):
            cat2metrics[METRIC_CATS.system] = self._compute_system_diagnostics(tokens)

        # 6) Metric Logging
        if self._tracking_enabled:
            self._log_metrics(cat2metrics)

        return cat2metrics.get(METRIC_CATS.train, {})

    # --- INTERNALS ---
    # LOGGING METHODS
    def _log_metrics(self, cat2metrics, step=None):
        if not self._tracking_enabled:
            return

        if step is None:
            step = self.step_idx

        # Always log metrics as JSONL
        self.run.log_metrics(cat2metrics, step)

        # Log metrics to TensorBoard if enabled
        if self.cfg.enable_tb:
            self.run.log_tb(cat2metrics, step)

    # POST-STEP PROCESSING METHODS
    def _reduce_step_metrics(
        self,
        loss_sum,
        tokens,
        samples,
    ) -> tuple[torch.Tensor, int, int]:
        tokens = torch.tensor(tokens, device=self.device, dtype=torch.long)
        samples = torch.tensor(samples, device=self.device, dtype=torch.long)

        apply_all_reduce_if_distributed(loss_sum)
        apply_all_reduce_if_distributed(tokens)
        apply_all_reduce_if_distributed(samples)

        tokens = int(tokens.item())
        samples = int(samples.item())

        return loss_sum, tokens, samples

    # METRIC_CATS/DIAGNOSTICS
    def _compute_network_diagnostics(self):
        return {
            "grad_zero_frac": float(compute_grad_zero_frac(self.model)),
            "grad_norm": (gn_pre := float(compute_grad_norm(self.model))),
            "param_norm": (pn_pre := float(compute_param_norm(self.model))),
            "grad_to_param_ratio": float(compute_grad_to_param_ratio(gn_pre, pn_pre)),
        }

    def _compute_system_diagnostics(self, tokens):
        is_cuda = str(self.device).startswith("cuda")

        # Peak GPU memory allocated (in GB)
        metrics = {
            "peak_alloc_gb": float(
                torch.cuda.max_memory_allocated() / 1024**3 if is_cuda else 0.0
            ),
        }

        # CUDA time (GPU compute only)
        if self.cuda_timer is not None:
            cuda_ms = self.cuda_timer.elapsed_ms()

            if cuda_ms > 0:
                cuda_tokens_per_sec_per_gpu = (tokens / self._world_size) / (
                    cuda_ms / 1e3
                )
                cuda_tokens_per_sec = tokens / (cuda_ms / 1e3)
            else:
                cuda_tokens_per_sec_per_gpu = float("nan")
                cuda_tokens_per_sec = float("nan")

            metrics["cuda_step_ms"] = float(cuda_ms)
            metrics["cuda_tokens_per_sec_per_gpu"] = float(cuda_tokens_per_sec_per_gpu)
            metrics["cuda_tokens_per_sec"] = float(cuda_tokens_per_sec)

        return metrics

    def _compute_training_metrics(
        self,
        loss_sum: torch.Tensor,
        tokens: int,
    ) -> dict[str, Any]:
        nll = float((loss_sum / max(1, tokens)).item())
        ppl = math.exp(nll) if nll < 20 else float("inf")
        wall_ms = (
            self.wall_timer.elapsed_ms()
            if self.wall_timer is not None
            else float("nan")
        )
        tokens_per_sec_per_gpu = (
            (tokens / self._world_size) / (wall_ms / 1e3)
            if wall_ms > 0
            else float("nan")
        )
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
            "tokens_per_gpu": int(
                tokens / self._world_size
            ),  # estimate of tokens processed per GPU for this step
            "tokens_per_sec_per_gpu": float(tokens_per_sec_per_gpu),
            "lr": float(self.optimizer.param_groups[0]["lr"]),
        }

    def _create_ckpt_manager(self):
        """
        Attach ckpt manager to the active run's checkpoint dir
        """
        return CheckpointManager(
            self.run.checkpoints_dir,
            self.raw_model,
            self.optimizer,
            self.scaler,
            self.lr_scheduler,
            keep_last_n=self.cfg.keep_last_n,
        )

    def _init_train_iter(self, train_dl, iter_mode):
        if train_dl is None:
            return None
        return make_train_iterator(train_dl, iter_mode)
