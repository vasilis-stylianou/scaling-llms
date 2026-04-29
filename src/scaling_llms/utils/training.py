import os
import logging
import math
from contextlib import nullcontext
from enum import StrEnum
import torch
from torch.optim.lr_scheduler import LambdaLR
from scaling_llms.constants import METADATA_FILES
from scaling_llms.tracking import Run
from scaling_llms.utils.loggers import TrainerLogger
from scaling_llms.utils.timer import DeviceTimer


# -----------------------------
# DETERMINISM
# -----------------------------
def set_determinism(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Strict determinism (can throw if you use nondeterministic ops)
    # Comment out if it blocks you during early bring-up.
    torch.use_deterministic_algorithms(True)


# -----------------------------
# DATALOADER MODES
# -----------------------------
def create_infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch


def create_single_batch_loader(loader):
    first_batch = next(iter(loader))
    while True:
        yield first_batch


def make_train_iterator(loader, mode: str):
    if mode == "infinite":
        return create_infinite_loader(loader)
    elif mode == "single-batch":
        return create_single_batch_loader(loader)
    else:
        raise ValueError(f"Unknown iter_mode: {mode}")

# -----------------------------
# METRIC_CATS/DIAGNOSTICS
# -----------------------------
@torch.no_grad()
def compute_grad_zero_frac(model: torch.nn.Module) -> float:
    total = 0
    zeros = 0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad
        total += g.numel()
        zeros += (g == 0).sum().item()
    return 0.0 if total == 0 else zeros / total


@torch.no_grad()
def compute_grad_norm(model: torch.nn.Module) -> float:
    # global L2 norm: ||g||_2 over all parameters
    sq = 0.0
    for p in model.parameters():
        g = p.grad
        if g is None:
            continue
        gn = g.float().norm().item()
        sq += gn * gn
    return math.sqrt(sq)


@torch.no_grad()
def compute_param_norm(model: torch.nn.Module) -> float:
    # global L2 norm: ||θ||_2 over all parameters
    sq = 0.0
    for p in model.parameters():
        pn = p.data.float().norm().item()
        sq += pn * pn
    return math.sqrt(sq)


@torch.no_grad()
def compute_grad_to_param_ratio(grad_norm, param_norm, eps=1e-12) -> float:
    return grad_norm / (param_norm + eps)


# -----------------------------
# DEVICE
# -----------------------------
def make_autocast_context(device, precision: str):
    is_cuda = (str(device).startswith("cuda")) or (hasattr(device, "type") and device.type == "cuda")
    if not is_cuda:
        return nullcontext()
    if precision == "fp16":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    if precision == "bf16":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()

def make_timer(timer_mode: str) -> DeviceTimer:

    if timer_mode == "wall":
        return DeviceTimer(device="cpu", sync=False)
    elif timer_mode == "cuda_async":
        return DeviceTimer(device="cuda", sync=False)
    elif timer_mode == "cuda_sync":
        return DeviceTimer(device="cuda", sync=True)
    else:
        raise ValueError(f"Invalid timer_mode: {timer_mode}")


# -----------------------------
# OPTIMIZER
# -----------------------------
def compute_effective_weight_decay(
    weight_decay: float,
    n_layer: int,
    weight_decay_base_depth: int | None,
) -> float:
    if weight_decay < 0:
        raise ValueError(f"weight_decay must be >= 0; got {weight_decay}")
    if n_layer <= 0:
        raise ValueError(f"n_layer must be > 0; got {n_layer}")

    if weight_decay_base_depth is None:
        return weight_decay

    if weight_decay_base_depth <= 0:
        raise ValueError(
            f"weight_decay_base_depth must be > 0 when set; got {weight_decay_base_depth}"
        )

    return weight_decay * (weight_decay_base_depth / n_layer) ** 2


def make_adamw_optimizer(
    model,
    lr: float,
    weight_decay: float,
    beta1: float,
    beta2: float,
    fused: bool | None,
    n_layer: int,
    weight_decay_base_depth: int | None = None,
):
    effective_weight_decay = compute_effective_weight_decay(
        weight_decay=weight_decay,
        n_layer=n_layer,
        weight_decay_base_depth=weight_decay_base_depth,
    )
    param_groups = model.get_param_groups(
        base_lr=lr,
        weight_decay=effective_weight_decay,
    )
    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(beta1, beta2),
        fused=fused,
    )
    return optimizer


# -----------------------------
# LR SCHEDULER
# -----------------------------
class LRSchedule(StrEnum):
    none = "none"
    linear = "linear"
    cosine = "cosine"
    trapezoidal = "trapezoidal"

    
def make_lr_scheduler(
    optimizer,
    lr_schedule: str | None,
    num_steps: int,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.0,
    decay_fraction: float = 0.0,
):

    if lr_schedule is None or lr_schedule == LRSchedule.none:
        return None

    try:
        lr_schedule = LRSchedule(lr_schedule)
    except ValueError as e:
        allowed = [s.value for s in LRSchedule]
        raise ValueError(
            f"Unsupported lr_schedule: {lr_schedule}; must be one of {allowed}"
        ) from e

    warmup_steps = int(warmup_steps)
    total_steps = int(num_steps)
    min_lr_ratio = float(min_lr_ratio)

    if total_steps <= 0:
        raise ValueError(f"num_steps must be > 0; got {total_steps}")
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be >= 0; got {warmup_steps}")
    if warmup_steps > total_steps:
        raise ValueError(
            f"warmup_steps ({warmup_steps}) must be <= num_steps ({total_steps})"
        )
    if not (0.0 <= min_lr_ratio <= 1.0):
        raise ValueError(f"min_lr_ratio must be in [0, 1]; got {min_lr_ratio}")

    decay_steps = None
    stable_end = None

    if lr_schedule == LRSchedule.trapezoidal:
        decay_fraction = float(decay_fraction)

        if not (0.0 < decay_fraction <= 1.0):
            raise ValueError(
                f"decay_fraction must be in (0, 1] for trapezoidal; got {decay_fraction}"
            )

        decay_steps = int(round(decay_fraction * total_steps))

        if warmup_steps + decay_steps > total_steps:
            raise ValueError(
                f"warmup_steps ({warmup_steps}) + decay_steps ({decay_steps}) "
                f"exceeds num_steps ({total_steps})"
            )

        stable_end = total_steps - decay_steps

    def warmup_multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        return 1.0

    def decay_progress(step: int, start_step: int, num_decay_steps: int) -> float:
        progress = float(step - start_step + 1) / float(max(1, num_decay_steps))
        return min(max(progress, 0.0), 1.0)

    def lr_lambda(step: int):
        if warmup_steps > 0 and step < warmup_steps:
            return warmup_multiplier(step)

        if lr_schedule == LRSchedule.trapezoidal:
            assert stable_end is not None
            assert decay_steps is not None

            if step < stable_end:
                return 1.0

            progress = decay_progress(step, stable_end, decay_steps)
            return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 - progress)

        denom = max(1, total_steps - warmup_steps)
        progress = float(step - warmup_steps + 1) / float(denom)
        progress = min(max(progress, 0.0), 1.0)

        if lr_schedule == LRSchedule.cosine:
            return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (
                1.0 + math.cos(math.pi * progress)
            )

        return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 - progress)

    return LambdaLR(optimizer, lr_lambda)


# -----------------------------
# TOKEN BUDGETING
# -----------------------------
def compute_opt_steps_from_token_budget(
    train_tokens_budget: int, 
    micro_batch_size: int, 
    seq_len: int, 
    accum_steps: int
) -> dict[str, int]:
    def _ceil_div(a: int, b: int) -> int:
        """Helper function to compute ceiling division of a by b."""
        return (a + b - 1) // b
    
    # Validate inputs
    if train_tokens_budget is None:
        raise ValueError(
            "train_tokens_budget is None. "
            "Provide train_tokens_budget + (micro_batch_size, seq_len, accum_steps)."
        )

    if (micro_batch_size is None) or (seq_len is None):
        raise ValueError(
            "To derive num_steps you must provide "
            "micro_batch_size and seq_len (and accum_steps)."
        )

    if train_tokens_budget <= 0:
        raise ValueError(f"train_tokens_budget must be > 0; got {train_tokens_budget}")
    if micro_batch_size <= 0:
        raise ValueError(f"micro_batch_size must be > 0; got {micro_batch_size}")
    if seq_len <= 0:
        raise ValueError(f"seq_len must be > 0; got {seq_len}")
    if accum_steps <= 0:
        raise ValueError(f"accum_steps must be > 0; got {accum_steps}")

    # Compute tokens per step: micro_batch_size * seq_len * accum_steps
    tokens_per_step = int(micro_batch_size) * int(seq_len) * int(accum_steps)
    if tokens_per_step <= 0:
        raise ValueError(f"Invalid tokens_per_step computed: {tokens_per_step}")

    # Return only the derived number of optimizer steps (int)
    return max(1, _ceil_div(int(train_tokens_budget), tokens_per_step))


# -----------------------------
# LOGGING
# -----------------------------
def make_trainer_logger(run: Run | None) -> TrainerLogger:
    logger = TrainerLogger(
        name="Trainer",  # avoid collisions across runs
        log_dir=run.metadata_dir if run is not None else None,  # writes metadata/train.log
        file_name=str(METADATA_FILES.train_log) if run is not None else None,
        level=logging.DEBUG, # global logger threshold (allow DEBUG messages through)
        propagate_to_root=False, # do NOT propagate to root
        file_level=logging.DEBUG, # file captures everything
        console=True,
        console_level=logging.INFO, # console prints high-level only
    )

    # Ensure that the logger is properly initialized and the log file is created 
    if run:
        logger.info("Logger initialized")
        logger.flush()

    return logger