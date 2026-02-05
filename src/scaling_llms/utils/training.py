

import math
import random
from contextlib import nullcontext
from typing import Any
import torch
from torch.optim.lr_scheduler import LambdaLR
from scaling_llms.utils.timer import DeviceTimer


def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch


# -----------------------------
# METRICS/DIAGNOSTICS
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

def make_timer(cfg) -> DeviceTimer | None:
    if cfg.timer_mode is None:
        return None

    if cfg.timer_mode == "wall":
        return DeviceTimer(device="cpu", sync=False)
    elif cfg.timer_mode == "cuda_async":
        return DeviceTimer(device="cuda", sync=False)
    elif cfg.timer_mode == "cuda_sync":
        return DeviceTimer(device="cuda", sync=True)
    else:
        raise ValueError(f"Invalid timer_mode: {cfg.timer_mode}")


# -----------------------------
# OPTIMIZATION
# -----------------------------
def make_lr_scheduler(optimizer, cfg: Any):
    if cfg.lr_schedule in (None, "none"):
        return None

    if cfg.lr_schedule not in ("cosine", "linear"):
        raise ValueError(f"Unsupported lr_schedule: {cfg.lr_schedule}")

    warmup_steps = int(getattr(cfg, "warmup_steps", 0))
    total_steps = int(getattr(cfg, "num_steps", 0))
    min_lr_ratio = float(getattr(cfg, "min_lr_ratio", 0.0))

    if total_steps <= 0:
        raise ValueError(f"num_steps must be > 0; got {total_steps}")

    def lr_lambda(step: int):
        # warmup: linear ramp to 1.0
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))

        # progress in [0,1]
        denom = max(1, total_steps - warmup_steps)
        progress = float(step - warmup_steps) / float(denom)
        progress = min(max(progress, 0.0), 1.0)

        if cfg.lr_schedule == "cosine":
            return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))

        # linear
        return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 - progress)

    return LambdaLR(optimizer, lr_lambda)


# -----------------------------
# CHECKPOINTING
# -----------------------------
def _get_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }

def _set_rng_state(rng: dict[str, Any]) -> None:
    if rng.get("python") is not None:
        random.setstate(rng["python"])
    if rng.get("torch") is not None:
        torch.set_rng_state(rng["torch"])
    if torch.cuda.is_available() and rng.get("cuda") is not None:
        torch.cuda.set_rng_state_all(rng["cuda"])