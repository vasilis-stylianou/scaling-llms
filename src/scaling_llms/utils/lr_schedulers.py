import math
from enum import StrEnum
from typing import Callable

from torch.optim.lr_scheduler import LambdaLR


class LRSchedule(StrEnum):
    none = "none"
    linear = "linear"
    cosine = "cosine"
    trapezoidal = "trapezoidal"


def _validate_common(num_steps: int, warmup_steps: int, min_lr_ratio: float) -> None:
    if num_steps <= 0:
        raise ValueError(f"num_steps must be > 0; got {num_steps}")
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be >= 0; got {warmup_steps}")
    if warmup_steps > num_steps:
        raise ValueError(
            f"warmup_steps ({warmup_steps}) must be <= num_steps ({num_steps})"
        )
    if not (0.0 <= min_lr_ratio <= 1.0):
        raise ValueError(f"min_lr_ratio must be in [0, 1]; got {min_lr_ratio}")


def _warmup_multiplier(step: int, warmup_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    return 1.0


def _post_warmup_progress(step: int, warmup_steps: int, num_steps: int) -> float:
    denom = max(1, num_steps - warmup_steps)
    progress = float(step - warmup_steps + 1) / float(denom)
    return min(max(progress, 0.0), 1.0)


def make_linear_lr_lambda(
    num_steps: int,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.0,
) -> Callable[[int], float]:
    _validate_common(num_steps, warmup_steps, min_lr_ratio)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return _warmup_multiplier(step, warmup_steps)
        progress = _post_warmup_progress(step, warmup_steps, num_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 - progress)

    return lr_lambda


def make_cosine_lr_lambda(
    num_steps: int,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.0,
) -> Callable[[int], float]:
    _validate_common(num_steps, warmup_steps, min_lr_ratio)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return _warmup_multiplier(step, warmup_steps)
        progress = _post_warmup_progress(step, warmup_steps, num_steps)
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (
            1.0 + math.cos(math.pi * progress)
        )

    return lr_lambda


def make_trapezoidal_lr_lambda(
    num_steps: int,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.0,
    decay_fraction: float = 0.0,
) -> Callable[[int], float]:
    _validate_common(num_steps, warmup_steps, min_lr_ratio)

    decay_fraction = float(decay_fraction)
    if not (0.0 < decay_fraction <= 1.0):
        raise ValueError(
            f"decay_fraction must be in (0, 1] for trapezoidal; got {decay_fraction}"
        )

    decay_steps = int(round(decay_fraction * num_steps))
    if warmup_steps + decay_steps > num_steps:
        raise ValueError(
            f"warmup_steps ({warmup_steps}) + decay_steps ({decay_steps}) "
            f"exceeds num_steps ({num_steps})"
        )
    stable_end = num_steps - decay_steps

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return _warmup_multiplier(step, warmup_steps)

        if step < stable_end:
            return 1.0

        progress = float(step - stable_end + 1) / float(max(1, decay_steps))
        progress = min(max(progress, 0.0), 1.0)
        return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 - progress)

    return lr_lambda


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
    num_steps = int(num_steps)
    min_lr_ratio = float(min_lr_ratio)

    if lr_schedule == LRSchedule.linear:
        lr_lambda = make_linear_lr_lambda(
            num_steps=num_steps,
            warmup_steps=warmup_steps,
            min_lr_ratio=min_lr_ratio,
        )
    elif lr_schedule == LRSchedule.cosine:
        lr_lambda = make_cosine_lr_lambda(
            num_steps=num_steps,
            warmup_steps=warmup_steps,
            min_lr_ratio=min_lr_ratio,
        )
    elif lr_schedule == LRSchedule.trapezoidal:
        lr_lambda = make_trapezoidal_lr_lambda(
            num_steps=num_steps,
            warmup_steps=warmup_steps,
            min_lr_ratio=min_lr_ratio,
            decay_fraction=decay_fraction,
        )
    else:
        raise ValueError(f"Unsupported lr_schedule: {lr_schedule}")

    return LambdaLR(optimizer, lr_lambda)
