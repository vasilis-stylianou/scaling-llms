"""Single-node multi-GPU DDP utilities."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def is_distributed() -> bool:
    """True when a DDP process group has been initialised."""
    return dist.is_available() and dist.is_initialized()


def get_global_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return get_global_rank() == 0


def barrier_if_distributed() -> None:
    """Synchronise all ranks. No-op when not distributed."""
    if is_distributed():
        dist.barrier()


def ddp_setup(backend: str) -> None:
    """Initialise the process group when launched by torchrun."""
    if not dist.is_available():
        return
    if "RANK" not in os.environ:
        return
    if dist.is_initialized():
        return

    dist.init_process_group(backend=backend, init_method="env://")

    if backend == "nccl" and torch.cuda.is_available():
        torch.cuda.set_device(get_local_rank())
    # if not dist.is_available():
    #     return
    # # torchrun sets RANK; if absent we are not in a DDP launch
    # if "RANK" not in os.environ:
    #     return
    # if dist.is_initialized():
    #     return
    # dist.init_process_group(backend=backend)
    # torch.cuda.set_device(get_local_rank())


def ddp_cleanup() -> None:
    if is_distributed():
        dist.destroy_process_group()


def is_ddp_model(model) -> bool:
    # unwrap torch.compile's OptimizedModule if present
    inner = getattr(model, "_orig_mod", model)
    return isinstance(inner, DDP)


def apply_all_reduce_if_distributed(tensor: torch.Tensor) -> torch.Tensor:
    """Helper to apply all-reduce to a tensor across DDP ranks, returning the reduced value on each rank."""
    if is_distributed():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor