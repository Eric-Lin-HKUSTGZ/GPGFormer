from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistInfo:
    distributed: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device


def _get_env_int(name: str, default: int) -> int:
    v = os.environ.get(name, None)
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default


def setup_distributed(backend: str | None = None) -> DistInfo:
    """
    Setup torch.distributed using env vars set by `torchrun`.
    Mirrors UniHandFormer behavior:
    - If RANK/WORLD_SIZE are missing -> single process.
    - Bind each process to its LOCAL_RANK GPU.
    """
    rank = _get_env_int("RANK", -1)
    local_rank = _get_env_int("LOCAL_RANK", -1)
    world_size = _get_env_int("WORLD_SIZE", -1)

    if rank == -1 or world_size == -1:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return DistInfo(False, 0, 0, 1, device)

    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    # init_method defaults to env:// when using torchrun
    dist.init_process_group(backend=backend)

    if torch.cuda.is_available():
        if local_rank < 0:
            # torchrun always sets LOCAL_RANK; keep safe fallback
            local_rank = rank % max(torch.cuda.device_count(), 1)
        torch.cuda.set_device(local_rank)
        torch.cuda.empty_cache()
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
        local_rank = 0

    return DistInfo(True, rank, local_rank, world_size, device)


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    return (not (dist.is_available() and dist.is_initialized())) or dist.get_rank() == 0


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def seed_everything(seed: int, rank: int = 0) -> None:
    s = int(seed) + int(rank)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def all_reduce_sum(t: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t





