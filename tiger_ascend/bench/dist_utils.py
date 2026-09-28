"""Distributed helpers that avoid ``torch.npu.set_device`` (Abort on some boxes)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class DistInfo:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    backend: str
    device: torch.device


def _pick_backend(device_kind: str) -> str:
    if device_kind == "npu":
        return "hccl"
    if device_kind == "cuda":
        return "nccl"
    return "gloo"


def init_dist(device_kind: str = "npu") -> DistInfo:
    """Initialize process group if launched under torchrun / torch.distributed.

    Uses ``device=npu:{LOCAL_RANK}`` without calling ``set_device``.
    """
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    if world <= 1:
        if device_kind == "npu":
            return DistInfo(False, 0, 0, 1, "none", torch.device("npu:0"))
        if device_kind == "cuda":
            return DistInfo(False, 0, 0, 1, "none", torch.device("cuda:0"))
        return DistInfo(False, 0, 0, 1, "none", torch.device("cpu"))

    backend = _pick_backend(device_kind)
    if not torch.distributed.is_initialized():
        # Ascend: do NOT call set_device — allocate with explicit device index.
        torch.distributed.init_process_group(backend=backend)
    if device_kind == "npu":
        device = torch.device(f"npu:{local_rank}")
    elif device_kind == "cuda":
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    return DistInfo(True, rank, local_rank, world, backend, device)


def barrier(dist: DistInfo) -> None:
    if dist.enabled and torch.distributed.is_initialized():
        torch.distributed.barrier()


def cleanup_dist(dist: DistInfo) -> None:
    if dist.enabled and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def is_main(dist: DistInfo) -> bool:
    return dist.rank == 0
