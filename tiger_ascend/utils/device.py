"""Device helpers for CPU / CUDA / Ascend NPU."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class DeviceInfo:
    kind: str  # cpu | cuda | npu
    device: torch.device
    device_count: int  # number of visible accelerators
    backend: Optional[str] = None
    torch_npu_imported: bool = False
    notes: str = ""


def try_import_torch_npu() -> bool:
    try:
        import torch_npu  # noqa: F401

        return True
    except Exception:
        return False


def resolve_device(preferred: str = "auto") -> DeviceInfo:
    preferred = (preferred or "auto").lower()
    npu_ok = try_import_torch_npu() and hasattr(torch, "npu") and torch.npu.is_available()
    cuda_ok = torch.cuda.is_available()

    if preferred == "npu":
        if not npu_ok:
            raise RuntimeError(
                "preferred=npu but Ascend NPU is unavailable. "
                "Install matching torch-npu + CANN, then source set_env.sh."
            )
        return DeviceInfo("npu", torch.device("npu:0"), torch.npu.device_count(), "hccl", True)
    if preferred == "cuda":
        if not cuda_ok:
            raise RuntimeError("preferred=cuda but CUDA is unavailable.")
        return DeviceInfo("cuda", torch.device("cuda:0"), torch.cuda.device_count(), "nccl", npu_ok)
    if preferred == "cpu":
        return DeviceInfo("cpu", torch.device("cpu"), 0, "gloo", npu_ok)

    if npu_ok:
        return DeviceInfo("npu", torch.device("npu:0"), torch.npu.device_count(), "hccl", True)
    if cuda_ok:
        return DeviceInfo("cuda", torch.device("cuda:0"), torch.cuda.device_count(), "nccl", npu_ok)
    return DeviceInfo("cpu", torch.device("cpu"), 0, "gloo", npu_ok)


def synchronize(info: DeviceInfo) -> None:
    if info.kind == "npu":
        torch.npu.synchronize()
    elif info.kind == "cuda":
        torch.cuda.synchronize()


def amp_device_type(info: DeviceInfo) -> str:
    return info.kind if info.kind in {"cuda", "npu"} else "cpu"


def dataloader_kwargs(
    info: DeviceInfo,
    num_workers: int = 0,
    prefetch_factor: int = 2,
) -> dict:
    """DataLoader kwargs tuned for accelerator host/device overlap.

    Default ``num_workers=0`` is safest for smoke tests; pass ``>=2`` for real runs.
    ``pin_memory`` helps CUDA; Ascend often prefers ``False`` (H2D path differs).
    """
    kwargs = {
        "num_workers": max(0, int(num_workers)),
        "pin_memory": info.kind == "cuda",
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = max(2, int(prefetch_factor))
    return kwargs


def move_batch_to_device(batch: dict, device: torch.device, non_blocking: bool = True) -> dict:
    return {
        k: (v.to(device, non_blocking=non_blocking) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }
