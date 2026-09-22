"""Device helpers for CPU / CUDA / Ascend NPU."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Union

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


def prepare_npu_runtime(device_index: int = 0) -> None:
    """Set device + disable JIT compile paths that often Abort on Ascend .to()."""
    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        raise RuntimeError("NPU unavailable")
    if hasattr(torch.npu, "set_compile_mode"):
        try:
            torch.npu.set_compile_mode(jit_compile=False)
        except Exception:
            pass
    torch.npu.set_device(int(device_index))
    torch.npu.synchronize()


def _device_sync(device: torch.device) -> Optional[Callable[[], None]]:
    if device.type == "npu":
        return torch.npu.synchronize
    if device.type == "cuda":
        return torch.cuda.synchronize
    return None


def move_module_to_device_safe(
    module: torch.nn.Module,
    device: Union[torch.device, str],
    *,
    log: bool = False,
    sync_each: bool = True,
) -> torch.nn.Module:
    """Move params/buffers once each (by object id).

    HuggingFace T5 reuses the same ``shared`` Embedding under
    ``encoder.embed_tokens`` / ``decoder.embed_tokens``. Recursive
    ``Module.to(npu)`` therefore applies H2D twice on one module; some
    torch_npu builds Abort there even when tiny Linear / basic ops work.

    This helper walks unique Parameter/buffer storages only.
    """
    device = torch.device(device)
    sync = _device_sync(device)
    log_fn = print if log else (lambda *_a, **_k: None)

    if device.type == "npu":
        idx = device.index if device.index is not None else 0
        prepare_npu_runtime(idx)

    seen_param: set[int] = set()
    n_moved = 0
    for name, param in module.named_parameters(remove_duplicate=False):
        pid = id(param)
        if pid in seen_param:
            continue
        seen_param.add(pid)
        if param.device == device:
            continue
        log_fn(f"[npu-move] param {name} shape={tuple(param.shape)} dtype={param.dtype}", flush=True)
        with torch.no_grad():
            param.data = param.data.to(device, non_blocking=False).contiguous()
        n_moved += 1
        if sync_each and sync is not None:
            sync()

    seen_buf: set[int] = set()
    for name, buf in module.named_buffers(remove_duplicate=False):
        if not torch.is_tensor(buf):
            continue
        bid = id(buf)
        if bid in seen_buf:
            continue
        seen_buf.add(bid)
        if buf.device == device:
            continue
        log_fn(f"[npu-move] buffer {name} shape={tuple(buf.shape)}", flush=True)
        # Buffers live in module._buffers; replace via named path on parent.
        parts = name.rsplit(".", 1)
        if len(parts) == 1:
            parent, bname = module, parts[0]
        else:
            parent = module.get_submodule(parts[0])
            bname = parts[1]
        parent._buffers[bname] = buf.to(device, non_blocking=False).contiguous()
        n_moved += 1
        if sync_each and sync is not None:
            sync()

    if sync is not None:
        sync()
    log_fn(f"[npu-move] done moved={n_moved} device={device}", flush=True)
    return module


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
