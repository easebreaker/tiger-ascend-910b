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


def warmup_npu(device_index: int = 0) -> None:
    """Match npu_model_to_probe m0/m1 before large T5 H2D.

    Some torch_npu builds Abort on the first multi-megabyte transfer if the
    runtime never allocated a small tensor / Linear first.
    """
    prepare_npu_runtime(device_index)
    dev = torch.device(f"npu:{int(device_index)}")
    x = torch.randn(8, 32, device=dev)
    y = x @ x.T
    torch.npu.synchronize()
    lin = torch.nn.Linear(32, 32)
    # Single small Module.to is OK (no tied Embedding graph).
    lin.to(dev)
    torch.npu.synchronize()
    _ = lin(x)
    torch.npu.synchronize()
    del x, y, lin
    if hasattr(torch.npu, "empty_cache"):
        torch.npu.empty_cache()


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
    progress_every: int = 1,
    strategy: str = "copy_",
) -> torch.nn.Module:
    """Move params/buffers once each (by object id).

    HuggingFace T5 reuses the same ``shared`` Embedding under
    ``encoder.embed_tokens`` / ``decoder.embed_tokens``. Recursive
    ``Module.to(npu)`` therefore applies H2D twice on one module; some
    torch_npu builds Abort there even when tiny Linear / basic ops work.

    strategy:
      - ``copy_``: empty_like on device then copy_ (often stabler on Ascend)
      - ``to``: ``param.data = param.data.to(device)``
    """
    device = torch.device(device)
    sync = _device_sync(device)
    log_fn = print if log else (lambda *_a, **_k: None)
    if strategy not in {"copy_", "to"}:
        raise ValueError(f"unknown strategy={strategy}")

    if device.type == "npu":
        idx = device.index if device.index is not None else 0
        prepare_npu_runtime(idx)

    def _h2d(src: torch.Tensor) -> torch.Tensor:
        if strategy == "copy_":
            dst = torch.empty(src.shape, dtype=src.dtype, device=device)
            dst.copy_(src, non_blocking=False)
            return dst
        return src.to(device, non_blocking=False)

    unique_params = list(module.named_parameters(remove_duplicate=True))
    unique_bufs = [
        (n, b)
        for n, b in module.named_buffers(remove_duplicate=True)
        if torch.is_tensor(b)
    ]
    log_fn(
        f"[npu-move] start strategy={strategy} params={len(unique_params)} "
        f"buffers={len(unique_bufs)} device={device}",
        flush=True,
    )

    n_moved = 0
    for name, param in unique_params:
        if param.device == device:
            continue
        n_moved += 1
        if log and (progress_every <= 1 or n_moved % progress_every == 1 or n_moved <= 3):
            log_fn(
                f"[npu-move] ({n_moved}/{len(unique_params)}) param {name} "
                f"shape={tuple(param.shape)}",
                flush=True,
            )
        with torch.no_grad():
            param.data = _h2d(param.data)
        if sync_each and sync is not None:
            sync()

    n_buf = 0
    for name, buf in unique_bufs:
        if buf.device == device:
            continue
        n_buf += 1
        if log:
            log_fn(f"[npu-move] buffer {name} shape={tuple(buf.shape)}", flush=True)
        parts = name.rsplit(".", 1)
        if len(parts) == 1:
            parent, bname = module, parts[0]
        else:
            parent = module.get_submodule(parts[0])
            bname = parts[1]
        parent._buffers[bname] = _h2d(buf)
        if sync_each and sync is not None:
            sync()

    if sync is not None:
        sync()
    log_fn(
        f"[npu-move] done moved_params={n_moved} moved_buffers={n_buf} device={device}",
        flush=True,
    )
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
