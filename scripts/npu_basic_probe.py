#!/usr/bin/env python3
"""Ultra-fine Ascend NPU smoke (no TIGER). Find which primitive Aborts.

  source <cann>/set_env.sh
  export ASCEND_RT_VISIBLE_DEVICES=0
  export ASCEND_LAUNCH_BLOCKING=1
  python -u scripts/npu_basic_probe.py
"""

from __future__ import annotations

import os
import sys


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    log(f"[b0] pid={os.getpid()} ASCEND_RT_VISIBLE_DEVICES={os.environ.get('ASCEND_RT_VISIBLE_DEVICES')}")
    log(f"[b0] ASCEND_HOME={os.environ.get('ASCEND_HOME') or os.environ.get('ASCEND_TOOLKIT_HOME')}")

    log("[b1] import torch")
    import torch

    log(f"[b1] torch={torch.__version__} cuda={torch.cuda.is_available()}")

    log("[b2] import torch_npu")
    try:
        import torch_npu
    except Exception as e:
        log(f"[b2] FAIL import torch_npu: {e!r}")
        raise
    log(f"[b2] torch_npu={getattr(torch_npu, '__version__', '?')}")

    log("[b3] torch.npu.is_available / device_count")
    ok = torch.npu.is_available()
    n = torch.npu.device_count() if ok else 0
    log(f"[b3] available={ok} count={n}")
    if not ok:
        raise SystemExit("NPU not available — check driver/CANN/set_env.sh")

    log("[b4a] torch.npu.current_device()  (read only)")
    try:
        cur = torch.npu.current_device()
        log(f"[b4a] ok current_device={cur}")
    except Exception as e:
        log(f"[b4a] FAIL {e!r}")

    log("[b4b] torch.npu.set_device(0)")
    torch.npu.set_device(0)
    log("[b4b] ok")

    log("[b4c] torch.npu.synchronize()  <<< empty sync; if Abort here = runtime/driver")
    torch.npu.synchronize()
    log("[b4c] ok")

    log("[b5] empty((1,), device=npu) + sync")
    t = torch.empty((1,), device="npu:0")
    torch.npu.synchronize()
    log(f"[b5] ok dtype={t.dtype} device={t.device}")

    log("[b6] zeros(4) on npu + sync")
    z = torch.zeros(4, device="npu:0")
    torch.npu.synchronize()
    log(f"[b6] ok sum={float(z.cpu().sum())}")

    log("[b7] randn(8,8) H2D-style create on npu + sync (NO matmul)")
    x = torch.randn(8, 8, device="npu:0")
    torch.npu.synchronize()
    log(f"[b7] ok mean={float(x.float().mean().cpu()):.4f}")

    log("[b8] x+x add + sync")
    y = x + x
    torch.npu.synchronize()
    log(f"[b8] ok y[0,0]={float(y[0, 0].cpu()):.4f}")

    log("[b9] matmul 8x8 + sync  <<< previous crash was around here")
    z = x @ x.T
    torch.npu.synchronize()
    log(f"[b9] ok z.shape={tuple(z.shape)}")

    log("[b10] matmul 8x128 @ 128x8")
    a = torch.randn(8, 128, device="npu:0")
    b = torch.randn(128, 8, device="npu:0")
    c = a @ b
    torch.npu.synchronize()
    log(f"[b10] ok c.shape={tuple(c.shape)}")

    log("[DONE] basic NPU ops OK — TIGER abort is elsewhere; re-run npu_xlt_probe.py")


if __name__ == "__main__":
    main()
