#!/usr/bin/env python3
"""Probe set_device vs alloc-without-set_device (p2 Abort case).

  export ASCEND_RT_VISIBLE_DEVICES=7   # physical → logical npu:0
  python -u scripts/npu_set_device_probe.py

Default path skips set_device (matches train). Optional:
  TIGER_NPU_SET_DEVICE=1 python -u scripts/npu_set_device_probe.py
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    log(f"[s0] ASCEND_RT_VISIBLE_DEVICES={os.environ.get('ASCEND_RT_VISIBLE_DEVICES')!r}")
    log(f"[s0] TIGER_NPU_SET_DEVICE={os.environ.get('TIGER_NPU_SET_DEVICE', '0')!r}")
    log("[s1] import torch")
    import torch

    log("[s2] import torch_npu")
    import torch_npu  # noqa: F401

    log(f"[s3] available={torch.npu.is_available()} count={torch.npu.device_count()}")
    assert torch.npu.is_available() and torch.npu.device_count() >= 1

    log("[s4] set_compile_mode(False)")
    if hasattr(torch.npu, "set_compile_mode"):
        try:
            torch.npu.set_compile_mode(jit_compile=False)
            log("[s4] ok")
        except Exception as e:
            log(f"[s4] skip {e!r}")

    if os.environ.get("TIGER_NPU_SET_DEVICE", "0").strip() in {"1", "true", "True"}:
        log("[s5] set_device(0)  <<< known Abort site on some boxes")
        torch.npu.set_device(0)
        log("[s5] ok")
        log("[s6] synchronize()")
        torch.npu.synchronize()
        log("[s6] ok")
    else:
        log("[s5] skip set_device (default — train uses this path)")

    log("[s7] empty(2,2) on npu:0  (no prior set_device)")
    t = torch.empty(2, 2, device="npu:0")
    log("[s7a] synchronize after alloc")
    torch.npu.synchronize()
    log(f"[s7] ok device={t.device}")

    log("[s8] matmul")
    x = torch.randn(8, 8, device="npu:0")
    y = x @ x.T
    torch.npu.synchronize()
    log(f"[s8] ok y={tuple(y.shape)}")

    log("[DONE] alloc-without-set_device works")


if __name__ == "__main__":
    main()
