#!/usr/bin/env python3
"""Minimal set_device probe when smoke dies right after npu-preflight line.

  export ASCEND_RT_VISIBLE_DEVICES=7   # physical card; torch sees npu:0
  python -u scripts/npu_set_device_probe.py
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

    log("[s5] set_device(0)  # logical 0 = first visible physical card")
    torch.npu.set_device(0)
    log("[s5] ok")

    log("[s6] synchronize()")
    torch.npu.synchronize()
    log("[s6] ok")

    log("[s7] empty(2,2) on npu:0")
    t = torch.empty(2, 2, device="npu:0")
    torch.npu.synchronize()
    log(f"[s7] ok device={t.device}")

    log("[DONE] set_device path works on this visible card")


if __name__ == "__main__":
    main()
