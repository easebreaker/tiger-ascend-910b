#!/usr/bin/env python3
"""First-NPU-alloc probe (w2 Abort / set_device skipped).

  export ASCEND_RT_VISIBLE_DEVICES=7   # must be a FREE physical card
  python -u scripts/npu_set_device_probe.py

Cards 1/2 busy do not affect you only if THIS visible id is free.
"""

from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


def log(msg: str) -> None:
    print(msg, flush=True)


def _hint_physical(vis: str) -> None:
    log(f"[s0] logical npu:0 maps to first id in ASCEND_RT_VISIBLE_DEVICES={vis!r}")
    if not vis:
        log("[s0] WARNING: ASCEND_RT_VISIBLE_DEVICES unset — may grab card 0")
        return
    first = vis.split(",")[0].strip()
    log(f"[s0] check THIS physical NPU id is free: {first}")
    try:
        out = subprocess.check_output(["npu-smi", "info"], text=True, stderr=subprocess.STDOUT)
        # print a short slice; full table is noisy
        lines = out.splitlines()
        for i, line in enumerate(lines):
            if first in line or "NPU" in line or "Process" in line or "Name" in line:
                log("  | " + line)
                for j in range(1, 3):
                    if i + j < len(lines):
                        log("  | " + lines[i + j])
    except Exception as e:
        log(f"[s0] npu-smi unavailable: {e!r}")


def main() -> None:
    vis = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    log(f"[s0] ASCEND_RT_VISIBLE_DEVICES={vis!r}")
    log(f"[s0] TIGER_NPU_SET_DEVICE={os.environ.get('TIGER_NPU_SET_DEVICE', '0')!r}")
    _hint_physical(vis)

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
        log("[s5] set_device(0)")
        torch.npu.set_device(0)
        log("[s5] ok")
    else:
        log("[s5] skip set_device")

    log("[s7a] torch.zeros(1).npu()  <<< first ACL context")
    t0 = torch.zeros(1, dtype=torch.float32).npu()
    log(f"[s7a] ok device={t0.device}")

    log("[s7b] synchronize()")
    torch.npu.synchronize()
    log("[s7b] ok")

    log("[s7c] empty(2,2, device='npu:0')")
    t = torch.empty(2, 2, device="npu:0")
    torch.npu.synchronize()
    log(f"[s7c] ok device={t.device}")

    log("[s8] matmul")
    x = torch.randn(8, 8, device="npu:0")
    y = x @ x.T
    torch.npu.synchronize()
    log(f"[s8] ok y={tuple(y.shape)}")

    log("[DONE] first alloc path works")


if __name__ == "__main__":
    main()
