#!/usr/bin/env python3
"""Diagnose first ACL context failure (dies at zeros(1).npu()).

This is an Ascend runtime / device access issue, not TIGER model code.
If s7a Aborts, training cannot proceed on this visible card.

  export ASCEND_RT_VISIBLE_DEVICES=<free_physical_id>
  # 修法2 driver libs must be first in LD_LIBRARY_PATH
  python -u scripts/npu_acl_context_probe.py
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    vis = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    log(f"[a0] ASCEND_RT_VISIBLE_DEVICES={vis!r}")
    log(f"[a0] ASCEND_LAUNCH_BLOCKING={os.environ.get('ASCEND_LAUNCH_BLOCKING')!r}")
    log(f"[a0] ASCEND_DEVICE_ID={os.environ.get('ASCEND_DEVICE_ID')!r}")
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    head = ":".join(ld.split(":")[:5]) if ld else ""
    log(f"[a0] LD_LIBRARY_PATH head={head}")
    if "driver/lib64" not in ld:
        log(
            "[a0] WARNING: LD_LIBRARY_PATH missing Ascend driver/lib64 — "
            "re-apply 修法2 after set_env.sh"
        )

    log("[a1] /dev/davinci* nodes visible to this process")
    nodes = sorted(glob.glob("/dev/davinci*"))
    log(f"[a1] {nodes if nodes else '(none — container may not mount NPUs)'}")
    if vis:
        for part in vis.split(","):
            part = part.strip()
            if not part.isdigit():
                continue
            path = f"/dev/davinci{part}"
            ok = os.path.exists(path)
            log(f"[a1] {path} exists={ok}")

    log("[a2] npu-smi info (look at YOUR visible physical id)")
    try:
        out = subprocess.check_output(["npu-smi", "info"], text=True, stderr=subprocess.STDOUT)
        for line in out.splitlines()[:80]:
            log("  | " + line)
    except Exception as e:
        log(f"[a2] npu-smi failed: {e!r}")

    log("[a3] import torch + torch_npu")
    import torch
    import torch_npu  # noqa: F401

    log(
        f"[a3] torch={torch.__version__} torch_npu={getattr(torch_npu, '__version__', '?')} "
        f"available={torch.npu.is_available()} count={torch.npu.device_count()}"
    )
    if not torch.npu.is_available() or torch.npu.device_count() < 1:
        raise SystemExit("NPU not visible to torch — fix set_env / VISIBLE_DEVICES / device mount")

    if hasattr(torch.npu, "set_compile_mode"):
        try:
            torch.npu.set_compile_mode(jit_compile=False)
        except Exception:
            pass

    log("[a4] about to create first ACL context via zeros(1).npu()")
    log("[a4] if process Aborts here: physical card unusable / busy / bad mount / bad libs")
    t = torch.zeros(1, dtype=torch.float32).npu()
    log(f"[a4] SUCCESS device={t.device}")
    torch.npu.synchronize()
    log("[DONE] ACL context OK — retry smoke")


if __name__ == "__main__":
    main()
