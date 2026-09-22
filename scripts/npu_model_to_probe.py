#!/usr/bin/env python3
"""Probe which part of T5 model.to(npu) Aborts (basic probe must already pass).

  source CANN set_env + driver LD_LIBRARY_PATH fix
  python -u scripts/npu_model_to_probe.py
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    import torch
    import torch_npu  # noqa: F401
    from transformers import T5Config, T5ForConditionalGeneration

    from tiger_ascend.model.tiger import build_xlt_t5_config

    assert torch.npu.is_available()
    log("[m0] set_device(0)")
    torch.npu.set_device(0)
    torch.npu.synchronize()
    log("[m0] ok")

    log("[m1] tiny Linear.to(npu)")
    lin = torch.nn.Linear(32, 32)
    lin.to("npu:0")
    torch.npu.synchronize()
    y = lin(torch.randn(2, 32, device="npu:0"))
    torch.npu.synchronize()
    log(f"[m1] ok y={tuple(y.shape)}")

    log("[m2] 1-layer tiny T5 .to(npu) then forward")
    tiny = T5Config(
        vocab_size=128,
        d_model=64,
        d_ff=128,
        d_kv=16,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=4,
        dropout_rate=0.0,
        feed_forward_proj="relu",
        is_encoder_decoder=True,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
        use_cache=False,
    )
    m2 = T5ForConditionalGeneration(tiny)
    log("[m2] built on CPU, moving...")
    m2.to("npu:0")
    torch.npu.synchronize()
    log("[m2] on npu, forward...")
    ids = torch.randint(2, 50, (2, 16), device="npu:0")
    lab = torch.randint(2, 50, (2, 4), device="npu:0")
    loss = m2(input_ids=ids, attention_mask=torch.ones_like(ids), labels=lab).loss
    loss.backward()
    torch.npu.synchronize()
    log(f"[m2] ok loss={float(loss.detach().cpu()):.4f}")

    log("[m3] XLT npu_safe T5 on CPU")
    cfg = build_xlt_t5_config(1025, layout="npu_safe", use_cache=False)
    try:
        m3 = T5ForConditionalGeneration(cfg, attn_implementation="eager")
    except TypeError:
        m3 = T5ForConditionalGeneration(cfg)
    if hasattr(m3.config, "_attn_implementation"):
        m3.config._attn_implementation = "eager"
    n = sum(p.numel() for p in m3.parameters()) / 1e6
    log(f"[m3] params={n:.2f}M — moving whole model.to(npu) ...")

    m3.to("npu:0")
    torch.npu.synchronize()
    log("[m3] whole model on npu OK")

    log("[m4] submodule-wise move (if m3 aborted, re-run; else skip detail)")
    # Rebuild on CPU and move child-by-child for diagnosis when needed
    try:
        m4 = T5ForConditionalGeneration(cfg)
        torch.npu.set_device(0)
        for name, child in m4.named_children():
            log(f"[m4] moving child={name} ...")
            child.to("npu:0")
            torch.npu.synchronize()
            log(f"[m4] child={name} ok")
        log("[m4] all children ok")
    except Exception as e:
        log(f"[m4] FAIL {e!r}")

    log("[DONE] model.to probes finished")


if __name__ == "__main__":
    main()
