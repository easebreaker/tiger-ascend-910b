#!/usr/bin/env python3
"""Stepwise Ascend NPU probe for XLT TIGER crash isolation.

Run on 910B after sourcing CANN:
  python -u scripts/npu_xlt_probe.py
Prints [probe N] before each risky step — last printed line is where it died.
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    log(f"[probe 0] cwd={os.getcwd()} python={sys.version.split()[0]}")
    log("[probe 1] import torch")
    import torch

    log(f"[probe 1] torch={torch.__version__}")
    log("[probe 2] import torch_npu")
    import torch_npu  # noqa: F401

    log(f"[probe 2] npu_available={torch.npu.is_available()} count={torch.npu.device_count()}")
    assert torch.npu.is_available(), "NPU not available"

    log("[probe 3] tiny tensor H2D + matmul")
    x = torch.randn(8, 128, device="npu:0")
    y = x @ x.T
    torch.npu.synchronize()
    log(f"[probe 3] ok y.shape={tuple(y.shape)}")

    log("[probe 4] build npu_safe T5 (128/4/32)")
    from transformers import T5ForConditionalGeneration

    from tiger_ascend.model.tiger import build_xlt_t5_config

    cfg = build_xlt_t5_config(1025, layout="npu_safe", use_cache=False)
    try:
        model = T5ForConditionalGeneration(cfg, attn_implementation="eager")
    except TypeError:
        model = T5ForConditionalGeneration(cfg)
    if hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = "eager"
    n = sum(p.numel() for p in model.parameters()) / 1e6
    log(f"[probe 4] params={n:.2f}M d_model={cfg.d_model} heads={cfg.num_heads} d_kv={cfg.d_kv}")

    log("[probe 5] model.to(npu)")
    model.to("npu:0")
    torch.npu.synchronize()
    log("[probe 5] ok")

    log("[probe 6] forward+backward batch=2 seq=80")
    model.train()
    input_ids = torch.randint(1, 1024, (2, 80), device="npu:0")
    attention_mask = torch.ones_like(input_ids)
    labels = torch.randint(1, 1024, (2, 4), device="npu:0")
    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = out.loss
    log(f"[probe 6] loss={float(loss.detach().cpu()):.4f} backward...")
    loss.backward()
    torch.npu.synchronize()
    log("[probe 6] ok")

    log("[probe 7] load Beauty subset one batch via XltSeqDataset")
    from torch.utils.data import DataLoader

    from tiger_ascend.data.dataset import load_json
    from tiger_ascend.data.xlt import XltSeqDataset

    data_dir = os.path.join(ROOT, "data", "amazon_beauty", "subset_512u")
    inters = load_json(os.path.join(data_dir, "inter.json"))
    indices = load_json(os.path.join(data_dir, "semantic_ids.json"))
    ds = XltSeqDataset(inters, indices, max_his_len=20, mode="train")
    loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=ds.get_collate_fn(), num_workers=0)
    batch = next(iter(loader))
    batch = {k: v.to("npu:0") for k, v in batch.items()}
    log(f"[probe 7] batch input={tuple(batch['input_ids'].shape)} labels={tuple(batch['labels'].shape)}")
    out = model(**batch)
    out.loss.backward()
    torch.npu.synchronize()
    log("[probe 7] ok")

    log("[probe DONE] all steps passed — XLT train crash is likely later (opt/amp/full data)")


if __name__ == "__main__":
    main()
