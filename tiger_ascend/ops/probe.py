"""Ascend / torch_npu operator probe for TIGER bring-up."""

from __future__ import annotations

import argparse
import json
import traceback
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List

import torch

from tiger_ascend.utils.device import resolve_device, synchronize


@dataclass
class OpResult:
    name: str
    status: str  # ok | fail | skip
    detail: str = ""


REQUIRED_OPS: List[Dict[str, str]] = [
    {"name": "tensor_alloc", "group": "runtime", "why": "allocate activations on device"},
    {"name": "copy_h2d_d2h", "group": "runtime", "why": "host/device transfer"},
    {"name": "embedding", "group": "sparse", "why": "token embedding lookup"},
    {"name": "linear_matmul", "group": "dense", "why": "QKV / FFN / lm_head"},
    {"name": "bmm", "group": "dense", "why": "attention scores / context"},
    {"name": "softmax", "group": "activation", "why": "attention weights"},
    {"name": "gelu_or_relu", "group": "activation", "why": "T5 FFN activation"},
    {"name": "layernorm", "group": "norm", "why": "T5LayerNorm / RMS-style norm"},
    {"name": "dropout", "group": "regularization", "why": "train-time dropout"},
    {"name": "cross_entropy", "group": "loss", "why": "teacher-forcing CE"},
    {"name": "sdpa_or_manual_attn", "group": "attention", "why": "self/cross attention"},
    {"name": "amp_autocast", "group": "amp", "why": "fp16/bf16 training path"},
]

OPTIONAL_OR_ABSENT: List[Dict[str, str]] = [
    {"name": "hstu_paged", "why": "HSTU-only fused op; TIGER does not need it"},
    {"name": "jagged_tensor_ops", "why": "TorchRec jagged path; TIGER uses dense pads"},
    {"name": "fbgemm_npu", "why": "embedding-bag stacks; optional later"},
    {"name": "flash_attention", "why": "speedup only; fallback to standard attn"},
]


def _run(name: str, fn: Callable[[], None], skip: bool = False) -> OpResult:
    if skip:
        return OpResult(name, "skip", "not applicable on this device")
    try:
        fn()
        return OpResult(name, "ok")
    except Exception as exc:  # noqa: BLE001
        return OpResult(name, "fail", f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=1)}")


def probe(device_pref: str = "auto") -> Dict[str, Any]:
    info = resolve_device(device_pref)
    device = info.device
    results: List[OpResult] = []

    def tensor_alloc():
        x = torch.empty((4, 4), device=device)
        assert x.device.type in {"cpu", "cuda", "npu"}

    def copy_h2d_d2h():
        h = torch.randn(8, 8)
        d = h.to(device)
        back = d.to("cpu")
        assert torch.allclose(h, back, atol=1e-5)

    def embedding():
        emb = torch.nn.Embedding(32, 16).to(device)
        idx = torch.randint(0, 32, (2, 5), device=device)
        y = emb(idx)
        assert y.shape == (2, 5, 16)

    def linear_matmul():
        lin = torch.nn.Linear(16, 32).to(device)
        x = torch.randn(2, 4, 16, device=device)
        y = lin(x)
        assert y.shape == (2, 4, 32)

    def bmm():
        a = torch.randn(2, 4, 8, device=device)
        b = torch.randn(2, 8, 4, device=device)
        y = torch.bmm(a, b)
        assert y.shape == (2, 4, 4)

    def softmax():
        x = torch.randn(2, 4, 8, device=device)
        y = torch.softmax(x, dim=-1)
        assert torch.allclose(y.sum(-1), torch.ones_like(y.sum(-1)), atol=1e-4)

    def gelu_or_relu():
        x = torch.randn(4, 8, device=device)
        y = torch.nn.functional.gelu(x)
        z = torch.nn.functional.relu(x)
        assert y.shape == z.shape == x.shape

    def layernorm():
        x = torch.randn(2, 4, 16, device=device)
        ln = torch.nn.LayerNorm(16).to(device)
        y = ln(x)
        assert y.shape == x.shape

    def dropout():
        m = torch.nn.Dropout(0.1).to(device)
        x = torch.randn(8, 8, device=device)
        y = m(x)
        assert y.shape == x.shape

    def cross_entropy():
        logits = torch.randn(6, 10, device=device, requires_grad=True)
        target = torch.randint(0, 10, (6,), device=device)
        loss = torch.nn.functional.cross_entropy(logits, target)
        loss.backward()

    def sdpa_or_manual_attn():
        b, h, s, d = 1, 2, 8, 16
        q = torch.randn(b, h, s, d, device=device)
        k = torch.randn(b, h, s, d, device=device)
        v = torch.randn(b, h, s, d, device=device)
        try:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        except Exception:
            scores = torch.matmul(q, k.transpose(-2, -1)) / (d**0.5)
            probs = torch.softmax(scores, dim=-1)
            y = torch.matmul(probs, v)
        assert y.shape == (b, h, s, d)

    def amp_autocast():
        x = torch.randn(4, 16, device=device)
        lin = torch.nn.Linear(16, 16).to(device)
        with torch.autocast(device_type=info.kind, dtype=torch.float16):
            y = lin(x)
        assert y is not None

    probes = [
        ("tensor_alloc", tensor_alloc, False),
        ("copy_h2d_d2h", copy_h2d_d2h, False),
        ("embedding", embedding, False),
        ("linear_matmul", linear_matmul, False),
        ("bmm", bmm, False),
        ("softmax", softmax, False),
        ("gelu_or_relu", gelu_or_relu, False),
        ("layernorm", layernorm, False),
        ("dropout", dropout, False),
        ("cross_entropy", cross_entropy, False),
        ("sdpa_or_manual_attn", sdpa_or_manual_attn, False),
        ("amp_autocast", amp_autocast, info.kind == "cpu"),
    ]

    for name, fn, skip in probes:
        results.append(_run(name, fn, skip=skip))

    synchronize(info)
    hard_fails = [r for r in results if r.status == "fail"]
    return {
        "device": {
            "kind": info.kind,
            "device": str(info.device),
            "device_count": info.device_count,
            "backend": info.backend,
            "torch_npu_imported": info.torch_npu_imported,
        },
        "required_ops_catalog": REQUIRED_OPS,
        "not_required_for_tiger": OPTIONAL_OR_ABSENT,
        "results": [asdict(r) for r in results],
        "ok": len(hard_fails) == 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Probe TIGER-required ops on Ascend/CUDA/CPU")
    parser.add_argument("--device", default="auto", choices=["auto", "npu", "cuda", "cpu"])
    parser.add_argument("--out", default="", help="optional JSON report path")
    args = parser.parse_args()
    report = probe(args.device)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
