#!/usr/bin/env python3
"""Multi-chip scheduling benchmark for TIGER-XLT on Ascend 910B.

Strategies (``--strategy``):
  single     1-card baseline
  dp         DDP data parallel (torchrun --nproc_per_node=N)
  overalloc  N processes, only rank0 trains (card-time waste)
  tp|pp|ep|sp  unsupported boundary → writes JSON reason and exits 0

Examples:
  # baseline
  python -u scripts/sched_bench.py --strategy single --device npu --steps 30

  # DP 2 cards (visible devices must match)
  ASCEND_RT_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 \\
    scripts/sched_bench.py --strategy dp --device npu --steps 30 --global_batch_mode fixed

  # over-alloc 4 cards
  ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 \\
    scripts/sched_bench.py --strategy overalloc --device npu --steps 30
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

import torch
from torch.utils.data import DataLoader, DistributedSampler

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tiger_ascend.bench.dist_utils import (  # noqa: E402
    barrier,
    cleanup_dist,
    init_dist,
    is_main,
)
from tiger_ascend.bench.metrics import BenchReport, MetricsCollector  # noqa: E402
from tiger_ascend.bench.strategies import (  # noqa: E402
    BOUNDARY_UNSUPPORTED,
    STRATEGIES,
    SUPPORTED_RUNNABLE,
)
from tiger_ascend.data.dataset import load_json  # noqa: E402
from tiger_ascend.data.xlt import XLT_VOCAB_SIZE, XltSeqDataset  # noqa: E402
from tiger_ascend.model.tiger import build_xlt_t5_config  # noqa: E402
from tiger_ascend.utils.device import (  # noqa: E402
    move_batch_to_device,
    move_module_to_device_safe,
    prepare_npu_runtime,
    warmup_npu,
)


def parse_args():
    p = argparse.ArgumentParser(description="TIGER multi-chip scheduling bench")
    p.add_argument("--strategy", choices=list(STRATEGIES.keys()), required=True)
    p.add_argument("--device", choices=["npu", "cuda", "cpu"], default="npu")
    p.add_argument("--data_dir", default=os.path.join(ROOT, "data", "amazon_beauty", "subset_512u"))
    p.add_argument("--out_dir", default=os.path.join(ROOT, "artifacts", "sched_bench"))
    p.add_argument("--steps", type=int, default=50, help="optimizer steps to measure")
    p.add_argument("--warmup_steps", type=int, default=5, help="steps excluded from percentiles")
    p.add_argument("--batch_size", type=int, default=32, help="per-rank micro batch")
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument(
        "--global_batch_mode",
        choices=["fixed", "scale"],
        default="fixed",
        help="fixed: keep global batch≈const (strong scale); scale: per-rank batch fixed (weak)",
    )
    p.add_argument("--target_global_batch", type=int, default=256)
    p.add_argument("--layout", choices=["npu_safe", "xlt", "npu_large"], default="npu_safe")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--baseline_json", default="", help="single-card report for speedup")
    p.add_argument("--tag", default="", help="optional run tag in filename")
    return p.parse_args()


def _write_boundary_report(args, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"boundary_{args.strategy}.json")
    payload = {
        "strategy": args.strategy,
        "supported": False,
        "reason": STRATEGIES[args.strategy],
        "recommendation": (
            "Use --strategy single|dp|overalloc. "
            "TP/PP/EP/SP are negative boundaries for ~5M T5 + short SID sequences."
        ),
        "workload": {
            "model": "T5 XLT npu_safe ~4-5M params",
            "seq": "history<=20 items × 4 codes",
            "task": "generative recommendation training step",
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[sched-bench] BOUNDARY strategy={args.strategy} → {path}", flush=True)
    return path


def _resolve_batch(args, world_size: int) -> tuple[int, int, int]:
    """Return per_rank_batch, grad_accum, global_batch."""
    if args.strategy in {"single", "overalloc"} or world_size <= 1:
        gb = args.batch_size * args.grad_accum
        return args.batch_size, args.grad_accum, gb

    if args.global_batch_mode == "scale":
        gb = args.batch_size * args.grad_accum * world_size
        return args.batch_size, args.grad_accum, gb

    # fixed global batch (strong scaling): shrink per-rank batch
    target = max(world_size, args.target_global_batch)
    per = max(1, target // (world_size * max(1, args.grad_accum)))
    gb = per * args.grad_accum * world_size
    return per, args.grad_accum, gb


def _build_model(layout: str, device: torch.device):
    from transformers import T5ForConditionalGeneration

    cfg = build_xlt_t5_config(XLT_VOCAB_SIZE, dropout_rate=0.1, layout=layout, use_cache=False)
    try:
        model = T5ForConditionalGeneration(cfg, attn_implementation="eager")
    except TypeError:
        model = T5ForConditionalGeneration(cfg)
    if hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = "eager"
    if device.type == "npu":
        prepare_npu_runtime(device.index or 0, log=False)
        # Warmup only on this device index via explicit alloc
        _ = torch.zeros(1, device=device)
        if hasattr(torch, "npu"):
            torch.npu.synchronize()
        model = move_module_to_device_safe(
            model, device, log=False, sync_each=False, strategy="copy_"
        )
    else:
        model.to(device)
    return model


def _maybe_ddp(model, dist, strategy: str):
    if strategy != "dp" or not dist.enabled or dist.world_size <= 1:
        return model
    return torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=None,  # device already set on params
        find_unused_parameters=False,
    )


def run_training(args) -> Optional[BenchReport]:
    dist = init_dist(args.device)
    if args.device == "npu" and dist.rank == 0:
        # optional host-side warmup on logical 0 when single-proc
        if not dist.enabled:
            try:
                warmup_npu(0, log=is_main(dist))
            except Exception as e:
                print(f"[sched-bench] warmup warn: {e!r}", flush=True)

    per_batch, grad_accum, global_batch = _resolve_batch(args, dist.world_size)
    metrics = MetricsCollector(dist.device)

    with metrics.phase("data"):
        inters = load_json(os.path.join(args.data_dir, "inter.json"))
        indices = load_json(os.path.join(args.data_dir, "semantic_ids.json"))
        ds = XltSeqDataset(inters, indices, max_his_len=20, mode="train")

    active_trainer = True
    if args.strategy == "overalloc":
        active_trainer = dist.rank == 0

    sampler = None
    if args.strategy == "dp" and dist.enabled and dist.world_size > 1:
        sampler = DistributedSampler(ds, num_replicas=dist.world_size, rank=dist.rank, shuffle=True)

    loader = DataLoader(
        ds,
        batch_size=per_batch,
        shuffle=(sampler is None and active_trainer),
        sampler=sampler,
        collate_fn=ds.get_collate_fn(),
        num_workers=args.num_workers,
        drop_last=True,
    )

    model = None
    optim = None
    if active_trainer:
        with metrics.phase("init_model"):
            torch.manual_seed(args.seed + dist.rank)
            model = _build_model(args.layout, dist.device)
            model = _maybe_ddp(model, dist, args.strategy)
            optim = torch.optim.Adam(model.parameters(), lr=1e-4)
            model.train()

    barrier(dist)

    steps_done = 0
    samples_seen = 0
    it = iter(loader) if active_trainer else None

    with metrics.phase("train"):
        for step_i in range(1, args.steps + 1):
            t0 = time.perf_counter()
            if active_trainer:
                try:
                    batch = next(it)
                except StopIteration:
                    if sampler is not None:
                        sampler.set_epoch(step_i)
                    it = iter(loader)
                    batch = next(it)
                batch = move_batch_to_device(batch, dist.device, non_blocking=False)
                loss = model(**batch).loss / max(1, grad_accum)
                loss.backward()
                # One measured step = one optimizer update (grad_accum microbatches inside)
                for _ in range(max(0, grad_accum - 1)):
                    try:
                        batch2 = next(it)
                    except StopIteration:
                        it = iter(loader)
                        batch2 = next(it)
                    batch2 = move_batch_to_device(batch2, dist.device, non_blocking=False)
                    (model(**batch2).loss / grad_accum).backward()
                optim.step()
                optim.zero_grad(set_to_none=True)
                if dist.device.type == "npu":
                    torch.npu.synchronize()
                elif dist.device.type == "cuda":
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                if step_i > args.warmup_steps:
                    metrics.record_step(dt)
                samples_seen += per_batch * grad_accum
                if is_main(dist) and (step_i % 10 == 0 or step_i == args.steps):
                    print(
                        f"[sched-bench] strategy={args.strategy} "
                        f"step={step_i}/{args.steps} "
                        f"loss={float(loss.detach()) * grad_accum:.4f}",
                        flush=True,
                    )
            else:
                # Held card, no compute — still sync so world stays aligned.
                if dist.device.type == "npu":
                    time.sleep(0.0)
                if dist.device.type == "npu" and hasattr(torch, "npu"):
                    try:
                        torch.npu.synchronize()
                    except Exception:
                        pass
            barrier(dist)
            steps_done = step_i

    barrier(dist)

    report = None
    if is_main(dist):
        # For overalloc, nproc is world_size (cards held); trainer is 1.
        nproc_held = dist.world_size
        notes = ""
        if args.strategy == "overalloc":
            notes = (
                f"overalloc: held_cards={nproc_held} active_trainers=1; "
                f"card_seconds reflects reserved cards"
            )
        report = metrics.build_report(
            strategy=args.strategy,
            nproc=nproc_held,
            world_size=dist.world_size,
            rank=dist.rank,
            device=str(dist.device),
            backend=dist.backend,
            steps=args.steps,
            batch_size=per_batch,
            grad_accum=grad_accum,
            global_batch=global_batch if args.strategy != "overalloc" else per_batch * grad_accum,
            global_batch_mode=args.global_batch_mode,
            samples_seen=samples_seen,
            notes=notes,
            extra={
                "data_dir": args.data_dir,
                "layout": args.layout,
                "warmup_steps": args.warmup_steps,
                "strategy_doc": STRATEGIES[args.strategy],
                "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES", ""),
            },
        )
        if args.baseline_json and os.path.isfile(args.baseline_json):
            with open(args.baseline_json, encoding="utf-8") as f:
                base = json.load(f)
            base_sps = base.get("samples_per_sec") or 0.0
            if base_sps > 0 and report.samples_per_sec:
                report.speedup_vs_baseline = report.samples_per_sec / base_sps
                if args.strategy == "dp" and nproc_held > 1:
                    report.parallel_efficiency = report.speedup_vs_baseline / nproc_held
                if args.strategy == "overalloc" and nproc_held > 1:
                    # Ideal speedup=1; efficiency vs cards held
                    report.parallel_efficiency = (report.speedup_vs_baseline or 1.0) / nproc_held
                    report.extra["waste_factor"] = nproc_held / max(
                        1e-9, report.speedup_vs_baseline or 1.0
                    )

        tag = args.tag or f"w{dist.world_size}"
        out_name = f"{args.strategy}_n{nproc_held}_{tag}.json"
        out_path = os.path.join(args.out_dir, out_name)
        report.save(out_path)
        print(f"[sched-bench] wrote {out_path}", flush=True)
        print(json.dumps(report.to_dict(), indent=2), flush=True)

    cleanup_dist(dist)
    return report


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[sched-bench] strategy={args.strategy}: {STRATEGIES[args.strategy]}", flush=True)

    if args.strategy in BOUNDARY_UNSUPPORTED:
        _write_boundary_report(args, args.out_dir)
        return

    if args.strategy not in SUPPORTED_RUNNABLE:
        raise SystemExit(f"unknown strategy {args.strategy}")

    if args.strategy in {"dp", "overalloc"}:
        world = int(os.environ.get("WORLD_SIZE", "1"))
        if world <= 1:
            print(
                "[sched-bench] WARNING: dp/overalloc expect torchrun --nproc_per_node>1; "
                "running as single process",
                flush=True,
            )

    run_training(args)


if __name__ == "__main__":
    main()
