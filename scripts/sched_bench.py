#!/usr/bin/env python3
"""Multi-chip scheduling bench — inference-first (TIGER / Ascend 910B).

Primary strategies (scheduling claims):
  single      1-card inference baseline (latency P50/P90/P99, QPS, HBM)
  replicas    N full-model copies; shard requests (small-model scale-out)
  overalloc   N cards reserved, only rank0 serves (quota waste)

Boundary (JSON only): tp | pp | ep | sp

Optional secondary:
  train_profile  short training-step profile (not primary)

Examples:
  source scripts/ascend_env.sh
  export ASCEND_RT_VISIBLE_DEVICES=7
  python -u scripts/sched_bench.py --strategy single --device npu --requests 64

  ASCEND_RT_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 \\
    scripts/sched_bench.py --strategy replicas --device npu --requests 128 \\
    --baseline_json artifacts/sched_bench/.../single_*.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Optional, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tiger_ascend.bench.strategies import (  # noqa: E402
    BOUNDARY_UNSUPPORTED,
    STRATEGIES,
    SUPPORTED_INFER,
    SUPPORTED_RUNNABLE,
    SUPPORTED_TRAIN,
)


def parse_args():
    p = argparse.ArgumentParser(description="TIGER multi-chip scheduling bench (infer-first)")
    p.add_argument("--strategy", choices=list(STRATEGIES.keys()), required=True)
    p.add_argument("--device", choices=["npu", "cuda", "cpu"], default="npu")
    p.add_argument("--data_dir", default=os.path.join(ROOT, "data", "amazon_beauty", "subset_512u"))
    p.add_argument("--out_dir", default=os.path.join(ROOT, "artifacts", "sched_bench"))
    p.add_argument("--ckpt", default="", help="optional pytorch_model.bin (else random init)")
    # infer
    p.add_argument("--requests", type=int, default=64, help="total request batches to serve (cluster)")
    p.add_argument("--batch_size", type=int, default=8, help="sequences per request batch")
    p.add_argument("--warmup_requests", type=int, default=5)
    p.add_argument("--infer_op", choices=["generate", "forward"], default="generate")
    p.add_argument("--beam_size", type=int, default=10, help="generate beam (keep modest for bench)")
    p.add_argument("--max_new_tokens", type=int, default=5)
    # train_profile only
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--warmup_steps", type=int, default=3)
    p.add_argument("--grad_accum", type=int, default=1)
    # model
    p.add_argument("--layout", choices=["npu_safe", "xlt", "npu_large"], default="npu_safe")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--baseline_json", default="")
    p.add_argument("--tag", default="")
    return p.parse_args()


def _write_boundary(args) -> str:
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"boundary_{args.strategy}.json")
    payload = {
        "strategy": args.strategy,
        "mode": "infer",
        "supported": False,
        "reason": STRATEGIES[args.strategy],
        "recommendation": (
            "For ~5M TIGER on Ascend: schedule as single-card or multi-replica "
            "full copies. Do not allocate TP/PP/EP/SP for this workload."
        ),
        "scheduling_implication": {
            "prefer": ["single", "replicas"],
            "avoid_for_this_workload": ["tp", "pp", "ep", "sp", "overalloc"],
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[sched-bench] BOUNDARY → {path}", flush=True)
    return path


def _attach_speedup(report, args, nproc: int, metric_key: str = "aggregate_qps") -> None:
    if not (args.baseline_json and os.path.isfile(args.baseline_json)):
        return
    with open(args.baseline_json, encoding="utf-8") as f:
        base = json.load(f)
    base_v = base.get(metric_key) or base.get("qps") or base.get("samples_per_sec") or 0.0
    cur_v = getattr(report, metric_key, None) or report.qps or report.samples_per_sec or 0.0
    if base_v and cur_v:
        report.speedup_vs_baseline = float(cur_v) / float(base_v)
        if nproc > 1:
            report.parallel_efficiency = report.speedup_vs_baseline / nproc
        if args.strategy == "overalloc" and nproc > 1:
            report.extra["waste_factor"] = nproc / max(1e-9, report.speedup_vs_baseline)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[sched-bench] {args.strategy}: {STRATEGIES[args.strategy]}", flush=True)

    if args.strategy in BOUNDARY_UNSUPPORTED:
        _write_boundary(args)
        return
    if args.strategy not in SUPPORTED_RUNNABLE:
        raise SystemExit(f"unsupported strategy {args.strategy}")

    import torch
    from torch.utils.data import DataLoader, DistributedSampler, Subset
    from transformers import T5ForConditionalGeneration

    from tiger_ascend.bench.dist_utils import barrier, cleanup_dist, init_dist, is_main
    from tiger_ascend.bench.metrics import MetricsCollector
    from tiger_ascend.data.dataset import load_json
    from tiger_ascend.data.xlt import XLT_EOS_ID, XLT_PAD_ID, XLT_VOCAB_SIZE, XltSeqDataset
    from tiger_ascend.model.tiger import build_xlt_t5_config
    from tiger_ascend.utils.device import (
        move_batch_to_device,
        move_module_to_device_safe,
        prepare_npu_runtime,
        warmup_npu,
    )

    if args.strategy in {"replicas", "overalloc"} and int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        print(
            "[sched-bench] WARNING: replicas/overalloc expect torchrun --nproc_per_node>1",
            flush=True,
        )

    dist = init_dist(args.device)
    if args.device == "npu" and not dist.enabled:
        try:
            warmup_npu(0, log=is_main(dist))
        except Exception as e:
            print(f"[sched-bench] warmup warn: {e!r}", flush=True)

    metrics = MetricsCollector(dist.device)

    with metrics.phase("data"):
        inters = load_json(os.path.join(args.data_dir, "inter.json"))
        indices = load_json(os.path.join(args.data_dir, "semantic_ids.json"))
        # valid split ≈ inference traffic shape
        ds = XltSeqDataset(inters, indices, max_his_len=20, mode="valid")
        if len(ds) == 0:
            ds = XltSeqDataset(inters, indices, max_his_len=20, mode="train")

    active = True
    if args.strategy == "overalloc":
        active = dist.rank == 0

    def build_model():
        cfg = build_xlt_t5_config(
            XLT_VOCAB_SIZE, dropout_rate=0.0, layout=args.layout, use_cache=True
        )
        try:
            model = T5ForConditionalGeneration(cfg, attn_implementation="eager")
        except TypeError:
            model = T5ForConditionalGeneration(cfg)
        if hasattr(model.config, "_attn_implementation"):
            model.config._attn_implementation = "eager"
        if args.ckpt:
            state = torch.load(args.ckpt, map_location="cpu")
            model.load_state_dict(state, strict=False)
        if dist.device.type == "npu":
            prepare_npu_runtime(dist.device.index or 0, log=False)
            _ = torch.zeros(1, device=dist.device)
            torch.npu.synchronize()
            model = move_module_to_device_safe(
                model, dist.device, log=False, sync_each=False, strategy="copy_"
            )
        else:
            model.to(dist.device)
        model.eval()
        return model

    # ---------- inference paths ----------
    if args.strategy in SUPPORTED_INFER:
        model = None
        if active:
            with metrics.phase("init_model"):
                torch.manual_seed(args.seed + dist.rank)
                model = build_model()

        # Shard requests across replicas; overalloc/single: all requests on active rank
        if args.strategy == "replicas" and dist.world_size > 1:
            # each rank handles ceil(total/world) batches
            local_reqs = (args.requests + dist.world_size - 1) // dist.world_size
        else:
            local_reqs = args.requests if active else 0

        sampler = None
        if args.strategy == "replicas" and dist.enabled and dist.world_size > 1 and active:
            sampler = DistributedSampler(
                ds, num_replicas=dist.world_size, rank=dist.rank, shuffle=False
            )

        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=sampler,
            collate_fn=ds.get_collate_fn(),
            num_workers=args.num_workers,
            drop_last=True,
        )
        it = iter(loader) if active else None

        def next_batch():
            nonlocal it
            try:
                return next(it)
            except StopIteration:
                it = iter(loader)
                return next(it)

        barrier(dist)
        done = 0
        with metrics.phase("infer"), torch.no_grad():
            # idle ranks still barrier-sync for overalloc card hold
            target = local_reqs if active else args.requests
            for i in range(1, (local_reqs if active else args.requests) + 1):
                if not active:
                    barrier(dist)
                    continue
                batch = next_batch()
                batch = move_batch_to_device(
                    {k: v for k, v in batch.items() if k != "labels"},
                    dist.device,
                    non_blocking=False,
                )
                t0 = time.perf_counter()
                if args.infer_op == "forward":
                    # cheap path: teacher-forcing style forward if labels present
                    full = next_batch()
                    full = move_batch_to_device(full, dist.device, non_blocking=False)
                    _ = model(**full).loss
                else:
                    _ = model.generate(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        max_length=args.max_new_tokens,
                        num_beams=max(1, args.beam_size),
                        num_return_sequences=1,
                        eos_token_id=XLT_EOS_ID,
                        pad_token_id=XLT_PAD_ID,
                        decoder_start_token_id=XLT_PAD_ID,
                    )
                if dist.device.type == "npu":
                    torch.npu.synchronize()
                elif dist.device.type == "cuda":
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                done += 1
                if done > args.warmup_requests:
                    metrics.record_latency(dt)
                if is_main(dist) and (done % 10 == 0 or done == local_reqs):
                    print(
                        f"[sched-bench] infer {args.strategy} "
                        f"req={done}/{local_reqs} last_ms={dt*1000:.1f}",
                        flush=True,
                    )
                barrier(dist)

        barrier(dist)
        if is_main(dist):
            # For replicas, aggregate QPS ≈ sum of per-rank; approximate as
            # (total requests) / wall using main rank wall and full request count.
            nproc = dist.world_size
            notes = ""
            if args.strategy == "overalloc":
                notes = f"overalloc: held={nproc} serving_replicas=1"
            if args.strategy == "replicas":
                notes = f"replicas: full-model copies={nproc} (not TP)"

            # Rebuild collector wall is from main only; use requests=cluster total for agg
            report = metrics.build_infer_report(
                strategy=args.strategy,
                nproc=nproc,
                world_size=dist.world_size,
                rank=dist.rank,
                device=str(dist.device),
                backend=dist.backend,
                requests=done if args.strategy != "replicas" else args.requests,
                batch_size=args.batch_size,
                beam_size=args.beam_size if args.infer_op == "generate" else 0,
                infer_op=args.infer_op,
                notes=notes,
                extra={
                    "data_dir": args.data_dir,
                    "ckpt": args.ckpt or None,
                    "layout": args.layout,
                    "local_requests": done,
                    "cluster_requests": args.requests,
                    "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES", ""),
                    "strategy_doc": STRATEGIES[args.strategy],
                },
            )
            if args.strategy == "replicas" and report.wall_s > 0:
                # Effective cluster QPS: all ranks finish ~same wall; total req / wall
                report.aggregate_qps = args.requests / report.wall_s
                report.qps = (done / report.wall_s) if report.wall_s > 0 else None
            _attach_speedup(report, args, nproc, metric_key="aggregate_qps")
            tag = args.tag or f"w{nproc}"
            out = os.path.join(args.out_dir, f"{args.strategy}_n{nproc}_{tag}.json")
            report.save(out)
            print(f"[sched-bench] wrote {out}", flush=True)
            print(json.dumps(report.to_dict(), indent=2), flush=True)
        cleanup_dist(dist)
        return

    # ---------- optional train_profile ----------
    if args.strategy in SUPPORTED_TRAIN:
        from torch.utils.data import DataLoader as DL

        train_ds = XltSeqDataset(inters, indices, max_his_len=20, mode="train")
        loader = DL(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=train_ds.get_collate_fn(),
            num_workers=0,
            drop_last=True,
        )
        with metrics.phase("init_model"):
            model = build_model()
            model.train()
            optim = torch.optim.Adam(model.parameters(), lr=1e-4)
        it = iter(loader)
        with metrics.phase("train"):
            for step in range(1, args.steps + 1):
                t0 = time.perf_counter()
                try:
                    batch = next(it)
                except StopIteration:
                    it = iter(loader)
                    batch = next(it)
                batch = move_batch_to_device(batch, dist.device, non_blocking=False)
                loss = model(**batch).loss
                loss.backward()
                optim.step()
                optim.zero_grad(set_to_none=True)
                if dist.device.type == "npu":
                    torch.npu.synchronize()
                dt = time.perf_counter() - t0
                if step > args.warmup_steps:
                    metrics.record_step(dt)
        if is_main(dist):
            report = metrics.build_train_report(
                strategy="train_profile",
                nproc=1,
                world_size=1,
                rank=0,
                device=str(dist.device),
                backend=dist.backend,
                steps=args.steps,
                samples_seen=args.steps * args.batch_size,
                notes="secondary training profile only",
                extra={"layout": args.layout},
            )
            _attach_speedup(report, args, 1, metric_key="samples_per_sec")
            tag = args.tag or "train"
            out = os.path.join(args.out_dir, f"train_profile_n1_{tag}.json")
            report.save(out)
            print(f"[sched-bench] wrote {out}", flush=True)
            print(json.dumps(report.to_dict(), indent=2), flush=True)
        cleanup_dist(dist)
        return


if __name__ == "__main__":
    main()
