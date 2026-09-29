#!/usr/bin/env python3
"""Multi-chip scheduling bench — inference-first (TIGER / Ascend 910B).

Primary strategies (NO HCCL required for multi-card):
  single / replicas / overalloc  — use launch_infer_mp.sh for multi-card
  tp|pp|ep|sp                    — boundary JSON only
  train_profile                  — optional single-card train short profile

Multi-card inference intentionally avoids torchrun+HCCL (error code 1 on many
boxes). Each process gets one ASCEND_RT_VISIBLE_DEVICES id → logical npu:0.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

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
    p.add_argument("--ckpt", default="", help="optional pytorch_model.bin")
    p.add_argument("--requests", type=int, default=64, help="cluster total request batches")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--warmup_requests", type=int, default=5)
    p.add_argument("--infer_op", choices=["generate", "forward"], default="generate")
    p.add_argument("--beam_size", type=int, default=10)
    p.add_argument("--max_new_tokens", type=int, default=5)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--warmup_steps", type=int, default=3)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--layout", choices=["npu_safe", "xlt", "npu_large"], default="npu_safe")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--baseline_json", default="")
    p.add_argument("--tag", default="")
    # Independent multi-process (no HCCL) — set by launch_infer_mp.sh
    p.add_argument("--mp_rank", type=int, default=0)
    p.add_argument("--mp_world_size", type=int, default=1)
    p.add_argument("--mp_role", choices=["", "replicas", "overalloc"], default="")
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
    print(
        f"[sched-bench] visible={os.environ.get('ASCEND_RT_VISIBLE_DEVICES')!r} "
        f"mp_rank={args.mp_rank}/{args.mp_world_size} role={args.mp_role!r}",
        flush=True,
    )

    if args.strategy in BOUNDARY_UNSUPPORTED:
        _write_boundary(args)
        return
    if args.strategy not in SUPPORTED_RUNNABLE:
        raise SystemExit(f"unsupported strategy {args.strategy}")

    import torch
    from torch.utils.data import DataLoader
    from transformers import T5ForConditionalGeneration

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

    # Resolve logical device: with one visible card, always npu:0 / cuda:0
    if args.device == "npu":
        device = torch.device("npu:0")
        try:
            warmup_npu(0, log=(args.mp_rank == 0))
        except Exception as e:
            print(f"[sched-bench] warmup warn: {e!r}", flush=True)
    elif args.device == "cuda":
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    mp_n = max(1, int(args.mp_world_size))
    mp_r = int(args.mp_rank)
    role = args.mp_role or args.strategy

    # overalloc idle ranks: hold device context until rank0 finishes
    done_flag = Path(args.out_dir) / f".overalloc_done_n{mp_n}"
    if args.strategy == "overalloc" and mp_r > 0:
        print(f"[sched-bench] overalloc idle rank={mp_r} holding {device}", flush=True)
        holder = None
        if device.type != "cpu":
            holder = torch.zeros(1, device=device)
            if device.type == "npu":
                torch.npu.synchronize()
        t0 = time.perf_counter()
        while not done_flag.exists():
            time.sleep(0.5)
            if time.perf_counter() - t0 > 7200:
                raise SystemExit("overalloc idle timeout waiting for rank0")
        del holder
        print(f"[sched-bench] idle rank={mp_r} exit", flush=True)
        return

    metrics = MetricsCollector(device)

    with metrics.phase("data"):
        inters = load_json(os.path.join(args.data_dir, "inter.json"))
        indices = load_json(os.path.join(args.data_dir, "semantic_ids.json"))
        ds = XltSeqDataset(inters, indices, max_his_len=20, mode="valid")
        if len(ds) == 0:
            ds = XltSeqDataset(inters, indices, max_his_len=20, mode="train")

    def build_model(train_mode: bool = False):
        cfg = build_xlt_t5_config(
            XLT_VOCAB_SIZE,
            dropout_rate=0.1 if train_mode else 0.0,
            layout=args.layout,
            use_cache=not train_mode,
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
        if device.type == "npu":
            prepare_npu_runtime(0, log=False)
            _ = torch.zeros(1, device=device)
            torch.npu.synchronize()
            model = move_module_to_device_safe(
                model, device, log=False, sync_each=False, strategy="copy_"
            )
        else:
            model.to(device)
        if train_mode:
            model.train()
        else:
            model.eval()
        return model

    # ---------- inference ----------
    if args.strategy in SUPPORTED_INFER:
        with metrics.phase("init_model"):
            torch.manual_seed(args.seed + mp_r)
            model = build_model(False)

        if args.strategy == "replicas" and mp_n > 1:
            # shard cluster requests across ranks
            local_reqs = args.requests // mp_n + (1 if mp_r < (args.requests % mp_n) else 0)
        else:
            local_reqs = args.requests

        # Deterministic shard of dataset indices per rank
        indices_all = list(range(len(ds)))
        if args.strategy == "replicas" and mp_n > 1:
            indices_all = indices_all[mp_r::mp_n] or indices_all
        from torch.utils.data import Subset

        subset = Subset(ds, indices_all)
        loader = DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=ds.get_collate_fn(),
            num_workers=args.num_workers,
            drop_last=True,
        )
        it = iter(loader)

        def next_batch():
            nonlocal it
            try:
                return next(it)
            except StopIteration:
                it = iter(loader)
                return next(it)

        done = 0
        with metrics.phase("infer"), torch.no_grad():
            for _ in range(local_reqs):
                batch = next_batch()
                t0 = time.perf_counter()
                if args.infer_op == "forward":
                    full = move_batch_to_device(batch, device, non_blocking=False)
                    _ = model(**full).loss
                else:
                    batch = move_batch_to_device(
                        {k: v for k, v in batch.items() if k != "labels"},
                        device,
                        non_blocking=False,
                    )
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
                if device.type == "npu":
                    torch.npu.synchronize()
                elif device.type == "cuda":
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                done += 1
                if done > args.warmup_requests:
                    metrics.record_latency(dt)
                if done % 10 == 0 or done == local_reqs:
                    print(
                        f"[sched-bench] rank={mp_r} {args.strategy} "
                        f"req={done}/{local_reqs} last_ms={dt*1000:.1f}",
                        flush=True,
                    )

        if args.strategy == "overalloc" and mp_r == 0:
            done_flag.write_text("ok")

        nproc = mp_n if args.strategy in {"replicas", "overalloc"} else 1
        notes = ""
        if args.strategy == "overalloc":
            notes = f"overalloc: held={nproc} serving=1 (no HCCL; independent procs)"
        if args.strategy == "replicas":
            notes = f"replicas: copies={nproc} independent procs (no HCCL)"

        report = metrics.build_infer_report(
            strategy=args.strategy,
            nproc=nproc,
            world_size=mp_n,
            rank=mp_r,
            device=str(device),
            backend="none-mp",
            requests=done,
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
                "mp_rank": mp_r,
                "mp_world_size": mp_n,
                "strategy_doc": STRATEGIES[args.strategy],
                "hccl": False,
            },
        )
        # Per-rank QPS; cluster agg filled later by summarize for replicas
        if report.wall_s > 0:
            report.qps = done / report.wall_s
            if args.strategy == "single":
                report.aggregate_qps = report.qps
            elif args.strategy == "overalloc":
                report.aggregate_qps = report.qps  # only one server
            else:
                # provisional; summarize will sum ranks
                report.aggregate_qps = report.qps * mp_n
        _attach_speedup(report, args, nproc, metric_key="aggregate_qps")
        tag = args.tag or f"w{nproc}"
        out = os.path.join(args.out_dir, f"{args.strategy}_n{nproc}_{tag}.json")
        report.save(out)
        print(f"[sched-bench] wrote {out}", flush=True)
        print(json.dumps(report.to_dict(), indent=2), flush=True)
        return

    # ---------- optional train_profile (single card) ----------
    if args.strategy in SUPPORTED_TRAIN:
        train_ds = XltSeqDataset(inters, indices, max_his_len=20, mode="train")
        loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=train_ds.get_collate_fn(),
            num_workers=0,
            drop_last=True,
        )
        with metrics.phase("init_model"):
            model = build_model(True)
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
                batch = move_batch_to_device(batch, device, non_blocking=False)
                loss = model(**batch).loss
                loss.backward()
                optim.step()
                optim.zero_grad(set_to_none=True)
                if device.type == "npu":
                    torch.npu.synchronize()
                dt = time.perf_counter() - t0
                if step > args.warmup_steps:
                    metrics.record_step(dt)
        report = metrics.build_train_report(
            strategy="train_profile",
            nproc=1,
            world_size=1,
            rank=0,
            device=str(device),
            backend="none",
            steps=args.steps,
            samples_seen=args.steps * args.batch_size,
            notes="secondary training profile only",
            extra={"layout": args.layout},
        )
        tag = args.tag or "train"
        out = os.path.join(args.out_dir, f"train_profile_n1_{tag}.json")
        report.save(out)
        print(f"[sched-bench] wrote {out}", flush=True)
        print(json.dumps(report.to_dict(), indent=2), flush=True)


if __name__ == "__main__":
    main()
