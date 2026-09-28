#!/usr/bin/env python3
"""Aggregate sched_bench JSON (infer-first) into summary.csv + summary.md."""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, List


def load_reports(in_dir: str) -> List[Dict[str, Any]]:
    rows = []
    for name in sorted(os.listdir(in_dir)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(in_dir, name), encoding="utf-8") as f:
            data = json.load(f)
        data["_file"] = name
        rows.append(data)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", required=True)
    p.add_argument("--out_dir", default="")
    args = p.parse_args()
    out_dir = args.out_dir or args.in_dir
    os.makedirs(out_dir, exist_ok=True)
    rows = load_reports(args.in_dir)

    baseline_qps = None
    for r in rows:
        if r.get("strategy") == "single" and r.get("mode", "infer") == "infer":
            baseline_qps = r.get("aggregate_qps") or r.get("qps")
            if baseline_qps:
                break

    for r in rows:
        if r.get("supported") is False:
            continue
        if r.get("mode") == "train":
            continue
        qps = r.get("aggregate_qps") or r.get("qps")
        if baseline_qps and qps and not r.get("speedup_vs_baseline"):
            r["speedup_vs_baseline"] = qps / baseline_qps
            n = max(1, int(r.get("nproc") or 1))
            if r.get("strategy") == "replicas" and n > 1:
                r["parallel_efficiency"] = r["speedup_vs_baseline"] / n
            if r.get("strategy") == "overalloc" and n > 1:
                r["parallel_efficiency"] = r["speedup_vs_baseline"] / n
                r.setdefault("extra", {})["waste_factor"] = n / max(1e-9, r["speedup_vs_baseline"])

    csv_path = os.path.join(out_dir, "summary.csv")
    fields = [
        "file",
        "mode",
        "strategy",
        "nproc",
        "infer_op",
        "batch_size",
        "qps",
        "aggregate_qps",
        "p50_ms",
        "p90_ms",
        "p99_ms",
        "peak_mem_gb",
        "wall_s",
        "card_seconds",
        "speedup",
        "efficiency",
        "waste_factor",
        "init_model_s",
        "infer_s",
        "data_s",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            if r.get("supported") is False:
                w.writerow(
                    {
                        "file": r.get("_file"),
                        "mode": "infer",
                        "strategy": r.get("strategy"),
                        "nproc": "",
                        "infer_op": "BOUNDARY",
                        "batch_size": "",
                        "qps": "",
                        "aggregate_qps": "",
                        "p50_ms": "",
                        "p90_ms": "",
                        "p99_ms": "",
                        "peak_mem_gb": "",
                        "wall_s": "",
                        "card_seconds": "",
                        "speedup": "",
                        "efficiency": "",
                        "waste_factor": "",
                        "init_model_s": "",
                        "infer_s": r.get("reason", ""),
                        "data_s": "",
                    }
                )
                continue
            lat = r.get("latency_ms") or r.get("step_ms") or {}
            phase = r.get("phase_s") or {}
            extra = r.get("extra") or {}
            w.writerow(
                {
                    "file": r.get("_file"),
                    "mode": r.get("mode", "infer"),
                    "strategy": r.get("strategy"),
                    "nproc": r.get("nproc"),
                    "infer_op": r.get("infer_op", ""),
                    "batch_size": r.get("batch_size"),
                    "qps": r.get("qps"),
                    "aggregate_qps": r.get("aggregate_qps"),
                    "p50_ms": lat.get("p50"),
                    "p90_ms": lat.get("p90"),
                    "p99_ms": lat.get("p99"),
                    "peak_mem_gb": r.get("peak_mem_gb"),
                    "wall_s": r.get("wall_s"),
                    "card_seconds": r.get("card_seconds"),
                    "speedup": r.get("speedup_vs_baseline"),
                    "efficiency": r.get("parallel_efficiency"),
                    "waste_factor": extra.get("waste_factor"),
                    "init_model_s": phase.get("init_model"),
                    "infer_s": phase.get("infer") or phase.get("train"),
                    "data_s": phase.get("data"),
                }
            )

    md_path = os.path.join(out_dir, "summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# sched_bench summary (inference-first)\n\n")
        f.write(
            "| strategy | nproc | agg QPS | p50 ms | p90 ms | p99 ms | peak GB | speedup | eff | waste |\n"
        )
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            if r.get("supported") is False:
                f.write(f"| {r.get('strategy')} | — | BOUNDARY | — | — | — | — | — | — | — |\n")
                continue
            lat = r.get("latency_ms") or r.get("step_ms") or {}
            extra = r.get("extra") or {}
            q = r.get("aggregate_qps") or r.get("qps") or r.get("samples_per_sec")
            f.write(
                f"| {r.get('strategy')} | {r.get('nproc')} | {q} | "
                f"{lat.get('p50')} | {lat.get('p90')} | {lat.get('p99')} | "
                f"{r.get('peak_mem_gb')} | {r.get('speedup_vs_baseline')} | "
                f"{r.get('parallel_efficiency')} | {extra.get('waste_factor')} |\n"
            )

    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
