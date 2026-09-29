#!/usr/bin/env python3
"""Aggregate sched_bench JSON (infer-first, no-HCCL multi-process) into summary."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List


def load_reports(in_dir: str) -> List[Dict[str, Any]]:
    rows = []
    for name in sorted(os.listdir(in_dir)):
        if not name.endswith(".json"):
            continue
        if name.startswith("."):
            continue
        with open(os.path.join(in_dir, name), encoding="utf-8") as f:
            data = json.load(f)
        data["_file"] = name
        rows.append(data)
    return rows


def merge_replica_shards(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Combine replicas_nK_mpK_r* into one cluster row with summed QPS."""
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    others: List[Dict[str, Any]] = []
    for r in rows:
        if r.get("supported") is False:
            others.append(r)
            continue
        m = re.match(r"replicas_n(\d+)_mp\d+_r(\d+)\.json", r.get("_file", ""))
        if r.get("strategy") == "replicas" and m:
            key = f"replicas_n{m.group(1)}"
            groups[key].append(r)
        else:
            others.append(r)

    merged: List[Dict[str, Any]] = []
    for key, parts in groups.items():
        if len(parts) == 1:
            merged.append(parts[0])
            continue
        walls = [float(p.get("wall_s") or 0) for p in parts]
        qps_list = [float(p.get("qps") or 0) for p in parts]
        local_reqs = [int((p.get("extra") or {}).get("local_requests") or p.get("requests") or 0) for p in parts]
        peak = max((p.get("peak_mem_gb") or 0) for p in parts)
        # Prefer sum of per-rank QPS; also total_req / max_wall
        sum_qps = sum(qps_list)
        total_req = sum(local_reqs)
        max_wall = max(walls) if walls else 0
        alt = (total_req / max_wall) if max_wall > 0 else sum_qps
        # Use min of the two conservative? Use sum_qps as primary for independent serving
        agg = sum_qps
        # latency: pool all ranks' p50 etc by averaging reported p50
        def avg_lat(k):
            vals = []
            for p in parts:
                lat = p.get("latency_ms") or {}
                if lat.get(k) is not None:
                    vals.append(lat[k])
            return sum(vals) / len(vals) if vals else None

        nproc = int(parts[0].get("nproc") or len(parts))
        row = {
            "_file": f"{key}_merged.json",
            "strategy": "replicas",
            "mode": "infer",
            "nproc": nproc,
            "infer_op": parts[0].get("infer_op"),
            "batch_size": parts[0].get("batch_size"),
            "qps": sum_qps / max(1, len(parts)),
            "aggregate_qps": agg,
            "latency_ms": {
                "p50": avg_lat("p50"),
                "p90": avg_lat("p90"),
                "p99": avg_lat("p99"),
            },
            "peak_mem_gb": peak,
            "wall_s": max_wall,
            "card_seconds": max_wall * nproc,
            "phase_s": {},
            "extra": {
                "merged_from": [p.get("_file") for p in parts],
                "alt_qps_total_req_over_max_wall": alt,
                "waste_factor": None,
            },
            "notes": f"merged {len(parts)} independent replica ranks (no HCCL)",
        }
        # persist merged
        merged.append(row)
    return others + merged


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", required=True)
    p.add_argument("--out_dir", default="")
    args = p.parse_args()
    out_dir = args.out_dir or args.in_dir
    os.makedirs(out_dir, exist_ok=True)
    rows = merge_replica_shards(load_reports(args.in_dir))

    # write merged replicas artifacts
    for r in rows:
        if r.get("_file", "").endswith("_merged.json"):
            path = os.path.join(out_dir, r["_file"])
            with open(path, "w", encoding="utf-8") as f:
                json.dump(r, f, indent=2)

    baseline_qps = None
    for r in rows:
        if r.get("strategy") == "single" and r.get("mode", "infer") != "train":
            baseline_qps = r.get("aggregate_qps") or r.get("qps")
            if baseline_qps:
                break

    for r in rows:
        if r.get("supported") is False:
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
                    }
                )
                continue
            lat = r.get("latency_ms") or r.get("step_ms") or {}
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
                }
            )

    md_path = os.path.join(out_dir, "summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# sched_bench summary (inference-first, no HCCL MP)\n\n")
        f.write(
            "| strategy | nproc | agg QPS | p50 ms | p90 ms | p99 ms | peak GB | speedup | eff | waste |\n"
        )
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            if r.get("supported") is False:
                f.write(f"| {r.get('strategy')} | — | BOUNDARY | — | — | — | — | — | — | — |\n")
                continue
            lat = r.get("latency_ms") or {}
            extra = r.get("extra") or {}
            q = r.get("aggregate_qps") or r.get("qps")
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
