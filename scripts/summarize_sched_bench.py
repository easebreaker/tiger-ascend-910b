#!/usr/bin/env python3
"""Aggregate sched_bench JSON runs into summary.csv + summary.md."""

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
        path = os.path.join(in_dir, name)
        with open(path, encoding="utf-8") as f:
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

    # Prefer single n1 as baseline if speedup missing
    baseline_sps = None
    for r in rows:
        if r.get("strategy") == "single" and r.get("samples_per_sec"):
            baseline_sps = r["samples_per_sec"]
            break

    for r in rows:
        if r.get("supported") is False:
            continue
        sps = r.get("samples_per_sec")
        if baseline_sps and sps and not r.get("speedup_vs_baseline"):
            r["speedup_vs_baseline"] = sps / baseline_sps
            n = max(1, int(r.get("nproc") or 1))
            if r.get("strategy") == "dp" and n > 1:
                r["parallel_efficiency"] = r["speedup_vs_baseline"] / n
            if r.get("strategy") == "overalloc" and n > 1:
                r["parallel_efficiency"] = r["speedup_vs_baseline"] / n
                r.setdefault("extra", {})["waste_factor"] = n / max(1e-9, r["speedup_vs_baseline"])

    csv_path = os.path.join(out_dir, "summary.csv")
    fields = [
        "file",
        "strategy",
        "nproc",
        "global_batch",
        "samples_per_sec",
        "steps_per_sec",
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
        "train_s",
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
                        "strategy": r.get("strategy"),
                        "nproc": "",
                        "global_batch": "",
                        "samples_per_sec": "BOUNDARY",
                        "steps_per_sec": "",
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
                        "train_s": "",
                        "data_s": r.get("reason", r.get("recommendation", "")),
                    }
                )
                continue
            step = r.get("step_ms") or {}
            phase = r.get("phase_s") or {}
            extra = r.get("extra") or {}
            w.writerow(
                {
                    "file": r.get("_file"),
                    "strategy": r.get("strategy"),
                    "nproc": r.get("nproc"),
                    "global_batch": r.get("global_batch"),
                    "samples_per_sec": r.get("samples_per_sec"),
                    "steps_per_sec": r.get("steps_per_sec"),
                    "p50_ms": step.get("p50"),
                    "p90_ms": step.get("p90"),
                    "p99_ms": step.get("p99"),
                    "peak_mem_gb": r.get("peak_mem_gb"),
                    "wall_s": r.get("wall_s"),
                    "card_seconds": r.get("card_seconds"),
                    "speedup": r.get("speedup_vs_baseline"),
                    "efficiency": r.get("parallel_efficiency"),
                    "waste_factor": extra.get("waste_factor"),
                    "init_model_s": phase.get("init_model"),
                    "train_s": phase.get("train"),
                    "data_s": phase.get("data"),
                }
            )

    md_path = os.path.join(out_dir, "summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# sched_bench summary\n\n")
        f.write("| strategy | nproc | samples/s | p50 ms | p99 ms | peak GB | speedup | eff | waste |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            if r.get("supported") is False:
                f.write(f"| {r.get('strategy')} | — | BOUNDARY | — | — | — | — | — | — |\n")
                continue
            step = r.get("step_ms") or {}
            extra = r.get("extra") or {}
            f.write(
                f"| {r.get('strategy')} | {r.get('nproc')} | "
                f"{r.get('samples_per_sec') or '—':.4g} | "
                f"{step.get('p50') or '—'} | {step.get('p99') or '—'} | "
                f"{r.get('peak_mem_gb') or '—'} | "
                f"{r.get('speedup_vs_baseline') or '—'} | "
                f"{r.get('parallel_efficiency') or '—'} | "
                f"{extra.get('waste_factor') or '—'} |\n"
            )

    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
