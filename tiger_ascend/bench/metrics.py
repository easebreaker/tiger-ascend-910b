"""Timing, percentile, and peak-memory collection for sched bench."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


def percentile(vals: List[float], p: float) -> Optional[float]:
    """Nearest-rank percentile; ``p`` in [0, 100]."""
    if not vals:
        return None
    xs = sorted(vals)
    if p <= 0:
        return float(xs[0])
    if p >= 100:
        return float(xs[-1])
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return float(xs[f])
    return float(xs[f] * (c - k) + xs[c] * (k - f))


def pct_dict_ms(times_s: List[float]) -> Dict[str, Optional[float]]:
    if not times_s:
        return {
            "count": 0.0,
            "mean": None,
            "p50": None,
            "p90": None,
            "p99": None,
            "min": None,
            "max": None,
        }
    return {
        "count": float(len(times_s)),
        "mean": (sum(times_s) / len(times_s) * 1000.0),
        "p50": percentile(times_s, 50) * 1000.0,
        "p90": percentile(times_s, 90) * 1000.0,
        "p99": percentile(times_s, 99) * 1000.0,
        "min": min(times_s) * 1000.0,
        "max": max(times_s) * 1000.0,
    }


@dataclass
class BenchReport:
    """Unified report; inference fields are primary for scheduling claims."""

    strategy: str
    mode: str  # infer | train
    nproc: int
    world_size: int
    rank: int
    device: str
    backend: str
    # infer
    requests: int = 0
    batch_size: int = 0
    beam_size: int = 0
    infer_op: str = ""  # generate | forward
    latency_ms: Dict[str, Optional[float]] = field(default_factory=dict)
    qps: Optional[float] = None  # requests / wall on this rank (or aggregate note)
    aggregate_qps: Optional[float] = None  # cluster effective QPS
    # train (optional)
    steps: int = 0
    step_ms: Dict[str, Optional[float]] = field(default_factory=dict)
    samples_per_sec: Optional[float] = None
    # shared
    wall_s: float = 0.0
    phase_s: Dict[str, float] = field(default_factory=dict)
    peak_mem_bytes: Optional[int] = None
    peak_mem_gb: Optional[float] = None
    card_seconds: Optional[float] = None
    speedup_vs_baseline: Optional[float] = None
    parallel_efficiency: Optional[float] = None
    notes: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


class MetricsCollector:
    def __init__(self, device):
        import torch

        self._torch = torch
        self.device = device
        self.phase_s: Dict[str, float] = {}
        self.latencies_s: List[float] = []
        self._wall0 = time.perf_counter()
        self._reset_peak_mem()

    def _reset_peak_mem(self) -> None:
        torch = self._torch
        if self.device.type == "npu" and hasattr(torch, "npu"):
            try:
                torch.npu.reset_peak_memory_stats(self.device)
            except Exception:
                try:
                    torch.npu.reset_peak_memory_stats()
                except Exception:
                    pass
        elif self.device.type == "cuda":
            try:
                torch.cuda.reset_peak_memory_stats(self.device)
            except Exception:
                pass

    def peak_mem_bytes(self) -> Optional[int]:
        torch = self._torch
        try:
            if self.device.type == "npu" and hasattr(torch, "npu"):
                return int(torch.npu.max_memory_allocated(self.device))
            if self.device.type == "cuda":
                return int(torch.cuda.max_memory_allocated(self.device))
        except Exception:
            return None
        return None

    @contextmanager
    def phase(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.phase_s[name] = self.phase_s.get(name, 0.0) + (time.perf_counter() - t0)

    def record_latency(self, dt_s: float) -> None:
        self.latencies_s.append(float(dt_s))

    # alias for train-profile reuse
    def record_step(self, dt_s: float) -> None:
        self.record_latency(dt_s)

    def wall_s(self) -> float:
        return time.perf_counter() - self._wall0

    def build_infer_report(self, **kwargs: Any) -> BenchReport:
        wall = self.wall_s()
        nproc = int(kwargs.get("nproc", 1))
        requests = int(kwargs.get("requests", 0))
        peak = self.peak_mem_bytes()
        rank_qps = (requests / wall) if wall > 0 else None
        # For replicas: each rank serves requests/nproc; aggregate ≈ rank_qps * nproc if balanced
        agg = kwargs.get("aggregate_qps")
        if agg is None and rank_qps is not None and kwargs.get("strategy") == "replicas":
            agg = rank_qps * max(1, nproc)
        elif agg is None:
            agg = rank_qps
        return BenchReport(
            strategy=str(kwargs.get("strategy", "")),
            mode="infer",
            nproc=nproc,
            world_size=int(kwargs.get("world_size", nproc)),
            rank=int(kwargs.get("rank", 0)),
            device=str(kwargs.get("device", self.device)),
            backend=str(kwargs.get("backend", "")),
            requests=requests,
            batch_size=int(kwargs.get("batch_size", 0)),
            beam_size=int(kwargs.get("beam_size", 0)),
            infer_op=str(kwargs.get("infer_op", "")),
            latency_ms=pct_dict_ms(self.latencies_s),
            qps=rank_qps,
            aggregate_qps=agg,
            wall_s=wall,
            phase_s=dict(self.phase_s),
            peak_mem_bytes=peak,
            peak_mem_gb=(peak / 1e9) if peak is not None else None,
            card_seconds=wall * max(1, nproc),
            notes=str(kwargs.get("notes", "")),
            extra=dict(kwargs.get("extra") or {}),
        )

    def build_train_report(self, **kwargs: Any) -> BenchReport:
        wall = self.wall_s()
        nproc = int(kwargs.get("nproc", 1))
        samples = int(kwargs.get("samples_seen", 0))
        steps = int(kwargs.get("steps", len(self.latencies_s)))
        peak = self.peak_mem_bytes()
        return BenchReport(
            strategy=str(kwargs.get("strategy", "train_profile")),
            mode="train",
            nproc=nproc,
            world_size=int(kwargs.get("world_size", nproc)),
            rank=int(kwargs.get("rank", 0)),
            device=str(kwargs.get("device", self.device)),
            backend=str(kwargs.get("backend", "")),
            steps=steps,
            step_ms=pct_dict_ms(self.latencies_s),
            samples_per_sec=(samples / wall) if wall > 0 else None,
            wall_s=wall,
            phase_s=dict(self.phase_s),
            peak_mem_bytes=peak,
            peak_mem_gb=(peak / 1e9) if peak is not None else None,
            card_seconds=wall * max(1, nproc),
            notes=str(kwargs.get("notes", "")),
            extra=dict(kwargs.get("extra") or {}),
        )
