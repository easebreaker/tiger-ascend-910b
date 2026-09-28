"""Timing, percentile, and peak-memory collection for sched bench."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


def percentile(sorted_vals: List[float], p: float) -> Optional[float]:
    """Nearest-rank percentile; ``p`` in [0, 100]. Input need not be pre-sorted."""
    if not sorted_vals:
        return None
    xs = sorted(sorted_vals)
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


@dataclass
class BenchReport:
    strategy: str
    nproc: int
    world_size: int
    rank: int
    device: str
    backend: str
    steps: int
    batch_size: int
    grad_accum: int
    global_batch: int
    global_batch_mode: str
    samples_seen: int
    wall_s: float
    phase_s: Dict[str, float] = field(default_factory=dict)
    step_ms: Dict[str, Optional[float]] = field(default_factory=dict)
    samples_per_sec: Optional[float] = None
    steps_per_sec: Optional[float] = None
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
        self.step_times_s: List[float] = []
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
            dt = time.perf_counter() - t0
            self.phase_s[name] = self.phase_s.get(name, 0.0) + dt

    def record_step(self, dt_s: float) -> None:
        self.step_times_s.append(float(dt_s))

    def wall_s(self) -> float:
        return time.perf_counter() - self._wall0

    def step_percentiles_ms(self) -> Dict[str, Optional[float]]:
        xs = list(self.step_times_s)
        return {
            "count": float(len(xs)),
            "mean": (sum(xs) / len(xs) * 1000.0) if xs else None,
            "p50": (percentile(xs, 50) * 1000.0) if xs else None,
            "p90": (percentile(xs, 90) * 1000.0) if xs else None,
            "p99": (percentile(xs, 99) * 1000.0) if xs else None,
            "min": (min(xs) * 1000.0) if xs else None,
            "max": (max(xs) * 1000.0) if xs else None,
        }

    def build_report(self, **kwargs: Any) -> BenchReport:
        wall = self.wall_s()
        samples = int(kwargs.get("samples_seen", 0))
        nproc = int(kwargs.get("nproc", 1))
        peak = self.peak_mem_bytes()
        steps = int(kwargs.get("steps", len(self.step_times_s)))
        return BenchReport(
            strategy=str(kwargs.get("strategy", "")),
            nproc=nproc,
            world_size=int(kwargs.get("world_size", nproc)),
            rank=int(kwargs.get("rank", 0)),
            device=str(kwargs.get("device", self.device)),
            backend=str(kwargs.get("backend", "")),
            steps=steps,
            batch_size=int(kwargs.get("batch_size", 0)),
            grad_accum=int(kwargs.get("grad_accum", 1)),
            global_batch=int(kwargs.get("global_batch", 0)),
            global_batch_mode=str(kwargs.get("global_batch_mode", "")),
            samples_seen=samples,
            wall_s=wall,
            phase_s=dict(self.phase_s),
            step_ms=self.step_percentiles_ms(),
            samples_per_sec=(samples / wall) if wall > 0 else None,
            steps_per_sec=(steps / wall) if wall > 0 and steps else None,
            peak_mem_bytes=peak,
            peak_mem_gb=(peak / 1e9) if peak is not None else None,
            card_seconds=wall * max(1, nproc),
            notes=str(kwargs.get("notes", "")),
            extra=dict(kwargs.get("extra") or {}),
        )
