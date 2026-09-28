"""Unit tests for sched bench metrics (no torch required for percentile)."""

import importlib.util
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_percentile():
    path = os.path.join(ROOT, "tiger_ascend", "bench", "metrics.py")
    spec = importlib.util.spec_from_file_location("bench_metrics_iso", path)
    mod = importlib.util.module_from_spec(spec)
    # Avoid importing package __init__ that pulls torch via MetricsCollector usage path —
    # loading metrics.py still executes imports; torch is only inside MetricsCollector now.
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.percentile


def test_percentile_basic():
    percentile = _load_percentile()
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(xs, 0) == 1.0
    assert percentile(xs, 100) == 5.0
    assert percentile(xs, 50) == 3.0
    assert percentile([], 50) is None


def test_strategies_catalog():
    # import strategies module without torch
    path = os.path.join(ROOT, "tiger_ascend", "bench", "strategies.py")
    spec = importlib.util.spec_from_file_location("bench_strategies_iso", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    assert set(mod.SUPPORTED_RUNNABLE).issubset(mod.STRATEGIES)
    assert set(mod.BOUNDARY_UNSUPPORTED).issubset(mod.STRATEGIES)
    assert "dp" in mod.SUPPORTED_RUNNABLE
    assert "tp" in mod.BOUNDARY_UNSUPPORTED
