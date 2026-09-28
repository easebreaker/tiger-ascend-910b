"""Unit tests for sched bench (no torch required)."""

import importlib.util
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load(name: str, rel: str):
    path = os.path.join(ROOT, rel)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_percentile_and_pct_dict():
    m = _load("bench_metrics_iso", "tiger_ascend/bench/metrics.py")
    assert m.percentile([1, 2, 3, 4, 5], 50) == 3.0
    d = m.pct_dict_ms([0.01, 0.02, 0.03])
    assert d["count"] == 3.0
    assert d["p50"] is not None


def test_infer_first_strategies():
    s = _load("bench_strategies_iso", "tiger_ascend/bench/strategies.py")
    assert "replicas" in s.SUPPORTED_INFER
    assert "single" in s.SUPPORTED_INFER
    assert "overalloc" in s.SUPPORTED_INFER
    assert "dp" not in s.SUPPORTED_RUNNABLE  # training DP removed from primary
    assert "train_profile" in s.SUPPORTED_TRAIN
    assert "tp" in s.BOUNDARY_UNSUPPORTED
