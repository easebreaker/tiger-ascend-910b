"""Scheduling / multi-chip benchmark utilities for TIGER on Ascend."""

from .metrics import BenchReport, MetricsCollector, percentile, pct_dict_ms
from .strategies import (
    BOUNDARY_UNSUPPORTED,
    STRATEGIES,
    SUPPORTED_INFER,
    SUPPORTED_RUNNABLE,
    SUPPORTED_TRAIN,
    StrategyName,
)

__all__ = [
    "BOUNDARY_UNSUPPORTED",
    "BenchReport",
    "MetricsCollector",
    "STRATEGIES",
    "SUPPORTED_INFER",
    "SUPPORTED_RUNNABLE",
    "SUPPORTED_TRAIN",
    "StrategyName",
    "percentile",
    "pct_dict_ms",
]
