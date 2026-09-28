"""Allocation / parallel strategies for sched bench."""

from __future__ import annotations

from typing import Dict, List

# Strategies exposed on CLI. TP/PP/EP are documented as unsupported for this
# workload (negative boundary) — selecting them exits with a structured report.
STRATEGIES: Dict[str, str] = {
    "single": "1-card training baseline (full model replica, all steps)",
    "dp": "Data Parallel (DDP): each card trains different shards, grads allreduce",
    "overalloc": (
        "Request N cards via torchrun but only rank0 trains; others idle at barrier. "
        "Measures scheduling waste of over-allocating small jobs."
    ),
    "tp": "UNSUPPORTED boundary: Tensor Parallel — not applicable to ~5M T5",
    "pp": "UNSUPPORTED boundary: Pipeline Parallel — not applicable",
    "ep": "UNSUPPORTED boundary: Expert Parallel — model is not MoE",
    "sp": "UNSUPPORTED boundary: Sequence Parallel — seq len too short",
}

StrategyName = str

SUPPORTED_RUNNABLE: List[str] = ["single", "dp", "overalloc"]
BOUNDARY_UNSUPPORTED: List[str] = ["tp", "pp", "ep", "sp"]
