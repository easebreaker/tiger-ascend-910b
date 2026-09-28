"""Allocation / parallel strategies for sched bench (inference-first)."""

from __future__ import annotations

from typing import Dict, List

# Inference-first strategies for multi-chip *scheduling* experiments.
# Small TIGER (~5M) does not use model-parallel serving; replicas = multi-copy.
STRATEGIES: Dict[str, str] = {
    "single": (
        "1-card inference baseline: full model replica, measure latency/QPS/HBM"
    ),
    "replicas": (
        "Multi-replica serving (data-parallel at request level): N cards each hold "
        "a full copy and serve a shard of requests — the practical scale-out for small models"
    ),
    "overalloc": (
        "Reserve N cards via torchrun but only rank0 serves; others idle. "
        "Measures scheduler waste when small inference jobs are over-granted chips"
    ),
    "train_profile": (
        "Optional short training-step profile (secondary). Not the primary scheduling signal."
    ),
    "tp": "UNSUPPORTED boundary: Tensor Parallel serving — not for ~5M T5",
    "pp": "UNSUPPORTED boundary: Pipeline Parallel serving — not applicable",
    "ep": "UNSUPPORTED boundary: Expert Parallel — model is not MoE",
    "sp": "UNSUPPORTED boundary: Sequence Parallel — seq too short",
}

StrategyName = str

# Primary matrix for multi-chip scheduling claims
SUPPORTED_INFER: List[str] = ["single", "replicas", "overalloc"]
# Optional secondary
SUPPORTED_TRAIN: List[str] = ["train_profile"]
SUPPORTED_RUNNABLE: List[str] = SUPPORTED_INFER + SUPPORTED_TRAIN
BOUNDARY_UNSUPPORTED: List[str] = ["tp", "pp", "ep", "sp"]
