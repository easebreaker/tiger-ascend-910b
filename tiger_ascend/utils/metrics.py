"""Training metrics helpers + paper Beauty reference numbers."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

# Rajput et al. Table 1 — Amazon Beauty (TIGER row).
PAPER_BEAUTY_METRICS: Dict[str, float] = {
    "recall@5": 0.0454,
    "ndcg@5": 0.0321,
    "recall@10": 0.0648,
    "ndcg@10": 0.0384,
}


def ndcg_at_k(rank: Optional[int], k: int) -> float:
    """Leave-one-out NDCG@K; ``rank`` is 0-based position or None if miss."""
    if rank is None or rank >= k:
        return 0.0
    return 1.0 / math.log2(rank + 2)


def rank_of_gold(gold: str, preds: Sequence[str]) -> Optional[int]:
    try:
        return list(preds).index(gold)
    except ValueError:
        return None


def summarize_ranking(
    golds: Sequence[str],
    pred_lists: Sequence[Sequence[str]],
    ks: Sequence[int],
) -> Dict[str, float]:
    """Aggregate Recall@K (=Hit@K for single GT) and NDCG@K."""
    assert len(golds) == len(pred_lists)
    n = max(1, len(golds))
    out: Dict[str, float] = {}
    for k in ks:
        hits = 0
        ndcg_sum = 0.0
        for gold, preds in zip(golds, pred_lists):
            r = rank_of_gold(gold, preds[:k])
            hits += int(r is not None)
            ndcg_sum += ndcg_at_k(r, k)
        out[f"recall@{k}"] = hits / n
        out[f"ndcg@{k}"] = ndcg_sum / n
        out[f"hit@{k}"] = out[f"recall@{k}"]
    return out


def format_vs_paper(metrics: Dict[str, float], paper: Dict[str, float] = None) -> List[str]:
    paper = paper or PAPER_BEAUTY_METRICS
    lines = []
    for key, ref in paper.items():
        got = metrics.get(key)
        if got is None:
            continue
        delta = got - ref
        lines.append(f"{key}: ours={got:.4f} paper={ref:.4f} delta={delta:+.4f}")
    return lines
