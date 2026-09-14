# src/metrics.py
from __future__ import annotations
from typing import Dict, List
import numpy as np


def moving_average(x: List[float], w: int = 50) -> float:
    if not x:
        return 0.0
    w = min(len(x), w)
    return float(np.mean(x[-w:]))


def episode_stats(rewards: List[float], infos: List[Dict]) -> Dict[str, float]:
    risk = [float(d.get("risk", 0.0)) for d in infos]
    early = [float(d.get("early_gain", 0.0)) for d in infos]
    fp = [float(d.get("fp_risk", 0.0)) for d in infos]
    cost = [float(d.get("act_cost", 0.0)) for d in infos]

    return {
        "reward_mean_200": moving_average(rewards, 200),
        "risk_mean_200": moving_average(risk, 200),
        "early_rate_200": float(np.mean(early[-200:])) if len(early) else 0.0,
        "fp_mean_200": moving_average(fp, 200),
        "cost_mean_200": moving_average(cost, 200),
    }


def plan_rank_stats(plan_ranks: List[int], n_candidates: int) -> Dict[str, float]:
    if not plan_ranks:
        return {"top1_ratio": 0.0, "top2_ratio": 0.0, "avg_rank": 0.0}

    rank_counts = np.bincount(plan_ranks, minlength=n_candidates)
    total = len(plan_ranks)

    top1_ratio = float(rank_counts[0] / total) if total > 0 else 0.0
    top2_ratio = float((rank_counts[0] + rank_counts[1]) / total) if total > 0 else 0.0
    avg_rank = float(np.mean(plan_ranks))

    return {
        "top1_ratio": top1_ratio,
        "top2_ratio": top2_ratio,
        "avg_rank": avg_rank,
    }
