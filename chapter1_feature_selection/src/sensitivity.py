from __future__ import annotations
from typing import Dict, List, Tuple

import numpy as np

from .clfs import encode_windows, MLPEncoder
from .preprocess import FeatureMeta


def compute_field_sensitivity(
    encoder: MLPEncoder,
    X: np.ndarray,
    meta: FeatureMeta,
    max_windows: int = 5000,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    输出：field -> raw sensitivity
    定义：sensitivity = mean(||z - z_mask||_2)
    """
    if len(X) > max_windows:
        idx = np.random.choice(len(X), size=max_windows, replace=False)
        X_use = X[idx]
    else:
        X_use = X

    Z_base = encode_windows(encoder, X_use, device=device)

    scores: Dict[str, float] = {}
    for field in meta.fields_order:
        start, end = meta.field_slices[field]
        X_mask = X_use.copy()
        X_mask[:, start:end] = 0.0
        Z_mask = encode_windows(encoder, X_mask, device=device)
        diff = np.linalg.norm(Z_base - Z_mask, axis=1).mean()
        scores[field] = float(diff)

    return scores


def _field_dim(field_slices: Dict[str, Tuple[int, int]], field: str) -> int:
    s, e = field_slices[field]
    return int(e - s)


def _adjust_score(raw_score: float, dim: int, score_norm: str) -> float:
    if score_norm == "sqrt_dim":
        return float(raw_score) / max(np.sqrt(float(dim)), 1.0)
    if score_norm == "dim":
        return float(raw_score) / max(float(dim), 1.0)
    return float(raw_score)


def compute_field_utility_scores(
    raw_scores: Dict[str, float],
    field_slices: Dict[str, Tuple[int, int]],
    score_norm: str = "none",
) -> Dict[str, float]:
    """
    计算最终用于候选池筛选的效用分数（敏感度 / 维度代价）
    """
    utility = {}
    for f, raw in raw_scores.items():
        dim = _field_dim(field_slices, f)
        utility[f] = float(_adjust_score(raw, dim, score_norm))
    return utility


def build_candidate_pool(
    scores: Dict[str, float],
    field_slices: Dict[str, Tuple[int, int]],
    top_ratio: float = 0.1,
    min_fields: int = 5,
    max_candidate_dim: int | None = None,
    score_norm: str = "none",
    min_raw_score_ratio: float = 0.0,
) -> List[str]:
    """
    候选池构造逻辑：
    1) 先算 raw sensitivity
    2) 再算 utility = sensitivity / cost
    3) 用 min_raw_score_ratio 过滤掉原始敏感度过低的字段
    4) 在维度预算 max_candidate_dim 下累计选择
    """
    items = []
    for field, raw_score in scores.items():
        dim = _field_dim(field_slices, field)
        adjusted = _adjust_score(raw_score, dim, score_norm)
        items.append((field, float(raw_score), int(dim), float(adjusted)))

    # 先按 utility 排序，再按 raw score 排
    items = sorted(items, key=lambda x: (x[3], x[1]), reverse=True)

    # 原始敏感度过滤
    if len(items) > 0 and float(min_raw_score_ratio) > 0:
        max_raw = max(x[1] for x in items)
        threshold = float(max_raw) * float(min_raw_score_ratio)
        eligible = [x for x in items if x[1] >= threshold]
        # 如果过滤后太少，则退回原列表
        if len(eligible) < int(min_fields):
            eligible = items[:]
    else:
        eligible = items[:]

    if max_candidate_dim is None or int(max_candidate_dim) <= 0:
        top_k = max(int(min_fields), int(len(eligible) * float(top_ratio)))
        return [field for field, _, _, _ in eligible[:top_k]]

    selected: List[str] = []
    used_dim = 0
    budget = int(max_candidate_dim)

    # 先在过滤后的候选中按预算选
    for field, raw_score, dim, adjusted in eligible:
        if len(selected) < int(min_fields):
            selected.append(field)
            used_dim += int(dim)
            continue
        if used_dim + int(dim) <= budget:
            selected.append(field)
            used_dim += int(dim)

    # 兜底：极端情况下至少保留 min_fields
    if len(selected) < int(min_fields):
        for field, raw_score, dim, adjusted in items:
            if field in selected:
                continue
            selected.append(field)
            if len(selected) >= int(min_fields):
                break

    if len(selected) == 0 and len(items) > 0:
        selected.append(items[0][0])

    return selected