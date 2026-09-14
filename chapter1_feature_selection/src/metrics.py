# src/metrics.py
from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
from sklearn.metrics import average_precision_score


def safe_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # 全0或全1时AUPRC可能异常，这里做保护
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(average_precision_score(y_true, y_score))


def fp_per_hour(y_true, y_pred, window_size_sec: int):
    """
    每小时误报数 FP/h
    修复点：y_true 和 y_pred 长度可能不同，统一对齐到 min_len
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    n = min(len(y_true), len(y_pred))
    if n <= 0:
        return 0.0

    # 对齐到最后 n 个（保证时间一致）
    y_true = y_true[-n:]
    y_pred = y_pred[-n:]

    fp = int(((y_true == 0) & (y_pred == 1)).sum())

    # 覆盖时长（小时）
    hours = (n * float(window_size_sec)) / 3600.0
    if hours <= 1e-9:
        return 0.0

    return fp / hours


def sparsity_ratio(mask: np.ndarray) -> float:
    """
    稀疏度：启用比例（越低越省资源）
    """
    return float(mask.mean())
