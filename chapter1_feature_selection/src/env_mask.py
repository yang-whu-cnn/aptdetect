from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from src.detector import Detector
from src.metrics import safe_auprc, fp_per_hour, sparsity_ratio


@dataclass
class PRFSEnvConfig:
    window_size_sec: int
    k_flip: int
    tau_soft_update: float
    early_horizon_steps: int

    lambda_sparse: float
    target_keep_ratio: float
    lambda_fp: float
    lambda_lag: float
    hour_window_steps: int


class PRFSEnv:
    """
    PRF-S 环境：回放式在线
    - 输入：X_F (T,D), y (T,)
    - 掩码作用：按列开关
    """
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        detector: Detector,
        cfg: PRFSEnvConfig,
    ):
        assert len(X) == len(y)
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.detector = detector
        self.cfg = cfg

        self.T, self.D = self.X.shape
        self.mask_prob = np.ones(self.D, dtype=np.float32)
        self.mask_hard = (self.mask_prob >= 0.5).astype(np.float32)
        self.t = 0

        self.hist_scores: List[float] = []
        self.hist_labels: List[int] = []
        self.hist_pred: List[int] = []

        self.in_positive_segment = False
        self.segment_start = 0
        self.first_detect = -1
        self.prev_auprc = 0.0

    def reset(self) -> np.ndarray:
        self.t = 0
        self.mask_prob[:] = 1.0
        self.mask_hard = (self.mask_prob >= 0.5).astype(np.float32)

        self.hist_scores.clear()
        self.hist_labels.clear()
        self.hist_pred.clear()

        self.in_positive_segment = False
        self.segment_start = 0
        self.first_detect = -1
        self.prev_auprc = 0.0
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        x = self.X[self.t] * self.mask_hard
        eps = float(self.detector.uncertainty(x.reshape(1, -1))[0])
        state = np.concatenate([[eps], self.mask_prob], axis=0).astype(np.float32)
        return state

    def step(self, flip_indices: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        hard = (self.mask_prob >= 0.5).astype(np.float32)
        for idx in flip_indices[: self.cfg.k_flip]:
            if 0 <= idx < self.D:
                hard[idx] = 1.0 - hard[idx]

        tau = self.cfg.tau_soft_update
        self.mask_prob = (1 - tau) * self.mask_prob + tau * hard
        self.mask_hard = (self.mask_prob >= 0.5).astype(np.float32)

        x = self.X[self.t] * self.mask_hard
        score = float(self.detector.predict_proba(x.reshape(1, -1))[0])
        pred = int(score >= self.detector.threshold)
        label = int(self.y[self.t])

        self.hist_scores.append(score)
        self.hist_labels.append(label)
        self.hist_pred.append(pred)
        if len(self.hist_scores) > self.cfg.early_horizon_steps:
            self.hist_scores.pop(0)
            self.hist_labels.pop(0)
        if len(self.hist_pred) > self.cfg.hour_window_steps:
            self.hist_pred.pop(0)

        auprc_now = safe_auprc(np.array(self.hist_labels), np.array(self.hist_scores))
        delta_auprc = auprc_now - self.prev_auprc
        self.prev_auprc = auprc_now

        keep_ratio = float(sparsity_ratio(self.mask_hard))
        sparse_excess = max(0.0, keep_ratio - float(self.cfg.target_keep_ratio))
        sparse_pen = self.cfg.lambda_sparse * sparse_excess

        fp_rate = fp_per_hour(
            y_true=np.array(self.hist_labels[-len(self.hist_pred):]),
            y_pred=np.array(self.hist_pred),
            window_size_sec=self.cfg.window_size_sec,
        )
        fp_pen = self.cfg.lambda_fp * fp_rate
        lag_pen = self._lag_penalty(label, pred)

        reward = float(delta_auprc - sparse_pen - fp_pen - lag_pen)

        self.t += 1
        done = (self.t >= self.T)
        info = {
            "delta_auprc": float(delta_auprc),
            "auprc_now": float(auprc_now),
            "sparsity": keep_ratio,
            "kept_dim": int(self.mask_hard.sum()),
            "fp_per_hour": float(fp_rate),
            "lag_pen": float(lag_pen),
        }

        if done:
            next_state = np.zeros(1 + self.D, dtype=np.float32)
        else:
            next_state = self._get_state()
        return next_state, reward, done, info

    def _lag_penalty(self, label: int, pred: int) -> float:
        if label == 1 and not self.in_positive_segment:
            self.in_positive_segment = True
            self.segment_start = self.t
            self.first_detect = -1

        if self.in_positive_segment:
            if pred == 1 and self.first_detect < 0:
                self.first_detect = self.t
            if label == 0:
                self.in_positive_segment = False
                self.segment_start = 0
                self.first_detect = -1
                return 0.0
            if (self.t - self.segment_start) > self.cfg.early_horizon_steps and self.first_detect < 0:
                return float(self.cfg.lambda_lag)
        return 0.0