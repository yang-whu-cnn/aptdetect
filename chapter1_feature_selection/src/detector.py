from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class Detector:
    model: LogisticRegression
    threshold: float = 0.5

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        # 对二分类 LogisticRegression 直接走线性层，明显快于 sklearn 的逐次调用
        if hasattr(self.model, "coef_") and hasattr(self.model, "intercept_"):
            coef = np.asarray(self.model.coef_[0], dtype=np.float32)
            intercept = float(np.asarray(self.model.intercept_).reshape(-1)[0])
            logits = X @ coef + intercept
            return _sigmoid(logits).reshape(-1)
        return self.model.predict_proba(X)[:, 1]

    def predict_alert(self, X: np.ndarray) -> np.ndarray:
        p = self.predict_proba(X)
        return (p >= self.threshold).astype(np.int64)

    def uncertainty(self, X: np.ndarray) -> np.ndarray:
        p = self.predict_proba(X)
        return (4.0 * p * (1.0 - p)).astype(np.float32)


def train_logreg_detector(X, y, threshold=0.5, class_weight=None, C=1.0, max_iter=1000):
    clf = LogisticRegression(
        solver="liblinear",
        max_iter=int(max_iter),
        class_weight=class_weight,
        C=float(C),
    )
    clf.fit(X, y)
    return Detector(clf, threshold=threshold)