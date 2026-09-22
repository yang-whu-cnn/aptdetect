from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np


HIDDEN_FIELDS = frozenset({"compromised", "true_compromise", "red_session", "ground_truth"})


def _d27(state: Sequence[float]) -> np.ndarray:
    value = np.asarray(state, dtype=np.float64)
    if value.shape != (27,) or not np.isfinite(value).all():
        raise ValueError("progress input must be a finite Blue-visible D27 vector")
    return value


class D27ProgressEnsemble:
    """Injectable scorer core. Production models must be train-replay artifacts."""

    def __init__(self, members: Iterable[Callable[[np.ndarray], float]]):
        self.members = tuple(members)
        if not self.members:
            raise ValueError("progress ensemble cannot be empty")

    def score(self, state: Sequence[float]) -> tuple[float, float]:
        x = _d27(state)
        predictions = np.asarray([float(member(x.copy())) for member in self.members])
        if not np.isfinite(predictions).all():
            raise ValueError("progress ensemble returned a non-finite value")
        if (predictions < 0).any() or (predictions > 1).any():
            raise ValueError("progress potential members must return values in [0,1]")
        return float(predictions.mean()), float(predictions.var())


class OfflinePrior:
    """Read-only prior adapter; a miss has no online-generation fallback."""

    def __init__(self, lookup: Callable[[tuple[float, ...]], Sequence[float] | None]):
        self.lookup = lookup

    def probabilities(self, state: Sequence[float], actions: Sequence[int]) -> np.ndarray:
        key = tuple(float(x) for x in _d27(state))
        raw = self.lookup(key)
        if raw is None:
            raise RuntimeError("offline UAMCTS prior cache miss (fail closed)")
        full = np.asarray(raw, dtype=np.float64)
        if full.shape != (4,) or not np.isfinite(full).all() or (full < 0).any():
            raise ValueError("offline prior must be a finite non-negative A4 vector")
        selected = full[np.asarray(actions, dtype=int)]
        total = float(selected.sum())
        if total <= 0:
            raise ValueError("offline prior has zero mass on available actions")
        return selected / total


@dataclass(frozen=True)
class ProductionGate:
    world_model_frozen: bool = False
    full_reward_frozen: bool = False
    progress_train_only_artifact: bool = False
    progress_calibration_frozen: bool = False
    offline_prior_preflight_passed: bool = False

    def require_ready(self) -> None:
        missing = [name for name, value in self.__dict__.items() if not value]
        if missing:
            raise RuntimeError("UAMCTS production gate blocked: " + ", ".join(missing))
