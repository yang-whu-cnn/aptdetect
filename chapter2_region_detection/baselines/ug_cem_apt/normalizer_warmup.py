from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Integral
from typing import Iterable, Tuple

import numpy as np
import torch

from baselines.ug_cem_apt.planner import UGCEMPlanner
from shared.formal_state import FORMAL_STATE_DIM


ALLOWED_WARMUP_SPLITS = frozenset({"train", "calibration"})
TRAIN_SEEDS = frozenset(range(1000, 1032))
CALIBRATION_SEEDS = frozenset(range(3000, 3008))


@dataclass(frozen=True)
class WarmupState:
    """Planner-visible state with provenance for Step-6 calibration."""

    state: np.ndarray
    split: str
    episode_seed: int
    agent_name: str

    def __post_init__(self) -> None:
        split = str(self.split)
        if split not in ALLOWED_WARMUP_SPLITS:
            raise ValueError(
                "warm-up split must be train or calibration; "
                f"got {split!r}"
            )

        state = np.asarray(self.state, dtype=np.float32)
        if state.shape != (FORMAL_STATE_DIM,):
            raise ValueError(
                f"warm-up state must have shape ({FORMAL_STATE_DIM},)"
            )
        if not np.isfinite(state).all():
            raise ValueError("warm-up state must be finite")

        if (
            isinstance(self.episode_seed, bool)
            or not isinstance(self.episode_seed, Integral)
        ):
            raise ValueError("episode_seed must be integer")

        episode_seed = int(self.episode_seed)
        expected_seeds = TRAIN_SEEDS if split == "train" else CALIBRATION_SEEDS
        if episode_seed not in expected_seeds:
            raise ValueError(
                f"episode_seed={episode_seed} is not valid for split={split!r}"
            )

        agent_name = str(self.agent_name).strip()
        if not agent_name:
            raise ValueError("agent_name must not be empty")

        object.__setattr__(self, "state", state.copy())
        object.__setattr__(self, "split", split)
        object.__setattr__(self, "episode_seed", episode_seed)
        object.__setattr__(self, "agent_name", agent_name)


@dataclass(frozen=True)
class UGNormalizerWarmupConfig:
    """Step-6 source-faithful calibration policy."""

    planner_calls: int = 100
    freeze_after_warmup: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.planner_calls, bool)
            or not isinstance(self.planner_calls, int)
            or self.planner_calls <= 0
        ):
            raise ValueError("planner_calls must be a positive integer")
        if not isinstance(self.freeze_after_warmup, bool):
            raise TypeError("freeze_after_warmup must be bool")


@dataclass(frozen=True)
class UGNormalizerSnapshot:
    obs_mean: torch.Tensor
    obs_std: torch.Tensor
    horizon_std: torch.Tensor


@dataclass(frozen=True)
class UGNormalizerWarmupReport:
    planner_calls: int
    train_calls: int
    calibration_calls: int
    unique_episode_seeds: Tuple[int, ...]
    agent_name: str
    all_warm_starts_disabled: bool
    finite: bool
    freeze_after_warmup: bool
    online_updates_after_warmup: bool
    snapshot: UGNormalizerSnapshot


class UGNormalizerWarmup:
    """
    Calibrate one planner's UG uncertainty running statistics.

    Contract:
      - input provenance may only be the frozen train/calibration seeds;
      - default warm-up length is 100 planner calls;
      - one runner is bound to exactly one agent/planner normalizer;
      - every call starts without MPC previous solution;
      - only planner-visible 27D states are consumed;
      - the planner's existing UGUncertainty object is updated in place;
      - normal episode reset does not clear the uncertainty statistics;
      - source-faithful main mode continues online EMA after warm-up;
      - freeze_after_warmup=True is a sensitivity mode that disables
        subsequent online EMA updates without deleting calibrated stats.
    """

    def __init__(
        self,
        *,
        planner: UGCEMPlanner,
        config: UGNormalizerWarmupConfig | None = None,
    ):
        if not isinstance(planner, UGCEMPlanner):
            raise TypeError("planner must be UGCEMPlanner")

        self.planner = planner
        self.config = (
            config if config is not None else UGNormalizerWarmupConfig()
        )

    @staticmethod
    def _snapshot(planner: UGCEMPlanner) -> UGNormalizerSnapshot:
        uncertainty = planner.uncertainty

        if (
            uncertainty.obs_mean is None
            or uncertainty.obs_std is None
            or uncertainty.horizon_std is None
        ):
            raise RuntimeError("uncertainty normalizer was not initialized")

        tensors = (
            uncertainty.obs_mean,
            uncertainty.obs_std,
            uncertainty.horizon_std,
        )
        if not all(torch.isfinite(item).all() for item in tensors):
            raise RuntimeError("uncertainty normalizer contains NaN/Inf")

        return UGNormalizerSnapshot(
            obs_mean=uncertainty.obs_mean.detach().clone(),
            obs_std=uncertainty.obs_std.detach().clone(),
            horizon_std=uncertainty.horizon_std.detach().clone(),
        )

    def run(
        self,
        states: Iterable[WarmupState],
    ) -> UGNormalizerWarmupReport:
        records = list(states)

        if len(records) < self.config.planner_calls:
            raise ValueError(
                "not enough warm-up states: "
                f"need {self.config.planner_calls}, got {len(records)}"
            )

        selected = records[: self.config.planner_calls]

        for item in selected:
            if not isinstance(item, WarmupState):
                raise TypeError("states must contain WarmupState items")

        agent_names = {item.agent_name for item in selected}
        if len(agent_names) != 1:
            raise ValueError(
                "one warm-up runner must contain states from exactly one agent"
            )
        agent_name = next(iter(agent_names))

        # Warm-up itself must update EMA irrespective of the post-warm-up
        # sensitivity setting. Preserve beta; change only update policy.
        self.planner.config = replace(
            self.planner.config,
            update_uncertainty_stats=True,
        )

        train_calls = 0
        calibration_calls = 0
        seeds = set()
        all_no_warm_start = True

        for item in selected:
            # Required by taskbook: warm-up does not use previous_solution.
            self.planner.reset_episode()
            result = self.planner.plan(item.state)

            if bool(result.warm_start_used):
                all_no_warm_start = False
                raise RuntimeError("warm-up unexpectedly used MPC warm-start")

            if item.split == "train":
                train_calls += 1
            elif item.split == "calibration":
                calibration_calls += 1
            else:  # defensive: WarmupState already rejects this
                raise RuntimeError("forbidden warm-up split")

            seeds.add(int(item.episode_seed))

        snapshot = self._snapshot(self.planner)

        # Do not carry the final warm-up CEM solution into the first real
        # decision. Statistics remain calibrated in the uncertainty object.
        self.planner.reset_episode()

        online_updates = not self.config.freeze_after_warmup
        self.planner.config = replace(
            self.planner.config,
            update_uncertainty_stats=online_updates,
        )

        finite = all(
            torch.isfinite(item).all().item()
            for item in (
                snapshot.obs_mean,
                snapshot.obs_std,
                snapshot.horizon_std,
            )
        )

        if not finite:
            raise RuntimeError("non-finite warm-up snapshot")

        return UGNormalizerWarmupReport(
            planner_calls=len(selected),
            train_calls=train_calls,
            calibration_calls=calibration_calls,
            unique_episode_seeds=tuple(sorted(seeds)),
            agent_name=agent_name,
            all_warm_starts_disabled=all_no_warm_start,
            finite=bool(finite),
            freeze_after_warmup=bool(self.config.freeze_after_warmup),
            online_updates_after_warmup=bool(online_updates),
            snapshot=snapshot,
        )
