from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import torch

from baselines.ug_cem_apt.planner import UGCEMPlanner
from shared.formal_state import FORMAL_STATE_DIM


ALLOWED_WARMUP_SEEDS = {
    "train": frozenset(range(1000, 1032)),
    "calibration": frozenset(range(3000, 3008)),
}


@dataclass(frozen=True)
class UGWarmupSample:
    """One planner-visible state used for uncertainty calibration."""

    state: object
    split: str
    episode_seed: int
    agent_name: str


@dataclass(frozen=True)
class UGWarmupConfig:
    """
    Step-6 warm-up policy.

    num_planner_calls:
        Taskbook default is 100 planner calls.

    freeze_after_warmup:
        False = source-faithful main version, keep EMA online.
        True  = sensitivity version, stop EMA updates after warm-up.
    """

    num_planner_calls: int = 100
    freeze_after_warmup: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.num_planner_calls, bool)
            or not isinstance(self.num_planner_calls, int)
            or self.num_planner_calls <= 0
        ):
            raise ValueError("num_planner_calls must be a positive int")
        if not isinstance(self.freeze_after_warmup, bool):
            raise TypeError("freeze_after_warmup must be bool")


@dataclass(frozen=True)
class UGWarmupResult:
    num_planner_calls: int
    freeze_after_warmup: bool
    final_update_mode: bool
    warm_start_used_count: int
    splits: Tuple[str, ...]
    episode_seeds: Tuple[int, ...]
    agent_names: Tuple[str, ...]
    obs_mean: torch.Tensor
    obs_std: torch.Tensor
    horizon_std: torch.Tensor
    all_finite: bool


class UGUncertaintyWarmup:
    """
    Calibrates one planner's own UGUncertainty normalizer.

    A separate UGCEMPlanner/UGUncertainty instance should be used for
    each agent/planner whose model distribution is calibrated.

    Every warm-up planner call starts from uniform CEM probabilities by
    clearing only the MPC warm-start cache. Uncertainty statistics are
    reset once at the start, then updated continuously across calls.
    """

    def __init__(
        self,
        *,
        planner: UGCEMPlanner,
        expected_agent_name: str,
        config: UGWarmupConfig | None = None,
    ):
        if not isinstance(planner, UGCEMPlanner):
            raise TypeError("planner must be UGCEMPlanner")

        expected_agent_name = str(expected_agent_name).strip()
        if not expected_agent_name:
            raise ValueError("expected_agent_name must not be empty")

        self.planner = planner
        self.expected_agent_name = expected_agent_name
        self.config = config if config is not None else UGWarmupConfig()

    @staticmethod
    def _validate_sample(sample: UGWarmupSample) -> np.ndarray:
        if not isinstance(sample, UGWarmupSample):
            raise TypeError("warm-up records must be UGWarmupSample")

        split = str(sample.split)
        if split not in ALLOWED_WARMUP_SEEDS:
            raise ValueError(
                "uncertainty warm-up only permits train or calibration split"
            )

        seed = int(sample.episode_seed)
        if seed not in ALLOWED_WARMUP_SEEDS[split]:
            raise ValueError(
                f"episode_seed={seed} is not in frozen {split} seed protocol"
            )

        agent_name = str(sample.agent_name).strip()
        if not agent_name:
            raise ValueError("agent_name must not be empty")

        state = np.asarray(sample.state, dtype=np.float32)
        if state.shape != (FORMAL_STATE_DIM,):
            raise ValueError(f"state must have shape ({FORMAL_STATE_DIM},)")
        if not np.isfinite(state).all():
            raise ValueError("warm-up state must be finite")

        return state

    def _snapshot(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        uncertainty = self.planner.uncertainty
        if (
            uncertainty.obs_mean is None
            or uncertainty.obs_std is None
            or uncertainty.horizon_std is None
        ):
            raise RuntimeError("uncertainty normalizer was not initialized")

        obs_mean = uncertainty.obs_mean.detach().clone()
        obs_std = uncertainty.obs_std.detach().clone()
        horizon_std = uncertainty.horizon_std.detach().clone()

        for name, value in (
            ("obs_mean", obs_mean),
            ("obs_std", obs_std),
            ("horizon_std", horizon_std),
        ):
            if not torch.isfinite(value).all():
                raise RuntimeError(f"{name} contains non-finite values")

        return obs_mean, obs_std, horizon_std

    def run(self, samples: Iterable[UGWarmupSample]) -> UGWarmupResult:
        iterator = iter(samples)

        # Source-faithful calibration starts from UG official defaults.
        self.planner.reset_uncertainty_stats()
        self.planner.set_uncertainty_update_mode(True)
        self.planner.reset_episode()

        warm_start_used_count = 0
        splits: list[str] = []
        seeds: list[int] = []
        agents: list[str] = []

        for _call_idx in range(self.config.num_planner_calls):
            try:
                sample = next(iterator)
            except StopIteration as exc:
                raise ValueError(
                    "not enough warm-up samples for requested planner calls"
                ) from exc

            state = self._validate_sample(sample)

            if str(sample.agent_name).strip() != self.expected_agent_name:
                raise ValueError(
                    "warm-up sample agent does not match planner instance"
                )

            # No previous_solution / MPC warm-start during calibration.
            self.planner.reset_episode()
            if self.planner.warm_start_probs is not None:
                raise RuntimeError("warm-start cache was not cleared")

            result = self.planner.plan(state)
            if result.warm_start_used:
                warm_start_used_count += 1
                raise RuntimeError("warm-up planner call used previous solution")

            self._snapshot()

            splits.append(str(sample.split))
            seeds.append(int(sample.episode_seed))
            agents.append(str(sample.agent_name).strip())

        final_update_mode = not self.config.freeze_after_warmup
        self.planner.set_uncertainty_update_mode(final_update_mode)
        self.planner.reset_episode()

        obs_mean, obs_std, horizon_std = self._snapshot()

        return UGWarmupResult(
            num_planner_calls=int(self.config.num_planner_calls),
            freeze_after_warmup=bool(self.config.freeze_after_warmup),
            final_update_mode=bool(final_update_mode),
            warm_start_used_count=int(warm_start_used_count),
            splits=tuple(sorted(set(splits))),
            episode_seeds=tuple(sorted(set(seeds))),
            agent_names=tuple(sorted(set(agents))),
            obs_mean=obs_mean,
            obs_std=obs_std,
            horizon_std=horizon_std,
            all_finite=True,
        )
