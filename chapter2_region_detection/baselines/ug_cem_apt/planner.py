from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from time import perf_counter
from typing import Tuple

import torch

from baselines.ug_cem_apt.categorical_cem import (
    CategoricalCEMOptimizer,
)
from baselines.ug_cem_apt.uncertainty import UGUncertainty
from shared.action_contract import N_ACTIONS
from shared.formal_state import FORMAL_STATE_DIM


FORMAL_HORIZON = 4
FORMAL_ENSEMBLE_SIZE = 5


@dataclass(frozen=True)
class UGCEMPlannerConfig:
    """
    Step-5 integration config.

    beta=0.1 is a development/default value only.
    Step 9 owns final validation-time beta selection.
    CEM-APT uses the same planner with beta=0.
    """

    beta: float = 0.10
    update_uncertainty_stats: bool = True

    def __post_init__(self) -> None:
        beta = float(self.beta)
        if not isfinite(beta) or beta < 0.0:
            raise ValueError("beta must be finite and >= 0")
        if not isinstance(self.update_uncertainty_stats, bool):
            raise TypeError("update_uncertainty_stats must be bool")


@dataclass(frozen=True)
class UGCEMIterationRecord:
    iteration: int
    best_score: float
    best_expected_return: float
    best_uncertainty: float
    mean_score: float
    mean_expected_return: float
    mean_uncertainty: float


@dataclass(frozen=True)
class UGCEMPlanResult:
    action_id: int
    best_plan: torch.Tensor
    best_score: float
    expected_return: float
    uncertainty: float
    final_probs: torch.Tensor
    final_probs_entropy: float
    final_probs_entropy_by_step: torch.Tensor
    iteration_history: Tuple[UGCEMIterationRecord, ...]
    planning_latency_sec: float
    warm_start_used: bool


class UGCEMPlanner:
    """
    Uncertainty-Guided Categorical CEM planner.

    Composition:
      CategoricalCEMOptimizer
      + SharedRolloutEvaluator
      + UGUncertainty
      + MPC probability warm-start.

    Score at zero-based CEM iteration k:

        J = expected_return - beta * uncertainty / (k + 1)

    Every call replans a full H=4 sequence, returns the best sampled
    plan, executes only best_plan[0] externally, and stores a shifted
    final categorical probability matrix for the next decision epoch.

    Episode reset clears only the MPC warm-start cache. This matches
    the source TrajectoryOptimizer reset semantics; uncertainty running
    statistics are intentionally preserved unless reset_uncertainty_stats()
    is called explicitly (Step 6 controls warm-up/freeze policy).
    """

    def __init__(
        self,
        *,
        cem_optimizer: CategoricalCEMOptimizer,
        rollout_evaluator,
        uncertainty: UGUncertainty,
        config: UGCEMPlannerConfig | None = None,
    ):
        self.cem_optimizer = cem_optimizer
        self.rollout_evaluator = rollout_evaluator
        self.uncertainty = uncertainty
        self.config = config if config is not None else UGCEMPlannerConfig()

        self._validate_components()
        self.device = torch.device(self.cem_optimizer.device)
        self._warm_start_probs: torch.Tensor | None = None
        self._update_uncertainty_stats = bool(
            self.config.update_uncertainty_stats
        )

    def _validate_components(self) -> None:
        cem_cfg = self.cem_optimizer.cfg

        if int(cem_cfg.horizon) != FORMAL_HORIZON:
            raise ValueError("formal UGCEM horizon must be 4")
        if int(cem_cfg.n_actions) != N_ACTIONS:
            raise ValueError("formal UGCEM n_actions must be 4")

        if not hasattr(self.rollout_evaluator, "config"):
            raise TypeError("rollout_evaluator must expose config")
        if not hasattr(self.rollout_evaluator, "device"):
            raise TypeError("rollout_evaluator must expose device")
        if not hasattr(self.rollout_evaluator, "ensemble_size"):
            raise TypeError("rollout_evaluator must expose ensemble_size")
        if not hasattr(self.rollout_evaluator, "evaluate"):
            raise TypeError("rollout_evaluator must expose evaluate")

        if int(self.rollout_evaluator.config.horizon) != FORMAL_HORIZON:
            raise ValueError("rollout evaluator horizon must be 4")
        if int(self.rollout_evaluator.ensemble_size) != FORMAL_ENSEMBLE_SIZE:
            raise ValueError("formal UGCEM ensemble size must be 5")

        if not isinstance(self.uncertainty, UGUncertainty):
            raise TypeError("uncertainty must be UGUncertainty")

        cem_device = torch.device(self.cem_optimizer.device)
        rollout_device = torch.device(self.rollout_evaluator.device)
        uncertainty_device = torch.device(self.uncertainty.device)

        if not (cem_device == rollout_device == uncertainty_device):
            raise ValueError("CEM, rollout evaluator, and uncertainty must share device")

    @property
    def update_uncertainty_stats(self) -> bool:
        return bool(self._update_uncertainty_stats)

    def set_uncertainty_update_mode(self, enabled: bool) -> None:
        """
        Runtime control used by Step 6 warm-up/freeze policy.

        This does not change beta, CEM, model, or reward semantics.
        """
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be bool")
        self._update_uncertainty_stats = enabled

    @property
    def warm_start_probs(self) -> torch.Tensor | None:
        if self._warm_start_probs is None:
            return None
        return self._warm_start_probs.detach().clone()

    def reset_episode(self) -> None:
        """Clear MPC previous-solution cache only."""
        self._warm_start_probs = None

    def reset(self) -> None:
        """Alias used by agent-style evaluation harnesses."""
        self.reset_episode()

    def reset_uncertainty_stats(self) -> None:
        """Explicit reset; not part of normal episode reset."""
        self.uncertainty.reset()

    def _shift_final_probs(self, final_probs: torch.Tensor) -> torch.Tensor:
        expected = (FORMAL_HORIZON, N_ACTIONS)
        if tuple(final_probs.shape) != expected:
            raise ValueError(f"final_probs must have shape {expected}")
        if not torch.isfinite(final_probs).all():
            raise ValueError("final_probs must be finite")

        shifted = torch.empty_like(final_probs)
        shifted[:-1] = final_probs[1:]
        shifted[-1] = self.cem_optimizer.uniform_probs[-1]
        return shifted.detach().clone()

    @staticmethod
    def _entropy(final_probs: torch.Tensor) -> tuple[float, torch.Tensor]:
        safe = final_probs.clamp_min(1e-12)
        by_step = -(safe * safe.log()).sum(dim=1)
        return float(by_step.mean().item()), by_step.detach().clone()

    def plan(self, state) -> UGCEMPlanResult:
        start = perf_counter()

        initial_probs = self._warm_start_probs
        warm_start_used = initial_probs is not None

        history: list[UGCEMIterationRecord] = []
        debug_best_score = float("-inf")
        debug_best_plan = None
        debug_best_return = None
        debug_best_uncertainty = None

        def objective(plans: torch.Tensor, iteration: int) -> torch.Tensor:
            nonlocal debug_best_score
            nonlocal debug_best_plan
            nonlocal debug_best_return
            nonlocal debug_best_uncertainty

            rollout = self.rollout_evaluator.evaluate(state, plans)

            expected_return = torch.as_tensor(
                rollout.expected_return,
                dtype=torch.float32,
                device=self.device,
            )

            uncertainty = self.uncertainty.compute(
                rollout.next_states,
                update_stats=self._update_uncertainty_stats,
            )

            population = int(plans.shape[0])
            if tuple(expected_return.shape) != (population,):
                raise ValueError("rollout expected_return shape mismatch")
            if tuple(uncertainty.shape) != (population,):
                raise ValueError("uncertainty shape mismatch")
            if not torch.isfinite(expected_return).all():
                raise ValueError("expected_return contains non-finite values")
            if not torch.isfinite(uncertainty).all():
                raise ValueError("uncertainty contains non-finite values")

            score = expected_return - (
                float(self.config.beta)
                * uncertainty
                / float(iteration + 1)
            )

            if not torch.isfinite(score).all():
                raise ValueError("UGCEM score contains non-finite values")

            iter_best_score_t, iter_best_idx_t = torch.max(score, dim=0)
            iter_best_idx = int(iter_best_idx_t.item())
            iter_best_score = float(iter_best_score_t.item())

            history.append(
                UGCEMIterationRecord(
                    iteration=int(iteration),
                    best_score=iter_best_score,
                    best_expected_return=float(expected_return[iter_best_idx].item()),
                    best_uncertainty=float(uncertainty[iter_best_idx].item()),
                    mean_score=float(score.mean().item()),
                    mean_expected_return=float(expected_return.mean().item()),
                    mean_uncertainty=float(uncertainty.mean().item()),
                )
            )

            if iter_best_score > debug_best_score:
                debug_best_score = iter_best_score
                debug_best_plan = plans[iter_best_idx].detach().clone()
                debug_best_return = float(expected_return[iter_best_idx].item())
                debug_best_uncertainty = float(uncertainty[iter_best_idx].item())

            return score

        cem_result = self.cem_optimizer.optimize(
            objective,
            initial_probs=initial_probs,
        )

        if debug_best_plan is None:
            raise RuntimeError("planner objective produced no best plan")

        if not torch.equal(cem_result.best_plan, debug_best_plan):
            raise RuntimeError("CEM best plan disagrees with planner debug tracking")

        if abs(float(cem_result.best_score) - debug_best_score) > 1e-5:
            raise RuntimeError("CEM best score disagrees with planner debug tracking")

        self._warm_start_probs = self._shift_final_probs(cem_result.final_probs)

        entropy, entropy_by_step = self._entropy(cem_result.final_probs)

        latency = float(perf_counter() - start)
        if not isfinite(latency) or latency < 0.0:
            raise RuntimeError("invalid planning latency")

        best_plan = cem_result.best_plan.detach().clone()
        action_id = int(best_plan[0].item())

        if not 0 <= action_id < N_ACTIONS:
            raise RuntimeError("planner returned invalid first action")

        return UGCEMPlanResult(
            action_id=action_id,
            best_plan=best_plan,
            best_score=float(cem_result.best_score),
            expected_return=float(debug_best_return),
            uncertainty=float(debug_best_uncertainty),
            final_probs=cem_result.final_probs.detach().clone(),
            final_probs_entropy=entropy,
            final_probs_entropy_by_step=entropy_by_step,
            iteration_history=tuple(history),
            planning_latency_sec=latency,
            warm_start_used=bool(warm_start_used),
        )
