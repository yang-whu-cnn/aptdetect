from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch

from baselines.ug_cem_apt.categorical_cem import CategoricalCEMOptimizer
from baselines.ug_cem_apt.uncertainty import UGUncertainty
from shared.action_contract import N_ACTIONS


def risk_adjusted_score(expected_return, uncertainty, *, beta: float) -> torch.Tensor:
    """Equation (3) of Webster & Flach: sum reward minus beta * uncertainty."""
    b = float(beta)
    if not isfinite(b) or b < 0:
        raise ValueError("beta must be finite and >= 0")
    reward = torch.as_tensor(expected_return, dtype=torch.float32)
    risk = torch.as_tensor(uncertainty, dtype=torch.float32, device=reward.device)
    if reward.shape != risk.shape or not torch.isfinite(reward).all() or not torch.isfinite(risk).all():
        raise ValueError("return and uncertainty must be finite with identical shapes")
    return reward - b * risk


@dataclass(frozen=True)
class RSMBRLConfig:
    method_name: str = "RSMBRL-CC4"
    beta: float = 0.1
    horizon: int = 4
    ensemble_size: int = 5
    population_size: int = 200
    num_iterations: int = 5
    elite_ratio: float = 0.3
    cem_alpha: float = 0.1
    update_normalizer_during_evaluation: bool = False

    def __post_init__(self) -> None:
        if self.method_name != "RSMBRL-CC4":
            raise ValueError("formal method name must be RSMBRL-CC4")
        if self.horizon != 4 or self.ensemble_size != 5 or N_ACTIONS != 4:
            raise ValueError("RSMBRL-CC4 requires D27/A4, H=4 and the shared 5-member WM")
        if self.population_size <= 0 or self.num_iterations <= 0:
            raise ValueError("CEM population and iterations must be positive")
        if not 0 < self.elite_ratio <= 1 or not 0 <= self.cem_alpha <= 1:
            raise ValueError("invalid CEM elite ratio or alpha")
        if not isfinite(float(self.beta)) or self.beta < 0:
            raise ValueError("beta must be finite and >= 0")
        if self.update_normalizer_during_evaluation:
            raise ValueError("evaluation-time normalizer updates are forbidden")


@dataclass(frozen=True)
class RSMBRLPlanResult:
    method: str
    requested_action_id: int
    best_plan: torch.Tensor
    best_score: float
    expected_return: float
    uncertainty: float
    final_probs: torch.Tensor


class RSMBRLPlanner:
    """Thin CC4 planner over the existing categorical CEM and ensemble rollout core.

    Unlike the historical UG-CEM development planner, the paper penalty is not
    annealed by CEM iteration. Calibration statistics are read-only here.
    """

    def __init__(self, *, cem_optimizer: CategoricalCEMOptimizer, rollout_evaluator,
                 uncertainty: UGUncertainty, config: RSMBRLConfig | None = None):
        self.config = config or RSMBRLConfig()
        self.cem_optimizer = cem_optimizer
        self.rollout_evaluator = rollout_evaluator
        self.uncertainty = uncertainty
        c = cem_optimizer.cfg
        if (c.horizon, c.n_actions, c.population_size, c.num_iterations) != (
            4, 4, self.config.population_size, self.config.num_iterations
        ):
            raise ValueError("categorical CEM does not match resolved RSMBRL config")
        if abs(float(c.elite_ratio) - self.config.elite_ratio) > 1e-12 or abs(float(c.alpha) - self.config.cem_alpha) > 1e-12:
            raise ValueError("categorical CEM elite ratio/alpha does not match resolved config")
        if int(rollout_evaluator.config.horizon) != 4 or int(rollout_evaluator.ensemble_size) != 5:
            raise ValueError("rollout evaluator violates shared H4/ensemble-5 contract")
        self._warm_start = None

    def reset(self, **_kwargs) -> None:
        self._warm_start = None

    def plan(self, state, *, root_action_mask=None, projection_context=None) -> RSMBRLPlanResult:
        best = {"score": float("-inf"), "return": 0.0, "risk": 0.0, "plan": None}

        def objective(plans: torch.Tensor, _iteration: int) -> torch.Tensor:
            rollout = self.rollout_evaluator.evaluate(state, plans, projection_context=projection_context)
            reward = torch.as_tensor(rollout.expected_return, dtype=torch.float32)
            risk = self.uncertainty.compute(rollout.next_states, update_stats=False).to(reward.device)
            score = risk_adjusted_score(reward, risk, beta=self.config.beta)
            idx = int(torch.argmax(score).item())
            if float(score[idx]) > best["score"]:
                best.update({"score": float(score[idx]), "return": float(reward[idx]),
                             "risk": float(risk[idx]), "plan": plans[idx].detach().clone()})
            return score

        result = self.cem_optimizer.optimize(
            objective, initial_probs=self._warm_start, root_action_mask=root_action_mask
        )
        if root_action_mask is not None and not bool(
            torch.as_tensor(root_action_mask, dtype=torch.bool)[int(result.best_plan[0])]
        ):
            raise RuntimeError("CEM returned an action forbidden by the root availability mask")
        shifted = torch.empty_like(result.final_probs)
        shifted[:-1] = result.final_probs[1:]
        shifted[-1] = self.cem_optimizer.uniform_probs[-1]
        self._warm_start = shifted.detach().clone()
        action = int(result.best_plan[0])
        return RSMBRLPlanResult("RSMBRL-CC4", action, result.best_plan.detach().clone(),
                               float(result.best_score), best["return"], best["risk"],
                               result.final_probs.detach().clone())
