"""Frozen RSMBRL-CC4 runtime construction."""
from __future__ import annotations
import hashlib
import torch

from baselines.rsmbrl_cc4.artifact_preflight import DEFAULT_NORMALIZER, DEFAULT_REWARD_MODEL, DEFAULT_WORLD_MODEL, resolve, run_preflight
from baselines.rsmbrl_cc4.planner import RSMBRLConfig, RSMBRLPlanner
from baselines.ug_cem_apt.categorical_cem import CategoricalCEMConfig, CategoricalCEMOptimizer
from baselines.ug_cem_apt.uncertainty import UGUncertainty, UGUncertaintyConfig
from formal_experiments.training.bootstrap_world_model import BootstrapProbabilisticWorldModel
from formal_experiments.training.response_reward_predictor import ResponseRewardPredictor
from shared.formal_state import BLUE_AGENTS
from shared.rollout_evaluator import SharedRolloutConfig, SharedRolloutEvaluator


def normalizer_hash(uncertainty: UGUncertainty) -> str:
    digest = hashlib.sha256()
    for tensor in (uncertainty.obs_mean, uncertainty.obs_std, uncertainty.horizon_std):
        if tensor is None:
            raise RuntimeError("normalizer is not initialized")
        digest.update(tensor.detach().cpu().to(torch.float32).contiguous().numpy().tobytes())
    return digest.hexdigest()


def build_runtime(*, device="cpu", planner_seed: int = 20260919):
    gate = run_preflight(device=device)
    if not gate["eligible"]:
        raise RuntimeError("RSMBRL artifact preflight failed: " + "; ".join(gate["errors"]))
    world = BootstrapProbabilisticWorldModel.load_checkpoint(resolve(DEFAULT_WORLD_MODEL), device=device)
    reward = ResponseRewardPredictor.load_checkpoint(resolve(DEFAULT_REWARD_MODEL), device=device)
    bundle = torch.load(resolve(DEFAULT_NORMALIZER), map_location=device, weights_only=False)
    rollout = SharedRolloutEvaluator(world_model=world, reward_predictor=reward,
                                     config=SharedRolloutConfig(horizon=4, gamma_tick=0.99))
    planners = {}
    for index, agent in enumerate(BLUE_AGENTS):
        uncertainty = UGUncertainty(UGUncertaintyConfig(alpha=.01, eps=1e-8, device=device))
        entry = bundle["agents"][agent]
        uncertainty.obs_mean = torch.as_tensor(entry["obs_mean"], device=device).detach().clone()
        uncertainty.obs_std = torch.as_tensor(entry["obs_std"], device=device).detach().clone()
        uncertainty.horizon_std = torch.as_tensor(entry["horizon_std"], device=device).detach().clone()
        uncertainty._state_dim, uncertainty._horizon = 27, 4
        cem = CategoricalCEMOptimizer(CategoricalCEMConfig(
            horizon=4, n_actions=4, population_size=200, num_iterations=5,
            elite_ratio=.3, alpha=.1, prob_floor=.01, seed=int(planner_seed) + index, device=device,
        ))
        planners[agent] = RSMBRLPlanner(cem_optimizer=cem, rollout_evaluator=rollout,
                                        uncertainty=uncertainty,
                                        config=RSMBRLConfig(beta=float(bundle["beta"])))
    return planners, gate
