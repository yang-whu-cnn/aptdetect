"""One-state/one-plan local RSMBRL artifact benchmark (not an episode pilot)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from statistics import mean, median

import numpy as np
import torch

from baselines.rsmbrl_cc4.artifact_preflight import (
    DEFAULT_NORMALIZER, DEFAULT_REWARD_MODEL, DEFAULT_WORLD_MODEL, resolve, run_preflight,
)
from baselines.rsmbrl_cc4.planner import RSMBRLConfig, RSMBRLPlanner
from baselines.ug_cem_apt.categorical_cem import CategoricalCEMConfig, CategoricalCEMOptimizer
from baselines.ug_cem_apt.uncertainty import UGUncertainty, UGUncertaintyConfig
from formal_experiments.training.bootstrap_world_model import BootstrapProbabilisticWorldModel
from formal_experiments.training.response_reward_predictor import ResponseRewardPredictor
from shared.rollout_evaluator import SharedRolloutConfig, SharedRolloutEvaluator
from shared.d27_projection import D27ProjectionContext, PROJECTION_SHA256, PROJECTION_VERSION

DEFAULT_STATE = "outputs/ug_cem_v2/step6/calibration_states.jsonl"
DEFAULT_REPORT = "outputs/rsmbrl_cc4/preflight/single_plan_benchmark.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--agent", default="blue_agent_0")
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup must be >=0 and repeats must be >0")
    gate = run_preflight(device=args.device)
    if not gate["eligible"]:
        raise RuntimeError("artifact preflight failed: " + "; ".join(gate["errors"]))
    world = BootstrapProbabilisticWorldModel.load_checkpoint(resolve(DEFAULT_WORLD_MODEL), device=args.device)
    reward = ResponseRewardPredictor.load_checkpoint(resolve(DEFAULT_REWARD_MODEL), device=args.device)
    bundle = torch.load(resolve(DEFAULT_NORMALIZER), map_location=args.device, weights_only=False)
    uncertainty = UGUncertainty(UGUncertaintyConfig(alpha=0.01, eps=1e-8, device=args.device))
    item = bundle["agents"][args.agent]
    uncertainty.obs_mean = torch.as_tensor(item["obs_mean"], device=args.device).clone()
    uncertainty.obs_std = torch.as_tensor(item["obs_std"], device=args.device).clone()
    uncertainty.horizon_std = torch.as_tensor(item["horizon_std"], device=args.device).clone()
    uncertainty._state_dim, uncertainty._horizon = 27, 4
    snapshot = tuple(x.clone() for x in (uncertainty.obs_mean, uncertainty.obs_std, uncertainty.horizon_std))
    rollout = SharedRolloutEvaluator(world_model=world, reward_predictor=reward,
                                     config=SharedRolloutConfig(horizon=4, gamma_tick=0.99))
    cem = CategoricalCEMOptimizer(CategoricalCEMConfig(
        horizon=4, n_actions=4, population_size=200, num_iterations=5,
        elite_ratio=0.3, alpha=0.1, prob_floor=0.01, seed=20260919, device=args.device,
    ))
    planner = RSMBRLPlanner(cem_optimizer=cem, rollout_evaluator=rollout,
                            uncertainty=uncertainty, config=RSMBRLConfig())
    state = None
    root_tick = None
    with resolve(DEFAULT_STATE).open("r", encoding="utf-8") as handle:
        for raw in handle:
            row = json.loads(raw)
            if row["agent_name"] == args.agent:
                state = np.asarray(row["state"], dtype=np.float32)
                root_tick = int(row["global_tick_start"])
                break
    if state is None:
        raise RuntimeError("benchmark state not found")
    def synchronize():
        if str(args.device).startswith("cuda"):
            torch.cuda.synchronize()

    for _ in range(args.warmup):
        planner.plan(state, projection_context=D27ProjectionContext.from_root(
            state, root_tick=root_tick, episode_steps=100))
    synchronize()
    timings = []
    result = None
    for _ in range(args.repeats):
        started = time.perf_counter()
        result = planner.plan(state, projection_context=D27ProjectionContext.from_root(
            state, root_tick=root_tick, episode_steps=100))
        synchronize()
        timings.append(time.perf_counter() - started)
    assert result is not None
    elapsed = float(sum(timings))
    frozen = all(torch.equal(before, after) for before, after in zip(
        snapshot, (uncertainty.obs_mean, uncertainty.obs_std, uncertainty.horizon_std)
    ))
    if not frozen:
        raise RuntimeError("normalizer mutated during evaluation benchmark")
    report = {
        "status": "PASS", "kind": "single_state_single_planning_call",
        "not_episode_pilot": True, "device": str(args.device), "wall_time_seconds": elapsed,
        "warmup_calls": int(args.warmup), "measured_calls": int(args.repeats),
        "seconds_per_call_mean": float(mean(timings)),
        "seconds_per_call_median": float(median(timings)),
        "seconds_per_call_min": float(min(timings)),
        "seconds_per_call_max": float(max(timings)),
        "agent": args.agent, "requested_action_id": result.requested_action_id,
        "best_plan": result.best_plan.cpu().tolist(), "best_score": result.best_score,
        "expected_return": result.expected_return, "uncertainty": result.uncertainty,
        "normalizer_frozen_before_after": frozen, "population_size": 200,
        "num_iterations": 5, "horizon": 4,
        "projection_version": PROJECTION_VERSION, "projection_sha256": PROJECTION_SHA256,
    }
    path = resolve(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
