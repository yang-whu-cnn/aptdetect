"""Exactly one CC4 episode with 20 post-reset transitions for wiring only."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean
import time
import numpy as np

from baselines.rsmbrl_cc4.artifact_preflight import resolve
from baselines.rsmbrl_cc4.runtime import build_runtime, normalizer_hash
from formal_experiments.data_collection.collect_cc4_formal_replay import (
    encode_decision_state, make_env, observation_success_bool,
    planner_visible_target_availability,
)
from shared.action_contract import ACTION_CONTRACTS, get_action
from shared.cyborg_action_adapter import CybORGActionAdapter
from shared.formal_state import BLUE_AGENTS, FormalStateEncoder, ObservableHostEvidenceTracker
from shared.d27_projection import D27ProjectionContext, PROJECTION_SHA256, PROJECTION_VERSION

ENVIRONMENT_STEPS = 20
SCENARIO_STEPS = 21  # CC4 reset is controller tick 0; terminal tick is steps-1.
EPISODES = 1
DEFAULT_OUTPUT = "outputs/rsmbrl_cc4/wiring_smoke/episode_3000_steps_20.json"
FAMILY_DURATION = {str(x.cyborg_action): int(x.duration_ticks) for x in ACTION_CONTRACTS}


def run(*, seed=3000, device="cpu") -> dict:
    if ENVIRONMENT_STEPS != 20 or SCENARIO_STEPS != 21 or EPISODES != 1:
        raise RuntimeError("wiring smoke hard gate must remain 1 episode x 20 post-reset transitions")
    planners, gate = build_runtime(device=device)
    before = {agent: normalizer_hash(planner.uncertainty) for agent, planner in planners.items()}
    env, reset_obs, _ = make_env(seed=seed, steps=SCENARIO_STEPS, pad_spaces=False)
    controller = env.env.environment_controller
    if int(controller.step_count) != 0:
        raise RuntimeError("post-reset controller must start at tick 0")
    adapter, encoder = CybORGActionAdapter(), FormalStateEncoder()
    trackers, observations = {}, {}
    active = {agent: None for agent in BLUE_AGENTS}
    for agent in BLUE_AGENTS:
        planners[agent].reset()
        tracker = ObservableHostEvidenceTracker(agent)
        tracker.reset(reset_obs[agent])
        trackers[agent], observations[agent] = tracker, reset_obs[agent]
    decisions, requested, executed = [], Counter(), Counter()
    for env_step in range(ENVIRONMENT_STEPS):
        tick = int(controller.step_count)
        joint = {}
        for agent in BLUE_AGENTS:
            if active[agent] is not None:
                continue
            state, host_scores, _ = encode_decision_state(
                env=env, encoder=encoder, tracker=trackers[agent], observation=observations[agent],
                agent_name=agent, global_tick=tick, episode_steps=SCENARIO_STEPS,
            )
            _any_available, family_availability = planner_visible_target_availability(
                env=env, agent_name=agent, observable_host_scores=host_scores
            )
            action_availability = {
                "no_op": True,
                "analyse": bool(family_availability["Analyse"]),
                "remove": bool(family_availability["Remove"]),
                "restore": bool(family_availability["Restore"]),
            }
            root_mask = [action_availability[name] for name in ("no_op", "analyse", "remove", "restore")]
            started = time.perf_counter()
            result = planners[agent].plan(state, root_action_mask=root_mask,
                projection_context=D27ProjectionContext.from_root(state,root_tick=tick,episode_steps=SCENARIO_STEPS))
            latency = time.perf_counter() - started
            action_id = int(result.requested_action_id)
            resolution = adapter.resolve(env=env, agent_name=agent, action_id=action_id,
                                         observable_host_scores=host_scores)
            action = list(env.actions(agent))[int(resolution.executed_index)]
            family, duration = type(action).__name__, int(action.duration)
            if duration != FAMILY_DURATION[family]:
                raise RuntimeError(f"duration mismatch for {family}: {duration}")
            active[agent] = {"ready_at": tick + duration, "family": family,
                             "target": resolution.target_host}
            joint[agent] = int(resolution.executed_index)
            requested[action_id] += 1
            executed[family] += 1
            decisions.append({
                "agent_name": agent, "controller_tick": tick, "requested_action_id": action_id,
                "requested_action_name": get_action(action_id).name,
                "executed_action_family": family, "executed_index": int(resolution.executed_index),
                "target_host": resolution.target_host, "fallback": bool(resolution.fallback),
                "fallback_reason": resolution.fallback_reason, "duration_ticks": duration,
                "ready_at": tick + duration, "planning_latency_seconds": latency,
                "best_plan": result.best_plan.cpu().tolist(), "best_score": result.best_score,
                "expected_return": result.expected_return, "uncertainty": result.uncertainty,
                "observable_target_family_availability": action_availability,
                "root_action_mask": root_mask,
            })
        new_obs, _rewards, terminated, truncated, _ = env.step(actions=joint)
        if int(controller.step_count) != tick + 1:
            raise RuntimeError("environment did not advance exactly one controller tick")
        tick_end = int(controller.step_count)
        for agent in BLUE_AGENTS:
            observations[agent] = new_obs[agent]
            meta = active[agent]
            if meta is None:
                raise RuntimeError(f"{agent}: scheduler lost active action")
            if tick_end > meta["ready_at"]:
                raise RuntimeError(f"{agent}: scheduler missed ready_at")
            completed = tick_end == meta["ready_at"]
            success = observation_success_bool(new_obs[agent]) if completed else None
            trackers[agent].update(
                observation=new_obs[agent], global_tick=tick_end,
                completed_action_family=meta["family"] if completed else None,
                completed_target_host=meta["target"] if completed else None,
                completed_action_success=success if completed else None,
            )
            if completed or terminated.get(agent, False) or truncated.get(agent, False):
                active[agent] = None
    if int(controller.step_count) != 20:
        raise RuntimeError(f"controller tick hard gate failed: {controller.step_count}")
    after = {agent: normalizer_hash(planner.uncertainty) for agent, planner in planners.items()}
    if before != after:
        raise RuntimeError("frozen normalizer mutated during wiring smoke")
    latencies = [x["planning_latency_seconds"] for x in decisions]
    return {
        "status": "PASS", "method": "RSMBRL-CC4", "formal_result_eligible": False,
        "development_wiring_smoke": True, "episodes": 1, "episode_seed": seed,
        "scenario_steps": 21, "environment_steps": 20,
        "controller_tick_start": 0, "controller_tick_end": 20,
        "device": device, "config": {"D": 27, "A": 4, "H": 4, "M": 5,
            "population_size": 200, "num_iterations": 5, "elite_ratio": .3,
            "cem_alpha": .1, "beta": .1, "duration_scheduler": True,
            "projection_version": PROJECTION_VERSION,
            "projection_sha256": PROJECTION_SHA256},
        "artifact_hashes": {k: v for k, v in gate["artifacts"].items() if k.endswith("sha256")},
        "normalizer_hash_before": before, "normalizer_hash_after": after,
        "normalizer_bitwise_frozen": True, "decision_count": len(decisions),
        "requested_action_counts": {str(k): v for k, v in sorted(requested.items())},
        "executed_action_counts": dict(sorted(executed.items())),
        "fallback_count": sum(int(x["fallback"]) for x in decisions),
        "mean_planning_latency_seconds": mean(latencies),
        "max_planning_latency_seconds": max(latencies), "decisions": decisions,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    report = run(seed=args.seed, device=args.device)
    path = resolve(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "decisions"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
