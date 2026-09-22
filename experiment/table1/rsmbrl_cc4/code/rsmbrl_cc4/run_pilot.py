"""Development-only RSMBRL pilot with exact CC4 transition accounting."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

import yaml

from baselines.rsmbrl_cc4.artifact_preflight import resolve, sha256_file
from baselines.rsmbrl_cc4.runtime import build_runtime, normalizer_hash
from formal_experiments.data_collection.collect_cc4_formal_replay import (
    encode_decision_state, make_env, observation_success_bool,
    planner_visible_target_availability,
)
from shared.action_contract import ACTION_CONTRACTS, get_action
from shared.cyborg_action_adapter import CybORGActionAdapter
from shared.formal_state import BLUE_AGENTS, FormalStateEncoder, ObservableHostEvidenceTracker
from shared.d27_projection import D27ProjectionContext


DEFAULT_CONFIG = "configs/formal_v3/methods/rsmbrl_cc4.yaml"
FAMILY_DURATION = {str(x.cyborg_action): int(x.duration_ticks) for x in ACTION_CONTRACTS}
PILOT_EPISODE_SEEDS = tuple(range(3200, 3210))
PILOT_POLICY_SEEDS = (61001, 61002)


def _line(handle, value) -> None:
    handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def _episode_passed(item: dict) -> bool:
    ticks = int(item["requested_transitions"])
    return bool(item["all_agents_done"] and not item["errors"]
                and int(item["environment_steps"]) == ticks
                and int(item["controller_tick_end"]) == ticks)


def _run_episode(*, planners: dict, seed: int, requested_transitions: int,
                 repeat: int, episode: int, decisions_handle) -> dict:
    env, reset_obs, _ = make_env(seed=seed, steps=requested_transitions + 1, pad_spaces=False)
    controller = env.env.environment_controller
    adapter, encoder = CybORGActionAdapter(), FormalStateEncoder()
    trackers, observations = {}, {}
    active = {agent: None for agent in BLUE_AGENTS}; done = set()
    requested = Counter(); executed = Counter(); fallback = Counter(); rewards = Counter()
    decisions = Counter(); planning_seconds = 0.0; errors = []
    for agent in BLUE_AGENTS:
        planners[agent].reset()
        tracker = ObservableHostEvidenceTracker(agent); tracker.reset(reset_obs[agent])
        trackers[agent], observations[agent] = tracker, reset_obs[agent]
    started_episode = time.perf_counter()
    env_steps = 0
    while env_steps < requested_transitions:
        tick = int(controller.step_count); joint = {}
        for agent in BLUE_AGENTS:
            if agent in done or active[agent] is not None:
                continue
            state, host_scores, _ = encode_decision_state(
                env=env, encoder=encoder, tracker=trackers[agent], observation=observations[agent],
                agent_name=agent, global_tick=tick, episode_steps=requested_transitions + 1,
            )
            _, availability = planner_visible_target_availability(
                env=env, agent_name=agent, observable_host_scores=host_scores)
            root_mask = [True, bool(availability["Analyse"]),
                         bool(availability["Remove"]), bool(availability["Restore"])]
            plan_started = time.perf_counter()
            result = planners[agent].plan(state, root_action_mask=root_mask,
                projection_context=D27ProjectionContext.from_root(state,root_tick=tick,episode_steps=requested_transitions+1))
            latency = time.perf_counter() - plan_started; planning_seconds += latency
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
            requested[action_id] += 1; executed[family] += 1; decisions[agent] += 1
            if resolution.fallback:
                fallback[resolution.fallback_reason or "unknown"] += 1
            _line(decisions_handle, {
                "formal_result_eligible": False, "method": "RSMBRL-CC4",
                "repeat": repeat, "episode": episode, "episode_seed": seed,
                "controller_tick": tick, "agent_name": agent,
                "requested_action_id": action_id,
                "requested_action_name": get_action(action_id).name,
                "executed_action_family": family, "executed_index": int(resolution.executed_index),
                "target_host": resolution.target_host, "fallback": bool(resolution.fallback),
                "fallback_reason": resolution.fallback_reason, "duration_ticks": duration,
                "root_action_mask": root_mask, "best_plan": result.best_plan.cpu().tolist(),
                "best_score": result.best_score, "expected_return": result.expected_return,
                "uncertainty": result.uncertainty, "planning_latency_seconds": latency,
            })
        new_obs, step_rewards, terminated, truncated, _ = env.step(actions=joint)
        env_steps += 1; tick_end = int(controller.step_count)
        if tick_end != tick + 1:
            raise RuntimeError("environment did not advance exactly one controller tick")
        for agent in BLUE_AGENTS:
            if agent in done: continue
            rewards[agent] += float(step_rewards.get(agent, 0.0))
            observations[agent] = new_obs[agent]; meta = active[agent]
            if meta is None: raise RuntimeError(f"{agent}: scheduler lost active action")
            if tick_end > int(meta["ready_at"]): raise RuntimeError(f"{agent}: missed ready_at")
            completed = tick_end == int(meta["ready_at"])
            finished = bool(terminated.get(agent, False) or truncated.get(agent, False))
            success = observation_success_bool(new_obs[agent]) if completed else None
            trackers[agent].update(
                observation=new_obs[agent], global_tick=tick_end,
                completed_action_family=meta["family"] if completed else None,
                completed_target_host=meta["target"] if completed else None,
                completed_action_success=success if completed else None,
            )
            if completed or finished: active[agent] = None
            if finished: done.add(agent)
    return {
        "formal_result_eligible": False, "repeat": repeat, "episode": episode,
        "episode_seed": seed, "requested_transitions": requested_transitions,
        "environment_steps": env_steps, "controller_tick_end": int(controller.step_count),
        "all_agents_done": len(done) == len(BLUE_AGENTS), "errors": errors,
        "per_agent_decisions": dict(decisions),
        "requested_action_counts": {str(i): requested[i] for i in range(4)},
        "executed_action_counts": dict(executed), "fallback_reasons": dict(fallback),
        "official_reward_by_agent": dict(rewards), "planning_seconds": planning_seconds,
        "wall_seconds": time.perf_counter() - started_episode,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=DEFAULT_CONFIG); p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu"); args = p.parse_args(argv)
    out = resolve(args.out); out.mkdir(parents=True, exist_ok=True)
    config_path = resolve(args.config); shutil.copyfile(config_path, out / "config.resolved.yaml")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    pilot = cfg["pilot"]
    if (int(pilot["training_repeats"]), int(pilot["episodes_per_repeat"]),
        int(pilot["ticks_per_episode"])) != (2, 5, 100):
        raise RuntimeError("resolved config must freeze development pilot at 2x5x100")
    summaries = []; repeat_normalizer_hashes = {}; gates = []
    with (out / "decisions.jsonl").open("w", encoding="utf-8") as decisions_handle:
        seed_index = 0
        for repeat, policy_seed in enumerate(PILOT_POLICY_SEEDS, start=1):
            planners, gate = build_runtime(device=args.device, planner_seed=policy_seed); gates.append(gate)
            before = {a: normalizer_hash(planners[a].uncertainty) for a in BLUE_AGENTS}
            for episode in range(1, 6):
                summaries.append(_run_episode(
                    planners=planners, seed=PILOT_EPISODE_SEEDS[seed_index],
                    requested_transitions=100, repeat=repeat, episode=episode,
                    decisions_handle=decisions_handle))
                seed_index += 1
            after = {a: normalizer_hash(planners[a].uncertainty) for a in BLUE_AGENTS}
            repeat_normalizer_hashes[str(repeat)] = {"before": before, "after": after,
                                                      "bitwise_frozen": before == after}
    with (out / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for item in summaries: _line(handle, item)
    passed = all(_episode_passed(x) for x in summaries) and all(
        x["bitwise_frozen"] for x in repeat_normalizer_hashes.values())
    decision_path, episode_path = out / "decisions.jsonl", out / "episodes.jsonl"
    artifact_hashes = gates[0]["artifacts"]
    report = {
        "schema": "rsmbrl_cc4_development_pilot_v1", "method": "RSMBRL-CC4",
        "formal_result_eligible": False, "passed": passed,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "device": args.device,
        "repeats": 2, "episodes_per_repeat": 5, "transitions_per_episode": 100,
        "episode_seeds": list(PILOT_EPISODE_SEEDS), "policy_seeds": list(PILOT_POLICY_SEEDS),
        "episodes": summaries, "normalizer_freeze_evidence": repeat_normalizer_hashes,
        "artifact_hashes": {k: v for k, v in artifact_hashes.items() if k.endswith("sha256")},
        "output_hashes": {
            "config.resolved.yaml": sha256_file(out / "config.resolved.yaml"),
            "decisions.jsonl": sha256_file(decision_path),
            "episodes.jsonl": sha256_file(episode_path),
        },
        "total_wall_seconds": sum(x["wall_seconds"] for x in summaries),
        "total_planning_seconds": sum(x["planning_seconds"] for x in summaries),
        "fallback_count": sum(sum(x["fallback_reasons"].values()) for x in summaries),
    }
    (out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                                   sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": passed, "episodes": len(summaries), "out": str(out)}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
