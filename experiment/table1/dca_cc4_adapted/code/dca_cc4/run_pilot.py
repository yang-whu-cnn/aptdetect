from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

import yaml

from baselines.dca_cc4 import DCAConfig
from baselines.dca_cc4.runtime import DCAAgentRuntime
from formal_experiments.data_collection.collect_cc4_formal_replay import (
    make_env, observation_success_bool,
)
from shared.cyborg_action_adapter import CybORGActionAdapter
from shared.formal_state import BLUE_AGENTS, ObservableHostEvidenceTracker


DEFAULT_CONFIG = Path("configs/formal_v3/methods/dca_cc4.yaml")
FAMILY_DURATION = {"Sleep": 1, "Analyse": 2, "Remove": 3, "Restore": 5}


def _dump_line(handle, value) -> None:
    handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def _run_episode(*, seed: int, steps: int, repeat: int, episode: int,
                 decisions_handle, events_handle, dca_config: DCAConfig) -> dict:
    # CC4 reset starts at controller tick 0 and terminates at scenario_steps-1.
    # Therefore ticks+1 scenario steps are required for exactly `steps`
    # post-reset environment transitions.
    env, observations, _info = make_env(seed=seed, steps=steps + 1, pad_spaces=False)
    controller = env.env.environment_controller
    adapter = CybORGActionAdapter()
    trackers = {}
    runtimes = {}
    current = {}
    active = {agent: None for agent in BLUE_AGENTS}
    done = set()
    requested = Counter(); executed = Counter(); fallbacks = Counter()
    reward_sums = Counter(); decision_count = Counter(); raw_event_count = Counter()
    errors = []

    for index, agent in enumerate(BLUE_AGENTS):
        tracker = ObservableHostEvidenceTracker(agent)
        tracker.reset(observations[agent])
        trackers[agent] = tracker
        runtimes[agent] = DCAAgentRuntime(tracker=tracker, config=dca_config,
                                          seed=repeat * 100 + index)
        current[agent] = observations[agent]

    env_steps = 0
    started = time.perf_counter()
    while int(controller.step_count) < steps and len(done) < len(BLUE_AGENTS):
        tick = int(controller.step_count)
        joint_actions = {}
        for agent in BLUE_AGENTS:
            if agent in done or active[agent] is not None:
                continue
            decision = runtimes[agent].act()
            host_scores = trackers[agent].observable_host_scores(tick)
            resolver_scores = dict(host_scores)
            if decision.target in resolver_scores:
                resolver_scores[decision.target] = max(resolver_scores.values(), default=0.0) + 1.0
            resolution = adapter.resolve(env=env, agent_name=agent,
                                         action_id=decision.requested_action_id,
                                         observable_host_scores=resolver_scores)
            action_obj = list(env.actions(agent))[resolution.executed_index]
            duration = int(action_obj.duration)
            if duration != FAMILY_DURATION[resolution.executed_action_family]:
                raise RuntimeError("shared action duration mismatch")
            active[agent] = {"ready_at": tick + duration,
                             "family": resolution.executed_action_family,
                             "target": resolution.target_host}
            joint_actions[agent] = resolution.executed_index
            requested[decision.requested_action_id] += 1
            executed[resolution.executed_action_family] += 1
            if resolution.fallback:
                fallbacks[resolution.fallback_reason or "unknown"] += 1
            decision_count[agent] += 1
            _dump_line(decisions_handle, {
                "formal_result_eligible": False, "repeat": repeat, "episode": episode,
                "episode_seed": seed, "tick": tick, "agent": agent,
                "method": decision.method, "requested_action_id": decision.requested_action_id,
                "dca_target": decision.target, "dca_reason": decision.reason,
                "inferred_attacker_action": decision.inferred_attacker_action,
                "evidence_count": decision.evidence_count,
                "executed_family": resolution.executed_action_family,
                "executed_index": resolution.executed_index,
                "executed_target": resolution.target_host, "fallback": resolution.fallback,
                "fallback_reason": resolution.fallback_reason,
            })

        next_obs, rewards, terminated, truncated, _info = env.step(actions=joint_actions)
        env_steps += 1
        tick_end = int(controller.step_count)
        for agent in BLUE_AGENTS:
            if agent in done:
                continue
            observation = next_obs[agent]
            reward_sums[agent] += float(rewards.get(agent, 0.0))
            meta = active[agent]
            completed = meta is not None and tick_end == int(meta["ready_at"])
            finished = bool(terminated.get(agent, False) or truncated.get(agent, False))
            success = observation_success_bool(observation) if completed else None
            trackers[agent].update(
                observation=observation, global_tick=tick_end,
                completed_action_family=meta["family"] if completed else None,
                completed_target_host=meta["target"] if completed else None,
                completed_action_success=success if completed else None,
            )
            fresh = runtimes[agent].observe(
                observation, tick=tick_end,
                completed_family=meta["family"] if completed else None,
                completed_target=meta["target"] if completed else None,
                completed_success=success if completed else None,
            )
            for token in fresh:
                raw_event_count[agent] += 1
                _dump_line(events_handle, {
                    "formal_result_eligible": False, "repeat": repeat, "episode": episode,
                    "episode_seed": seed, "agent": agent, **asdict(token),
                })
            current[agent] = observation
            if completed or finished:
                active[agent] = None
            if finished:
                done.add(agent)

    return {
        "repeat": repeat, "episode": episode, "episode_seed": seed, "requested_ticks": steps,
        "controller_tick_end": int(controller.step_count), "environment_steps": env_steps,
        "all_agents_done": len(done) == len(BLUE_AGENTS),
        "decisions": dict(decision_count), "raw_events": dict(raw_event_count),
        "requested_action_counts": {str(i): requested[i] for i in range(4)},
        "executed_action_counts": dict(executed), "fallback_reasons": dict(fallbacks),
        "official_reward_by_agent": dict(reward_sums), "errors": errors,
        "wall_seconds": time.perf_counter() - started,
    }


def _episode_passed(item: dict) -> bool:
    requested = int(item["requested_ticks"])
    return bool(
        item["all_agents_done"]
        and not item["errors"]
        and int(item["environment_steps"]) == requested
        and int(item["controller_tick_end"]) == requested
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("wiring", "pilot"), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    resolved = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    dca_config = DCAConfig(
        min_pts=int(resolved["alert_processing"]["min_pts"]),
        epsilon=float(resolved["alert_processing"]["epsilon"]),
        time_window=int(resolved["alert_processing"]["time_window"]),
        high_confidence=float(resolved["response_mapping"]["high_confidence"]),
        multi_channel_threshold=int(resolved["response_mapping"]["multi_channel_threshold"]),
    )
    repeats, episodes, steps, seeds = ((1, 1, 20, [3199]) if args.profile == "wiring"
                                       else (2, 5, 100, list(range(3200, 3210))))
    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, args.out / "config.resolved.yaml")
    summaries = []
    with (args.out / "decisions.jsonl").open("w", encoding="utf-8") as decisions, \
         (args.out / "raw_events.jsonl").open("w", encoding="utf-8") as events:
        index = 0
        for repeat in range(1, repeats + 1):
            for episode in range(1, episodes + 1):
                summaries.append(_run_episode(seed=seeds[index], steps=steps, repeat=repeat,
                                              episode=episode, decisions_handle=decisions,
                                              events_handle=events, dca_config=dca_config))
                index += 1
    passed = all(_episode_passed(x) for x in summaries)
    decisions_path = args.out / "decisions.jsonl"
    events_path = args.out / "raw_events.jsonl"
    decision_lines = sum(1 for _ in decisions_path.open("r", encoding="utf-8"))
    event_lines = sum(1 for _ in events_path.open("r", encoding="utf-8"))
    fallback_total = sum(sum(int(v) for v in item["fallback_reasons"].values()) for item in summaries)
    report = {
        "schema": "dca_cc4_development_pilot_v1", "method": "DCA-CC4 (adapted)",
        "profile": args.profile, "formal_result_eligible": False, "passed": passed,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": hashlib.sha256((args.out / "config.resolved.yaml").read_bytes()).hexdigest(),
        "repeats": repeats, "episodes_per_repeat": episodes, "ticks_per_episode": steps,
        "total_decisions": decision_lines, "total_raw_events": event_lines,
        "fallback_count": fallback_total,
        "fallback_rate": (fallback_total / decision_lines if decision_lines else None),
        "artifacts": {
            "decisions.jsonl": {"lines": decision_lines, "sha256": hashlib.sha256(decisions_path.read_bytes()).hexdigest()},
            "raw_events.jsonl": {"lines": event_lines, "sha256": hashlib.sha256(events_path.read_bytes()).hexdigest()},
            "config.resolved.yaml": {"sha256": hashlib.sha256((args.out / "config.resolved.yaml").read_bytes()).hexdigest()},
        },
        "episodes": summaries,
    }
    (args.out / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"passed": passed, "episodes": len(summaries), "out": str(args.out)}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
