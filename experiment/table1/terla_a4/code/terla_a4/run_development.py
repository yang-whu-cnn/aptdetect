from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.distributions import Categorical

from baselines.terla_a4.graph import HeteroGraph
from baselines.terla_a4.model import TERLAPolicy, TERLASharedController
from baselines.terla_a4.reward import CC4HiddenRewardAdapter, TERLARewardChannel
from baselines.terla_a4.runtime import ObservableTERLAState
from baselines.terla_a4.targeting import DURATIONS, resolve_terla_action
from baselines.terla_a4.training import PPOConfig, TERLAPPOTrainer, Transition, policy_hash
from formal_experiments.data_collection.collect_cc4_formal_replay import make_env, observation_success_bool
from shared.cyborg_action_adapter import CybORGActionAdapter
from shared.formal_state import BLUE_AGENTS, ObservableHostEvidenceTracker


def _observable_scores(graph: HeteroGraph, action: int) -> dict[str, float]:
    target = resolve_terla_action(action, graph).target
    if target is None: return {}
    return {host: (2.0 if host == target else 0.0) for host in graph.host_names}


def resolve_with_mask(env, agent: str, graph: HeteroGraph, adapter: CybORGActionAdapter):
    resolutions = [adapter.resolve(env=env, agent_name=agent, action_id=action,
                                   observable_host_scores=_observable_scores(graph, action))
                   for action in range(4)]
    mask = torch.tensor([action == 0 or not resolutions[action].fallback for action in range(4)],
                        dtype=torch.bool)
    return resolutions, mask


def select_observable_action(policy: TERLAPolicy, graph: HeteroGraph, mask: torch.Tensor,
                             *, deterministic: bool, generator: torch.Generator):
    device = next(policy.parameters()).device
    logits, value = policy(graph.to(device)); mask = mask.to(device)
    masked = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    dist = Categorical(logits=masked)
    if deterministic:
        action = int(torch.argmax(masked))
    else:
        action = int(torch.multinomial(dist.probs, 1, generator=generator))
    return action, float(dist.log_prob(torch.tensor(action, device=device)).detach()), float(value.detach())


def run_episode(*, policy: TERLAPolicy, seed: int, ticks: int, mode: str,
                decisions_handle, generator: torch.Generator):
    env, observations, _info = make_env(seed=seed, steps=ticks + 1, pad_spaces=False)
    controller = env.env.environment_controller
    adapter = CybORGActionAdapter(); shared = TERLASharedController(policy)
    trackers = {}; states = {}; active = {agent: None for agent in BLUE_AGENTS}
    previous_health = {}; rows: list[Transition] = []; done = set()
    requested = Counter(); executed = Counter(); durations = Counter(); fallbacks = Counter(); errors = []
    for agent in BLUE_AGENTS:
        tracker = ObservableHostEvidenceTracker(agent); tracker.reset(observations[agent])
        trackers[agent] = tracker; states[agent] = ObservableTERLAState(tracker)
        record = CC4HiddenRewardAdapter.record(controller, tracker.inventory)
        previous_health[agent] = TERLARewardChannel.health(record)

    started = time.perf_counter(); env_steps = 0
    while int(controller.step_count) < ticks and len(done) < len(BLUE_AGENTS):
        tick = int(controller.step_count); joint = {}
        for agent in BLUE_AGENTS:
            if agent in done or active[agent] is not None: continue
            graph = states[agent].graph(tick=tick, episode_ticks=ticks)
            resolutions, mask = resolve_with_mask(env, agent, graph, adapter)
            action, log_prob, value = select_observable_action(
                policy, graph, mask, deterministic=(mode == "eval"), generator=generator)
            resolution = resolutions[action]
            action_obj = list(env.actions(agent))[resolution.executed_index]
            duration = int(action_obj.duration)
            if duration != DURATIONS[action]: raise RuntimeError("TERLA duration mismatch")
            active[agent] = {"ready": tick + duration, "graph": graph, "action": action,
                             "log_prob": log_prob, "value": value, "duration": duration,
                             "reward": 0.0, "family": resolution.executed_action_family,
                             "target": resolution.target_host, "mask": mask.tolist()}
            joint[agent] = resolution.executed_index; requested[action] += 1
            executed[resolution.executed_action_family] += 1
            durations[duration] += 1
            if resolution.fallback: fallbacks[resolution.fallback_reason or "unknown"] += 1
            shared.record_action(agent, action)
            decisions_handle.write(json.dumps({
                "mode": mode, "seed": seed, "tick": tick, "agent": agent,
                "requested_action": action, "executed_family": resolution.executed_action_family,
                "target": resolution.target_host, "availability_mask": mask.tolist(),
                "duration": duration, "fallback": resolution.fallback,
                "formal_result_eligible": False}, sort_keys=True) + "\n")
        next_obs, _official, terminated, truncated, _info = env.step(actions=joint)
        env_steps += 1; tick_end = int(controller.step_count)
        for agent in BLUE_AGENTS:
            if agent in done: continue
            meta = active[agent]; observation = next_obs[agent]
            finished = bool(terminated.get(agent, False) or truncated.get(agent, False))
            completed = meta is not None and tick_end == meta["ready"]
            success = observation_success_bool(observation) if completed else None
            trackers[agent].update(observation=observation, global_tick=tick_end,
                completed_action_family=meta["family"] if completed else None,
                completed_target_host=meta["target"] if completed else None,
                completed_action_success=success if completed else None)
            states[agent].update(observation,
                restored_target=meta["target"] if completed and success and meta["family"] == "Restore" else None)
            current = TERLARewardChannel.health(CC4HiddenRewardAdapter.record(controller, trackers[agent].inventory))
            if meta is not None: meta["reward"] += current - previous_health[agent]
            previous_health[agent] = current
            if completed or (finished and meta is not None):
                next_graph = states[agent].graph(tick=tick_end, episode_ticks=ticks)
                next_value = 0.0 if finished else float(policy(next_graph.to(next(policy.parameters()).device))[1].detach())
                rows.append(Transition(meta["graph"], meta["action"], meta["log_prob"], meta["value"],
                    meta["reward"], next_value, finished, meta["duration"], f"{seed}:{agent}",
                    tuple(bool(x) for x in meta["mask"])))
                active[agent] = None
            if finished: done.add(agent)
    for agent in BLUE_AGENTS:
        indices = [i for i, row in enumerate(rows) if row.trajectory == f"{seed}:{agent}"]
        if indices:
            rows[indices[-1]].done = True; rows[indices[-1]].next_value = 0.0
    trajectory_counts = {agent: sum(row.trajectory == f"{seed}:{agent}" for row in rows)
                         for agent in BLUE_AGENTS}
    truncated = all(any(row.trajectory == f"{seed}:{agent}" and row.done for row in rows)
                    for agent in BLUE_AGENTS)
    return rows, {"mode": mode, "seed": seed, "requested_ticks": ticks,
        "environment_steps": env_steps, "controller_tick_end": int(controller.step_count),
        "all_agents_done": len(done) == len(BLUE_AGENTS), "errors": errors,
        "requested_action_counts": {str(i): requested[i] for i in range(4)},
        "executed_action_counts": dict(executed), "fallback_reasons": dict(fallbacks),
        "duration_counts": {str(i): durations[i] for i in sorted(durations)},
        "reward_sum": float(sum(row.reward for row in rows)),
        "transitions": len(rows), "trajectory_counts": trajectory_counts,
        "all_trajectories_episode_truncated": truncated,
        "wall_seconds": time.perf_counter()-started}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ticks", type=int, default=20); args = parser.parse_args()
    torch.manual_seed(51004); policy = TERLAPolicy(); trainer = TERLAPPOTrainer(policy, PPOConfig())
    generator = torch.Generator().manual_seed(51004); args.out.mkdir(parents=True, exist_ok=True)
    decisions_path = args.out / "decisions.jsonl"
    with decisions_path.open("w", encoding="utf-8") as handle:
        train_rows, train = run_episode(policy=policy, seed=3199, ticks=args.ticks, mode="train",
                                        decisions_handle=handle, generator=generator)
        before_train = policy_hash(policy); update = trainer.optimize(train_rows); after_train = policy_hash(policy)
        before_eval = policy_hash(policy)
        eval_rows, evaluation = run_episode(policy=policy, seed=3200, ticks=args.ticks, mode="eval",
                                            decisions_handle=handle, generator=generator)
        after_eval = policy_hash(policy)
    passed = all(x["environment_steps"] == args.ticks and x["controller_tick_end"] == args.ticks
                 and x["all_agents_done"] and x["all_trajectories_episode_truncated"]
                 and not x["errors"] for x in (train, evaluation))
    passed &= before_train != after_train and before_eval == after_eval
    report = {"schema": "terla_a4_train_eval_smoke_v1", "formal_result_eligible": False,
        "passed": bool(passed), "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "train": train, "eval": evaluation, "ppo_update": update,
        "ppo_config": {"gamma": .97, "learning_rate": 1e-4, "entropy": .01,
                       "rollout_target": 128, "episode_boundary_flush": True},
        "policy_hashes": {"before_train": before_train, "after_train": after_train,
                          "before_eval": before_eval, "after_eval": after_eval},
        "decisions_sha256": hashlib.sha256(decisions_path.read_bytes()).hexdigest()}
    (args.out / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "out": str(args.out)}))
    if not passed: raise SystemExit(1)


if __name__ == "__main__": main()
