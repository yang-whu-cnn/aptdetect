"""Reusable CC4 episode runner for formal-v3 method adapters.

Policy observations and evaluator truth are kept in separate local channels.
This module does not train, update normalizers, or call an online LLM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from baselines.dca_cc4 import DCAConfig
from baselines.dca_cc4.runtime import DCAAgentRuntime
from formal_experiments.common.run_manifest import PROTOCOL_VERSION
from formal_experiments.evaluation.metrics_v3 import compute_episode_metrics
from formal_experiments.evaluation.run_formal_suite import failure_owner_from_inventories
from shared.cyborg_action_adapter import CybORGActionAdapter
from shared.formal_state import BLUE_AGENTS, FormalStateEncoder, ObservableHostEvidenceTracker


METHODS = ("dca_cc4", "rsmbrl_cc4", "uamcts_cc4")
POLICY_SEEDS = (51001, 51002, 51003, 51004, 51005)
FAMILY_TO_A4 = {"Sleep": 0, "Analyse": 1, "Remove": 2, "Restore": 3}


def root_action_mask_from_availability(availability: dict[str, bool]) -> list[bool]:
    required = ("Analyse", "Remove", "Restore")
    if set(availability) != set(required) or any(
        not isinstance(availability[name], bool) for name in required
    ):
        raise ValueError("target availability must contain boolean Analyse/Remove/Restore")
    return [True, availability["Analyse"], availability["Remove"], availability["Restore"]]


def plan_rsmbrl_with_visible_mask(planner, state, availability: dict[str, bool]):
    mask = root_action_mask_from_availability(availability)
    return planner.plan(state, root_action_mask=mask), mask


def incident_active_at_action_start(target: str | None, previous_truth: dict[str, bool]) -> bool:
    """Evaluator-only TP/FP label captured before the current environment step."""
    return bool(target is not None and previous_truth.get(target, False))


def validate_episode_request(*, run_mode: str, ticks: int) -> None:
    if run_mode not in ("dev", "formal"):
        raise ValueError("run_mode must be dev or formal")
    if isinstance(ticks, bool) or not isinstance(ticks, int) or ticks <= 0:
        raise ValueError("ticks must be a positive integer")
    if run_mode == "formal" and ticks != 500:
        raise ValueError("formal method episodes are hard-locked to 500 ticks")


def derive_runtime_seed(*, policy_seed: int, episode_seed: int, agent_name: str) -> int:
    payload = f"cc4-v3|{int(policy_seed)}|{int(episode_seed)}|{agent_name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def run_method_episode(
    *, method: str, seed: int, ticks: int, run_mode: str = "dev", device: str = "cpu",
    policy_seed: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    validate_episode_request(run_mode=run_mode, ticks=ticks)
    if run_mode == "formal" and policy_seed not in POLICY_SEEDS:
        raise ValueError("formal episode requires one frozen policy_seed")
    if policy_seed is None:
        policy_seed = 0
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    from formal_experiments.data_collection.collect_cc4_formal_replay import (
        encode_decision_state,
        green_local_work_failures,
        ground_truth_red_presence,
        make_env,
        observation_success_bool,
    )
    env, observations, _ = make_env(seed=seed, steps=ticks + 1, pad_spaces=False)
    controller = env.env.environment_controller
    adapter = CybORGActionAdapter()
    encoder = FormalStateEncoder()
    trackers: dict[str, ObservableHostEvidenceTracker] = {}
    policies: dict[str, Any] = {}
    current = dict(observations)
    active: dict[str, dict[str, Any] | None] = {agent: None for agent in BLUE_AGENTS}
    done: set[str] = set()
    inventories: dict[str, tuple[str, ...]] = {}
    runtime_seeds: dict[str, int] = {}

    planners = None
    uamcts_priors = None
    if method == "rsmbrl_cc4":
        from baselines.rsmbrl_cc4.runtime import build_runtime
        planner_seed = derive_runtime_seed(
            policy_seed=policy_seed, episode_seed=seed, agent_name="planner_base"
        )
        planners, gate = build_runtime(device=device, planner_seed=planner_seed)
        if not gate["eligible"]:
            raise RuntimeError("RSMBRL artifact gate failed")
    elif method == "uamcts_cc4":
        from baselines.uamcts_cc4.runtime import build_runtime
        root = Path(__file__).resolve().parents[2]
        planner_seed = derive_runtime_seed(
            policy_seed=policy_seed, episode_seed=seed, agent_name="planner_base"
        )
        planners, uamcts_priors = build_runtime(
            world_path=root / "outputs/world_model_final_20260917/a4_5b/world_model_absolute.pt",
            reward_path=root / "outputs/world_model_final_20260917/a4_5c/response_reward_predictor.pt",
            progress_path=root / "outputs/uamcts_cc4/progress/progress_ensemble_train_only.pt",
            prototype_path=root / "outputs/priorrl_cc4/prototypes/frozen_prototypes.json",
            normalizers_path=root / "outputs/uamcts_cc4/calibration/uamcts_uncertainty_normalizers_frozen.pt",
            device=device, planner_seed=planner_seed,
        )

    for index, agent in enumerate(BLUE_AGENTS):
        tracker = ObservableHostEvidenceTracker(agent)
        tracker.reset(observations[agent])
        trackers[agent] = tracker
        inventories[agent] = tuple(tracker.inventory)
        if method == "dca_cc4":
            runtime_seed = derive_runtime_seed(
                policy_seed=policy_seed, episode_seed=seed, agent_name=agent
            )
            policy = DCAAgentRuntime(tracker=tracker, config=DCAConfig(), seed=runtime_seed)
            policy.observe(observations[agent], tick=0)
            policies[agent] = policy
            runtime_seeds[agent] = runtime_seed
        else:
            runtime_seeds[agent] = planner_seed + index

    jurisdiction_hosts = sorted({host for hosts in inventories.values() for host in hosts})
    previous = {host: ground_truth_red_presence(controller, host) for host in jurisdiction_hosts}
    incidents: list[dict[str, Any]] = []
    open_incidents: dict[str, dict[str, Any]] = {}
    for host, present in previous.items():
        if present:
            item = {"incident_id": f"{seed}:{host}:0", "host": host,
                    "t_compromise": 0, "t_recovered": None}
            incidents.append(item); open_incidents[host] = item

    decisions: list[dict[str, Any]] = []
    recovery_actions: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    excluded_failures: list[dict[str, Any]] = []
    tick_rewards: list[dict[str, float]] = []

    while int(controller.step_count) < ticks and len(done) < len(BLUE_AGENTS):
        tick = int(controller.step_count)
        joint: dict[str, int] = {}
        for agent in BLUE_AGENTS:
            if agent in done or active[agent] is not None:
                continue
            tracker = trackers[agent]
            state, host_scores, family_availability = encode_decision_state(
                env=env, encoder=encoder, tracker=tracker, observation=current[agent],
                agent_name=agent, global_tick=tick, episode_steps=ticks,
            )
            if method == "dca_cc4":
                policy_decision = policies[agent].act()
                requested = int(policy_decision.requested_action_id)
                preferred_target = policy_decision.target
                resolver_scores = dict(host_scores)
                if preferred_target in resolver_scores:
                    resolver_scores[preferred_target] = max(resolver_scores.values(), default=0.0) + 1.0
                diagnostics = {"reason": policy_decision.reason}
            elif method == "rsmbrl_cc4":
                plan, root_action_mask = plan_rsmbrl_with_visible_mask(
                    planners[agent], state, family_availability
                )
                requested = int(plan.requested_action_id)
                resolver_scores = host_scores
                diagnostics = {"best_score": float(plan.best_score),
                               "uncertainty": float(plan.uncertainty)}
            else:
                from shared.action_contract import ACTION_CONTRACTS
                from shared.d27_projection import D27ProjectionContext
                root_action_mask = root_action_mask_from_availability(family_availability)
                available = [index for index, enabled in enumerate(root_action_mask) if enabled]
                context = D27ProjectionContext.from_root(
                    state, root_tick=tick, episode_steps=ticks
                )
                plan = planners[agent].plan(
                    state, available_actions=available,
                    durations=tuple(int(item.duration_ticks) for item in ACTION_CONTRACTS),
                    projection_context=context,
                )
                requested = int(plan.action)
                resolver_scores = host_scores
                diagnostics = {"root_visits": list(plan.root_visits), "root_q": list(plan.root_q),
                               "simulations": int(plan.simulations)}
            if method == "dca_cc4":
                root_action_mask = root_action_mask_from_availability(family_availability)
            resolution = adapter.resolve(env=env, agent_name=agent, action_id=requested,
                                         observable_host_scores=resolver_scores)
            action = list(env.actions(agent))[resolution.executed_index]
            family = str(resolution.executed_action_family)
            recovery_record = None
            if requested in (2, 3):
                target = resolution.target_host
                active_before = incident_active_at_action_start(target, previous)
                recovery_record = {
                    "tick_started": tick, "agent_name": agent,
                    "requested_action": "remove" if requested == 2 else "restore",
                    "executed_action": family.lower(), "target": target,
                    "fallback": bool(resolution.fallback),
                    "fallback_reason": resolution.fallback_reason,
                    "started": family in ("Remove", "Restore") and not resolution.fallback,
                    "completed": False, "active_incident_before": active_before,
                }
                recovery_actions.append(recovery_record)
            active[agent] = {"ready_at": tick + int(action.duration), "family": family,
                             "target": resolution.target_host, "recovery": recovery_record}
            joint[agent] = int(resolution.executed_index)
            decisions.append({
                "schema_version": 1, "protocol_version": PROTOCOL_VERSION,
                "formal_result_eligible": run_mode == "formal", "run_mode": run_mode,
                "method": method, "episode_seed": seed, "tick": tick, "agent_name": agent,
                "policy_seed": policy_seed,
                "runtime_seed": runtime_seeds[agent],
                "requested_action": requested, "executed_action": FAMILY_TO_A4[family],
                "executed_family": family, "target": resolution.target_host,
                "fallback": bool(resolution.fallback), "fallback_reason": resolution.fallback_reason,
                "root_action_mask": list(root_action_mask),
                "diagnostics": diagnostics,
            })

        next_obs, rewards, terminated, truncated, _ = env.step(actions=joint)
        tick_end = int(controller.step_count)
        if tick_end != tick + 1:
            raise RuntimeError("CC4 step did not advance exactly one tick")
        tick_rewards.append({agent: float(rewards[agent]) for agent in BLUE_AGENTS})

        for failure in green_local_work_failures(controller):
            owner = failure_owner_from_inventories(failure.hostname, inventories)
            event = {"tick": tick_end, "host": failure.hostname,
                     "raw_penalty": float(failure.raw_lwf_penalty)}
            if owner is None:
                excluded_failures.append({**event, "reason": "outside_blue_jurisdiction"})
            else:
                failures.append({**event, "agent_name": owner})

        current_truth = {host: ground_truth_red_presence(controller, host) for host in jurisdiction_hosts}
        for host in jurisdiction_hosts:
            if current_truth[host] and not previous[host]:
                item = {"incident_id": f"{seed}:{host}:{tick_end}", "host": host,
                        "t_compromise": tick_end, "t_recovered": None}
                incidents.append(item); open_incidents[host] = item
            elif previous[host] and not current_truth[host]:
                if host not in open_incidents:
                    raise RuntimeError(f"recovery without open incident: {host}")
                open_incidents.pop(host)["t_recovered"] = tick_end

        for agent in BLUE_AGENTS:
            if agent in done:
                continue
            meta = active[agent]
            completed = meta is not None and tick_end == int(meta["ready_at"])
            finished = bool(terminated.get(agent, False) or truncated.get(agent, False))
            success = observation_success_bool(next_obs[agent]) if completed else None
            trackers[agent].update(
                observation=next_obs[agent], global_tick=tick_end,
                completed_action_family=meta["family"] if completed else None,
                completed_target_host=meta["target"] if completed else None,
                completed_action_success=success if completed else None,
            )
            if method == "dca_cc4":
                policies[agent].observe(
                    next_obs[agent], tick=tick_end,
                    completed_family=meta["family"] if completed else None,
                    completed_target=meta["target"] if completed else None,
                    completed_success=success if completed else None,
                )
            current[agent] = next_obs[agent]
            if completed and meta["recovery"] is not None:
                meta["recovery"]["completed"] = True
                meta["recovery"]["tick_completed"] = tick_end
                meta["recovery"]["action_success"] = success
            if completed or finished:
                active[agent] = None
            if finished:
                done.add(agent)
        previous = current_truth

    if int(controller.step_count) != ticks or len(tick_rewards) != ticks:
        raise RuntimeError("episode did not complete the exact requested tick count")
    if uamcts_priors is not None and any(prior.misses for prior in uamcts_priors.values()):
        raise RuntimeError("UAMCTS offline prior cache miss (fail closed)")
    episode = {
        "schema_version": 1, "protocol_version": PROTOCOL_VERSION,
        "episode_seed": seed, "tick_count": ticks, "episode_end_tick": ticks,
        "tick_team_rewards": tick_rewards, "operation_failure_events": failures,
        "excluded_non_blue_failure_events": excluded_failures,
        "recovery_actions": recovery_actions, "incidents": incidents,
        "method": method, "run_mode": run_mode,
        "policy_seed": policy_seed,
        "formal_result_eligible": run_mode == "formal",
    }
    compute_episode_metrics(episode, expected_ticks=ticks)
    return episode, decisions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ticks", type=int, default=20)
    parser.add_argument("--run-mode", choices=("dev", "formal"), default="dev")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--policy-seed", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    episode, decisions = run_method_episode(method=args.method, seed=args.seed, ticks=args.ticks,
                                            run_mode=args.run_mode, device=args.device,
                                            policy_seed=args.policy_seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "episode.json").write_text(json.dumps(episode, indent=2, sort_keys=True), encoding="utf-8")
    with (args.out / "decisions.jsonl").open("w", encoding="utf-8") as handle:
        for decision in decisions:
            handle.write(json.dumps(decision, sort_keys=True) + "\n")
    metrics = compute_episode_metrics(episode, expected_ticks=args.ticks).to_dict()
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"passed": True, "formal_result_eligible": episode["formal_result_eligible"],
                      "decisions": len(decisions), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
