from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from baselines.ug_cem_apt.normalizer_bundle import (
    load_ug_normalizer_bundle,
    restore_agent_normalizer,
)
from formal_experiments.data_collection.collect_cc4_formal_replay import (
    encode_decision_state,
    make_env,
    observation_success_bool,
)
from formal_experiments.evaluation.calibrate_ug_normalizer import (
    DEFAULT_REWARD_MODEL,
    DEFAULT_STATES,
    DEFAULT_WORLD_MODEL,
    build_planner,
    load_calibration_records,
    resolve_project_path,
)
from formal_experiments.training.bootstrap_world_model import (
    BootstrapProbabilisticWorldModel,
)
from formal_experiments.training.response_reward_predictor import (
    ResponseRewardPredictor,
)
from shared.action_contract import ACTION_CONTRACTS, get_action
from shared.cyborg_action_adapter import CybORGActionAdapter
from shared.formal_state import (
    BLUE_AGENTS,
    FormalStateEncoder,
    ObservableHostEvidenceTracker,
)


DEFAULT_NORMALIZERS = "outputs/ug_cem_v2/step6/ug_normalizers_online.pt"
DEFAULT_OUT = "outputs/ug_cem_v2/step7/step7_smoke_report.json"

LOCAL_SHORT_SEEDS = (3000,)
LOCAL_SHORT_DECISIONS_PER_EPISODE = 20
LOCAL_LONG_SEEDS = (3000, 3001, 3002, 3003, 3004)
LOCAL_LONG_DECISIONS_PER_EPISODE = 100

OFFICIAL_TRAIN_SEEDS = (1000, 1001)
OFFICIAL_TICKS = 50

FAMILY_DURATION = {
    str(item.cyborg_action): int(item.duration_ticks)
    for item in ACTION_CONTRACTS
}


def build_runtime_planners(
    *,
    device: str,
    world_model_path: str | Path = DEFAULT_WORLD_MODEL,
    reward_model_path: str | Path = DEFAULT_REWARD_MODEL,
    normalizer_path: str | Path = DEFAULT_NORMALIZERS,
) -> tuple[dict, BootstrapProbabilisticWorldModel, ResponseRewardPredictor, dict]:
    world_model = BootstrapProbabilisticWorldModel.load_checkpoint(
        resolve_project_path(world_model_path),
        device=device,
    )
    reward_predictor = ResponseRewardPredictor.load_checkpoint(
        resolve_project_path(reward_model_path),
        device=device,
    )
    bundle = load_ug_normalizer_bundle(
        resolve_project_path(normalizer_path),
        map_location=device,
    )

    planners = {}
    for agent_index, agent_name in enumerate(BLUE_AGENTS):
        planner = build_planner(
            world_model=world_model,
            reward_predictor=reward_predictor,
            agent_index=agent_index,
            device=device,
        )
        restore_agent_normalizer(
            planner.uncertainty,
            bundle=bundle,
            agent_name=agent_name,
        )
        planner.set_uncertainty_update_mode(
            bool(bundle["online_updates_after_warmup"])
        )
        planner.reset_episode()
        planners[agent_name] = planner

    return planners, world_model, reward_predictor, bundle


def _ordered_local_records(records: list[dict], *, seed: int) -> list[dict]:
    agent_order = {agent: i for i, agent in enumerate(BLUE_AGENTS)}
    selected = [
        item for item in records
        if int(item["warmup"].episode_seed) == int(seed)
    ]
    selected.sort(
        key=lambda item: (
            item["global_tick_start"],
            agent_order[item["warmup"].agent_name],
            item["decision_index"],
        )
    )
    return selected


def run_local_state_smoke_profile(
    *,
    planners: dict,
    records: list[dict],
    seeds: tuple[int, ...],
    decisions_per_episode: int,
    profile_name: str,
) -> dict:
    if decisions_per_episode <= 0:
        raise ValueError("decisions_per_episode must be > 0")

    action_counts = Counter()
    latencies = []
    values = []
    uncertainties = []
    episode_summaries = []

    for seed in seeds:
        for planner in planners.values():
            planner.reset_episode()

        ordered = _ordered_local_records(records, seed=seed)
        if len(ordered) < decisions_per_episode:
            raise RuntimeError(
                f"local smoke seed {seed} has only {len(ordered)} states; "
                f"need {decisions_per_episode}"
            )

        episode_action_counts = Counter()
        for item in ordered[:decisions_per_episode]:
            warmup = item["warmup"]
            planner = planners[warmup.agent_name]
            result = planner.plan(warmup.state)

            if not 0 <= int(result.action_id) < 4:
                raise RuntimeError("local smoke planner returned invalid action")
            if not all(np.isfinite([
                result.best_score,
                result.expected_return,
                result.uncertainty,
                result.planning_latency_sec,
            ])):
                raise RuntimeError("local smoke produced non-finite planner diagnostics")

            action_counts[int(result.action_id)] += 1
            episode_action_counts[int(result.action_id)] += 1
            latencies.append(float(result.planning_latency_sec))
            values.append(float(result.expected_return))
            uncertainties.append(float(result.uncertainty))

        episode_summaries.append({
            "seed": int(seed),
            "planner_calls": int(decisions_per_episode),
            "action_counts": {str(i): int(episode_action_counts.get(i, 0)) for i in range(4)},
        })

    expected_total = len(seeds) * decisions_per_episode
    if sum(action_counts.values()) != expected_total:
        raise RuntimeError("local smoke planner-call count mismatch")

    return {
        "profile": profile_name,
        "development_only": True,
        "closed_loop_cc4": False,
        "seeds": list(seeds),
        "episodes": len(seeds),
        "decisions_per_episode": int(decisions_per_episode),
        "total_planner_calls": int(expected_total),
        "action_counts": {str(i): int(action_counts.get(i, 0)) for i in range(4)},
        "mean_planning_latency_sec": float(mean(latencies)),
        "max_planning_latency_sec": float(max(latencies)),
        "mean_expected_return": float(mean(values)),
        "mean_uncertainty": float(mean(uncertainties)),
        "episode_summaries": episode_summaries,
        "pass": True,
    }


def _assert_resolution_contract(*, state, action_id: int, resolution) -> None:
    requested = get_action(int(action_id))
    available = bool(float(state[17]) >= 0.5)
    if requested.cyborg_action == "Sleep":
        if bool(resolution.fallback) or str(resolution.executed_action_family) != "Sleep":
            raise RuntimeError("requested Sleep violated adapter contract")
        return

    if available:
        if bool(resolution.fallback):
            raise RuntimeError("valid targeted action unexpectedly fell back")
        if str(resolution.executed_action_family) != str(requested.cyborg_action):
            raise RuntimeError("valid targeted action changed family")
    else:
        if not bool(resolution.fallback):
            raise RuntimeError("unavailable targeted action did not fall back")
        if str(resolution.executed_action_family) != "Sleep":
            raise RuntimeError("fallback must execute Sleep")


def run_official_cc4_episode(
    *,
    seed: int,
    steps: int,
    planners: dict,
) -> dict:
    env, reset_observations, _ = make_env(
        seed=int(seed),
        steps=int(steps),
        pad_spaces=False,
    )
    controller = env.env.environment_controller
    adapter = CybORGActionAdapter()
    encoder = FormalStateEncoder()

    trackers = {}
    current_observation = {}
    active = {agent: None for agent in BLUE_AGENTS}
    done_agents = set()

    for agent in BLUE_AGENTS:
        planners[agent].reset_episode()
        if agent not in reset_observations:
            raise RuntimeError(f"{agent}: missing reset observation")
        tracker = ObservableHostEvidenceTracker(agent)
        tracker.reset(reset_observations[agent])
        trackers[agent] = tracker
        current_observation[agent] = reset_observations[agent]

    requested_counts = Counter()
    executed_counts = Counter()
    fallback_count = 0
    per_agent_decisions = Counter()
    official_rewards = Counter()
    latencies = []
    predicted_values = []
    uncertainties = []

    while int(controller.step_count) < int(steps) and len(done_agents) < len(BLUE_AGENTS):
        tick_start = int(controller.step_count)
        joint_actions = {}

        for agent in BLUE_AGENTS:
            if agent in done_agents or active[agent] is not None:
                continue

            tracker = trackers[agent]
            state, host_scores, _family_availability = encode_decision_state(
                env=env,
                encoder=encoder,
                tracker=tracker,
                observation=current_observation[agent],
                agent_name=agent,
                global_tick=tick_start,
                episode_steps=int(steps),
            )

            result = planners[agent].plan(state)
            action_id = int(result.action_id)
            resolution = adapter.resolve(
                env=env,
                agent_name=agent,
                action_id=action_id,
                observable_host_scores=host_scores,
            )
            _assert_resolution_contract(
                state=state,
                action_id=action_id,
                resolution=resolution,
            )

            action_objects = list(env.actions(agent))
            executed_index = int(resolution.executed_index)
            if not 0 <= executed_index < len(action_objects):
                raise RuntimeError(f"{agent}: invalid executed action index")
            action_object = action_objects[executed_index]
            actual_family = type(action_object).__name__
            if actual_family != str(resolution.executed_action_family):
                raise RuntimeError(f"{agent}: action adapter family mismatch")

            duration = int(action_object.duration)
            expected_duration = FAMILY_DURATION.get(actual_family)
            if expected_duration is None or duration != expected_duration:
                raise RuntimeError(
                    f"{agent}: executed duration mismatch family={actual_family} "
                    f"actual={duration} expected={expected_duration}"
                )

            active[agent] = {
                "ready_at": tick_start + duration,
                "executed_family": actual_family,
                "target_host": resolution.target_host,
            }
            joint_actions[agent] = executed_index

            requested_counts[action_id] += 1
            executed_counts[actual_family] += 1
            fallback_count += int(bool(resolution.fallback))
            per_agent_decisions[agent] += 1
            latencies.append(float(result.planning_latency_sec))
            predicted_values.append(float(result.expected_return))
            uncertainties.append(float(result.uncertainty))

        observations, rewards, terminated, truncated, _info = env.step(actions=joint_actions)
        tick_end = int(controller.step_count)
        if tick_end != tick_start + 1:
            raise RuntimeError("official smoke must advance exactly one global tick")

        for agent in BLUE_AGENTS:
            if agent in done_agents:
                continue
            if agent not in observations:
                raise RuntimeError(f"{agent}: missing observation")

            observation = observations[agent]
            official_rewards[agent] += float(rewards.get(agent, 0.0))
            meta = active[agent]
            if meta is None:
                raise RuntimeError(f"{agent}: scheduler lost active decision")

            ready_at = int(meta["ready_at"])
            if tick_end > ready_at:
                raise RuntimeError(f"{agent}: scheduler missed decision epoch")
            completed = tick_end == ready_at
            done = bool(terminated.get(agent, False) or truncated.get(agent, False))
            success = observation_success_bool(observation) if completed else None

            trackers[agent].update(
                observation=observation,
                global_tick=tick_end,
                completed_action_family=str(meta["executed_family"]) if completed else None,
                completed_target_host=meta["target_host"] if completed else None,
                completed_action_success=success if completed else None,
            )
            current_observation[agent] = observation

            if completed or done:
                active[agent] = None
            if done:
                done_agents.add(agent)

    ticks = int(controller.step_count)
    total_decisions = int(sum(per_agent_decisions.values()))
    if ticks != int(steps):
        raise RuntimeError(f"official smoke ended at tick {ticks}, expected {steps}")
    if total_decisions <= 0:
        raise RuntimeError("official smoke produced zero decisions")
    missing_agents = [agent for agent in BLUE_AGENTS if per_agent_decisions.get(agent, 0) <= 0]
    if missing_agents:
        raise RuntimeError(f"official smoke missing planner decisions for {missing_agents}")
    if not all(np.isfinite(latencies + predicted_values + uncertainties)):
        raise RuntimeError("official smoke planner diagnostics contain NaN/Inf")

    normalizer_finite = {}
    for agent in BLUE_AGENTS:
        u = planners[agent].uncertainty
        normalizer_finite[agent] = bool(
            u.obs_mean is not None
            and u.obs_std is not None
            and u.horizon_std is not None
            and torch.isfinite(u.obs_mean).all().item()
            and torch.isfinite(u.obs_std).all().item()
            and torch.isfinite(u.horizon_std).all().item()
        )
    if not all(normalizer_finite.values()):
        raise RuntimeError("official smoke produced non-finite normalizer")

    return {
        "seed": int(seed),
        "ticks": ticks,
        "total_decisions": total_decisions,
        "per_agent_decisions": {agent: int(per_agent_decisions.get(agent, 0)) for agent in BLUE_AGENTS},
        "requested_action_counts": {str(i): int(requested_counts.get(i, 0)) for i in range(4)},
        "executed_family_counts": {family: int(executed_counts.get(family, 0)) for family in ("Sleep","Analyse","Remove","Restore")},
        "fallback_count": int(fallback_count),
        "fallback_rate": float(fallback_count / total_decisions),
        "official_reward_by_agent": {agent: float(official_rewards.get(agent, 0.0)) for agent in BLUE_AGENTS},
        "mean_planning_latency_sec": float(mean(latencies)),
        "max_planning_latency_sec": float(max(latencies)),
        "mean_predicted_return": float(mean(predicted_values)),
        "mean_uncertainty": float(mean(uncertainties)),
        "normalizer_finite": normalizer_finite,
        "pass": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Step-7 UG-CEM integration smoke tests")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--states", default=DEFAULT_STATES)
    parser.add_argument("--world-model", default=DEFAULT_WORLD_MODEL)
    parser.add_argument("--reward-model", default=DEFAULT_REWARD_MODEL)
    parser.add_argument("--normalizers", default=DEFAULT_NORMALIZERS)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    records = load_calibration_records(args.states)

    # Local development smoke uses a fresh calibrated runtime.
    local_planners, _, _, _ = build_runtime_planners(
        device=str(args.device),
        world_model_path=args.world_model,
        reward_model_path=args.reward_model,
        normalizer_path=args.normalizers,
    )
    local_short = run_local_state_smoke_profile(
        planners=local_planners,
        records=records,
        seeds=LOCAL_SHORT_SEEDS,
        decisions_per_episode=LOCAL_SHORT_DECISIONS_PER_EPISODE,
        profile_name="1_episode_x_20_decisions",
    )

    local_planners, _, _, _ = build_runtime_planners(
        device=str(args.device),
        world_model_path=args.world_model,
        reward_model_path=args.reward_model,
        normalizer_path=args.normalizers,
    )
    local_long = run_local_state_smoke_profile(
        planners=local_planners,
        records=records,
        seeds=LOCAL_LONG_SEEDS,
        decisions_per_episode=LOCAL_LONG_DECISIONS_PER_EPISODE,
        profile_name="5_episodes_x_100_decisions",
    )

    # Official train-seed smoke uses one planner set across episodes;
    # episode reset clears MPC warm-start while source-faithful EMA continues.
    official_planners, _, _, _ = build_runtime_planners(
        device=str(args.device),
        world_model_path=args.world_model,
        reward_model_path=args.reward_model,
        normalizer_path=args.normalizers,
    )
    official = [
        run_official_cc4_episode(
            seed=seed,
            steps=OFFICIAL_TICKS,
            planners=official_planners,
        )
        for seed in OFFICIAL_TRAIN_SEEDS
    ]

    report = {
        "status": "PASS",
        "development_only_local_smoke": {
            "short": local_short,
            "long": local_long,
            "eligible_for_paper_table": False,
        },
        "official_train_seed_smoke": {
            "seeds": list(OFFICIAL_TRAIN_SEEDS),
            "ticks_per_episode": OFFICIAL_TICKS,
            "episodes": official,
            "eligible_for_paper_table": False,
            "gate_b_ppo_result": False,
        },
        "pass": True,
    }

    out = resolve_project_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("[STEP7 SUMMARY]")
    print("local_short_calls:", local_short["total_planner_calls"])
    print("local_long_calls:", local_long["total_planner_calls"])
    print("official_seeds:", list(OFFICIAL_TRAIN_SEEDS))
    print("official_ticks:", [item["ticks"] for item in official])
    print("official_decisions:", [item["total_decisions"] for item in official])
    print("pass:", True)
    print("[OK] report:", out)


if __name__ == "__main__":
    main()
