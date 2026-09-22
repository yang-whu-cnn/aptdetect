from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Callable

import numpy as np
import torch

from formal_experiments.data_collection.collect_cc4_formal_replay import (
    encode_decision_state,
    green_local_work_failures,
    make_env,
    observation_success_bool,
    red_presence_for_hosts,
)
from formal_experiments.data_collection.decision_replay import DecisionEpochReplayCollector
from formal_experiments.data_collection.incident_response import IncidentResponseBookkeeper
from formal_experiments.evaluation.run_b2_multi_model_preflight import make_live_client
from formal_experiments.evaluation.run_b2_prior_quality import (
    DEFAULT_REWARD_MODEL,
    DEFAULT_WORLD_MODEL,
    build_evaluator,
    request_prior_once_logical_transaction,
    resolve_project_path,
)
from formal_experiments.ours.lwm_runtime import LWMDecisionRuntime, PreparedCandidates
from formal_experiments.ours.model_registry import DEFAULT_REGISTRY, load_model_registry
from formal_experiments.ours.ppo_core import CandidateActorCritic, PPOCoreConfig
from formal_experiments.ours.ppo_training import (
    AsyncPPORolloutBuffer,
    FORMAL_TRAIN_SEEDS,
    PPOTrainer,
    PPOTrainingConfig,
    RolloutStep,
    build_training_batch,
)
from formal_experiments.ours.prior_cache import CACHE_FORMAT_VERSION, PriorCache
from formal_experiments.ours.provisional_protocol import (
    DEFAULT_PROVISIONAL_SCENARIO_STEPS,
    DEFAULT_PROVISIONAL_STAGE_TARGET,
    FORMAL_ROLLOUT_TARGET,
    MAX_PROVISIONAL_TRANSITIONS,
    UpdateRecord,
    append_probe_records,
    count_probe_records,
    deserialize_rollout_step,
    safe_episode_steps_for_remaining,
    safe_update_barrier,
    serialize_rollout_step,
    validate_probe_record,
    validate_stage_target,
)
from formal_experiments.training.bootstrap_world_model import EXECUTED_FAMILY_TO_ACTION_ID
from shared.action_contract import get_action
from shared.cyborg_action_adapter import CybORGActionAdapter
from shared.formal_state import BLUE_AGENTS, FormalStateEncoder, ObservableHostEvidenceTracker
from shared.model_space_action import canonicalize_requested_action


THIS = Path(__file__).resolve()
PROJ = THIS.parents[2]
DEFAULT_CACHE_ROOT = "outputs/lwm_rl_v2/prior_cache"
DEFAULT_OUT_ROOT = "outputs/lwm_rl_v2/b4/provisional"
DEFAULT_PPO_SEED = 20260917
CHECKPOINT_FORMAT_VERSION = 1


def _file_sha256(path: str | Path) -> str:
    return hashlib.sha256(resolve_project_path(path).read_bytes()).hexdigest()


def _exact_state_match(left, right) -> bool:
    a = np.asarray(left, dtype=np.float32)
    b = np.asarray(right, dtype=np.float32)
    return a.shape == (27,) and b.shape == (27,) and bool(np.array_equal(a, b))


def _assert_runtime_resolution(*, action_id: int, family_availability: dict, resolution) -> None:
    requested = get_action(int(action_id))
    family = str(requested.cyborg_action)
    if family == "Sleep":
        if bool(resolution.fallback) or str(resolution.executed_action_family) != "Sleep":
            raise RuntimeError("requested Sleep must execute Sleep without fallback")
        return
    if family not in family_availability:
        raise RuntimeError(f"missing planner-visible availability for {family}")
    available = bool(family_availability[family])
    if bool(resolution.fallback) == available:
        raise RuntimeError("adapter fallback disagrees with planner-visible availability")
    expected_family = family if available else "Sleep"
    if str(resolution.executed_action_family) != expected_family:
        raise RuntimeError("adapter executed family violates shared resolver contract")


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=str(path.parent), prefix=".provisional.", suffix=".tmp", delete=False) as handle:
        temp = Path(handle.name)
    try:
        torch.save(payload, temp)
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _policy_snapshot(policy: CandidateActorCritic) -> torch.Tensor:
    return torch.cat([p.detach().cpu().reshape(-1) for p in policy.parameters()]).clone()


def _all_finite_update(record: dict) -> bool:
    metrics = record.get("metrics", {})
    for key in (
        "policy_loss_mean",
        "value_loss_mean",
        "entropy_mean",
        "approx_kl_mean",
        "clip_fraction_mean",
        "grad_norm_mean",
        "grad_norm_max",
    ):
        try:
            if not np.isfinite(float(metrics[key])):
                return False
        except Exception:
            return False
    return True


def _checkpoint_payload(
    *,
    model_alias: str,
    exact_model_id: str,
    registry_sha256: str,
    world_model_sha256: str,
    reward_model_sha256: str,
    ppo_seed: int,
    policy: CandidateActorCritic,
    trainer: PPOTrainer,
    total_transitions: int,
    next_episode_ordinal: int,
    policy_version: int,
    pending_steps: list[RolloutStep],
    update_history: list[dict],
    episode_history: list[dict],
    cumulative_cache_hits: int,
    cumulative_cache_misses: int,
    cumulative_live_api_calls: int,
    cumulative_cost_usd: float,
    cumulative_live_latency_ms: float,
) -> dict:
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "development_only": True,
        "formal_result_eligible": False,
        "model_alias": str(model_alias),
        "exact_model_id": str(exact_model_id),
        "registry_sha256": str(registry_sha256),
        "world_model_sha256": str(world_model_sha256),
        "reward_model_sha256": str(reward_model_sha256),
        "ppo_seed": int(ppo_seed),
        "total_transitions": int(total_transitions),
        "next_episode_ordinal": int(next_episode_ordinal),
        "policy_version": int(policy_version),
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "trainer_update_index": int(trainer.update_index),
        "pending_steps": [serialize_rollout_step(step) for step in pending_steps],
        "update_history": list(update_history),
        "episode_history": list(episode_history),
        "cumulative_cache_hits": int(cumulative_cache_hits),
        "cumulative_cache_misses": int(cumulative_cache_misses),
        "cumulative_live_api_calls": int(cumulative_live_api_calls),
        "cumulative_cost_usd": float(cumulative_cost_usd),
        "cumulative_live_latency_ms": float(cumulative_live_latency_ms),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return payload


def _load_checkpoint(path: Path, *, map_location: str | torch.device) -> dict:
    payload = torch.load(path, map_location=map_location)
    if int(payload.get("format_version", -1)) != CHECKPOINT_FORMAT_VERSION:
        raise RuntimeError("unsupported provisional checkpoint format")
    if not bool(payload.get("development_only")) or bool(payload.get("formal_result_eligible")):
        raise RuntimeError("provisional checkpoint eligibility flags are invalid")
    return payload


def run_provisional_stage(
    *,
    model_alias: str,
    stage_target: int = DEFAULT_PROVISIONAL_STAGE_TARGET,
    scenario_steps: int = DEFAULT_PROVISIONAL_SCENARIO_STEPS,
    ppo_seed: int = DEFAULT_PPO_SEED,
    registry_path: str | Path = DEFAULT_REGISTRY,
    world_model_path: str | Path = DEFAULT_WORLD_MODEL,
    reward_model_path: str | Path = DEFAULT_REWARD_MODEL,
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
    out_root: str | Path = DEFAULT_OUT_ROOT,
    device: str = "cpu",
    resume: bool = False,
    client_factory: Callable[[], object] = make_live_client,
) -> dict:
    target = validate_stage_target(stage_target)
    if target > MAX_PROVISIONAL_TRANSITIONS:
        raise ValueError("provisional target exceeds frozen 20k cap")
    if int(scenario_steps) < 5:
        raise ValueError("scenario_steps must be >=5")

    registry_file = resolve_project_path(registry_path)
    registry = load_model_registry(registry_file)
    if bool(registry.primary_model_selected):
        raise RuntimeError("provisional PPO cannot run after premature primary-model selection")
    spec = registry.by_alias(str(model_alias))
    if not bool(spec.structured_output_verified) or spec.actual_temperature is None:
        raise RuntimeError("provisional model must have passed B2 live capability verification")

    out_dir = resolve_project_path(out_root) / spec.experiment_alias
    checkpoint_path = out_dir / "resume.pt"
    probe_path = out_dir / "probe_transitions.jsonl"
    report_path = out_dir / f"stage_{target}.json"

    registry_sha = hashlib.sha256(registry_file.read_bytes()).hexdigest()
    world_sha = _file_sha256(world_model_path)
    reward_sha = _file_sha256(reward_model_path)

    torch_device = torch.device(device)
    torch.manual_seed(int(ppo_seed))
    np.random.seed(int(ppo_seed) % (2**32 - 1))
    policy = CandidateActorCritic(PPOCoreConfig()).to(torch_device)
    trainer = PPOTrainer(
        policy,
        training_config=PPOTrainingConfig(seed=int(ppo_seed)),
        device=torch_device,
    )

    total_transitions = 0
    next_episode_ordinal = 0
    policy_version = 0
    pending_steps: list[RolloutStep] = []
    update_history: list[dict] = []
    episode_history: list[dict] = []
    base_cache_hits = 0
    base_cache_misses = 0
    base_live_calls = 0
    base_cost = 0.0
    base_latency = 0.0

    if resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"resume requested but checkpoint missing: {checkpoint_path}")
        saved = _load_checkpoint(checkpoint_path, map_location=torch_device)
        expected = {
            "model_alias": spec.experiment_alias,
            "exact_model_id": spec.exact_model_id,
            "registry_sha256": registry_sha,
            "world_model_sha256": world_sha,
            "reward_model_sha256": reward_sha,
            "ppo_seed": int(ppo_seed),
        }
        for key, value in expected.items():
            if saved.get(key) != value:
                raise RuntimeError(f"resume provenance mismatch: {key}")
        policy.load_state_dict(saved["policy_state_dict"])
        trainer.optimizer.load_state_dict(saved["optimizer_state_dict"])
        trainer.update_index = int(saved["trainer_update_index"])
        total_transitions = int(saved["total_transitions"])
        next_episode_ordinal = int(saved["next_episode_ordinal"])
        policy_version = int(saved["policy_version"])
        pending_steps = [deserialize_rollout_step(item) for item in saved.get("pending_steps", [])]
        update_history = list(saved.get("update_history", []))
        episode_history = list(saved.get("episode_history", []))
        base_cache_hits = int(saved.get("cumulative_cache_hits", 0))
        base_cache_misses = int(saved.get("cumulative_cache_misses", 0))
        base_live_calls = int(saved.get("cumulative_live_api_calls", 0))
        base_cost = float(saved.get("cumulative_cost_usd", 0.0))
        base_latency = float(saved.get("cumulative_live_latency_ms", 0.0))
        torch.set_rng_state(saved["torch_rng_state"].cpu())
        np.random.set_state(saved["numpy_rng_state"])
        if torch.cuda.is_available() and "cuda_rng_state_all" in saved:
            torch.cuda.set_rng_state_all(saved["cuda_rng_state_all"])
        if count_probe_records(probe_path) != total_transitions:
            raise RuntimeError("resume probe JSONL count does not match checkpoint")
    else:
        if checkpoint_path.exists() or probe_path.exists():
            raise RuntimeError(
                "provisional artifacts already exist; use --resume or move/delete the development artifacts explicitly"
            )

    if total_transitions > target:
        raise RuntimeError("resume checkpoint already exceeds requested stage target")

    evaluator = build_evaluator(
        world_model_path=world_model_path,
        reward_model_path=reward_model_path,
        device=device,
    )
    prior_cache = PriorCache(resolve_project_path(cache_root))
    client_holder: dict[str, object] = {}

    def live_prior_generator(state: np.ndarray, agent_name: str):
        if "client" not in client_holder:
            client_holder["client"] = client_factory()
        return request_prior_once_logical_transaction(
            spec=spec,
            registry=registry,
            state=state,
            agent_name=agent_name,
            client=client_holder["client"],
        )

    runtime = LWMDecisionRuntime(
        registry=registry,
        model_spec=spec,
        registry_sha256=registry_sha,
        evaluator=evaluator,
        prior_cache=prior_cache,
        policy=policy,
        live_prior_generator=live_prior_generator,
        split="train",
    )

    run_cost = 0.0
    run_latency = 0.0
    requested_counts = Counter()
    canonical_counts = Counter()
    executed_counts = Counter()
    fallback_count = 0
    response_reward_total = 0.0
    official_reward_total = 0.0
    plan0_match_count = 0
    alignment_count = 0
    action_completed_count = 0

    def prepare_with_cost(state, agent_name: str, *, root_tick: int,
                          episode_steps: int) -> PreparedCandidates:
        nonlocal run_cost, run_latency
        misses_before = runtime.cache_misses
        prepared = runtime.prepare(state, agent_name=agent_name, root_tick=root_tick,
                                   episode_steps=episode_steps)
        if runtime.cache_misses > misses_before:
            cost = prepared.metadata.get("cost_usd")
            if cost is not None:
                run_cost += float(cost)
            run_latency += float(prepared.metadata.get("latency_ms", 0.0))
        return prepared

    def perform_update(*, final_flush: bool, episode_buffer: AsyncPPORolloutBuffer, replay) -> None:
        nonlocal policy_version, pending_steps, update_history
        if not safe_update_barrier(
            len(pending_steps),
            ppo_open_agents=episode_buffer.open_agents,
            replay_open_agents=replay.open_agents,
            rollout_target=FORMAL_ROLLOUT_TARGET,
            final_flush=final_flush,
        ):
            return
        misses_before = runtime.cache_misses
        calls_before = runtime.live_api_calls
        batch = build_training_batch(
            pending_steps,
            core_config=policy.config,
            normalize_advantage=True,
        )
        metrics = trainer.update(batch)
        misses_after = runtime.cache_misses
        calls_after = runtime.live_api_calls
        if misses_before != misses_after or calls_before != calls_after:
            raise RuntimeError("PPO optimizer triggered prior-cache/provider activity")
        record = UpdateRecord(
            policy_version_before=policy_version,
            policy_version_after=policy_version + 1,
            batch_size=batch.size,
            rollout_target=FORMAL_ROLLOUT_TARGET,
            overshoot=max(0, batch.size - FORMAL_ROLLOUT_TARGET),
            final_flush=bool(final_flush),
            cache_misses_before=misses_before,
            cache_misses_after=misses_after,
            live_api_calls_before=calls_before,
            live_api_calls_after=calls_after,
            metrics=metrics,
        ).to_json()
        if not _all_finite_update(record):
            raise RuntimeError("provisional PPO update emitted non-finite metrics")
        update_history.append(record)
        policy_version += 1
        pending_steps = []
        episode_buffer.clear_completed()

    while total_transitions < target:
        remaining = target - total_transitions
        episode_steps = safe_episode_steps_for_remaining(
            remaining,
            default_steps=int(scenario_steps),
        )
        if episode_steps is None:
            break

        episode_ordinal = int(next_episode_ordinal)
        seed = int(FORMAL_TRAIN_SEEDS[episode_ordinal % len(FORMAL_TRAIN_SEEDS)])
        env, reset_observations, _reset_info = make_env(
            seed=seed,
            steps=episode_steps,
            pad_spaces=False,
        )
        controller = env.env.environment_controller
        adapter = CybORGActionAdapter()
        encoder = FormalStateEncoder()
        replay = DecisionEpochReplayCollector(episode_seed=seed)
        ppo_buffer = AsyncPPORolloutBuffer(train_only=True)

        trackers = {}
        bookkeepers = {}
        current_observation = {}
        decision_index = {agent: 0 for agent in BLUE_AGENTS}
        active = {agent: None for agent in BLUE_AGENTS}
        ready_context: dict[str, PreparedCandidates | None] = {agent: None for agent in BLUE_AGENTS}
        done_agents: set[str] = set()
        episode_records: list[dict] = []
        episode_response_reward = 0.0
        episode_official_reward = 0.0
        environment_steps = 0

        for agent in BLUE_AGENTS:
            if agent not in reset_observations:
                raise RuntimeError(f"{agent}: missing reset observation")
            tracker = ObservableHostEvidenceTracker(agent)
            tracker.reset(reset_observations[agent])
            trackers[agent] = tracker
            current_observation[agent] = reset_observations[agent]
            bookkeeper = IncidentResponseBookkeeper(episode_seed=seed, agent_name=agent)
            bookkeeper.reset(
                initial_red_presence_by_host=red_presence_for_hosts(controller, tracker.inventory),
                global_tick=int(controller.step_count),
            )
            bookkeepers[agent] = bookkeeper

        while len(done_agents) < len(BLUE_AGENTS):
            tick_start = int(controller.step_count)
            joint_actions = {}

            for agent in BLUE_AGENTS:
                if agent in done_agents or active[agent] is not None:
                    continue
                state, host_scores, family_availability = encode_decision_state(
                    env=env,
                    encoder=encoder,
                    tracker=trackers[agent],
                    observation=current_observation[agent],
                    agent_name=agent,
                    global_tick=tick_start,
                    episode_steps=episode_steps,
                )

                prepared = ready_context[agent]
                if prepared is None:
                    prepared = prepare_with_cost(state, agent, root_tick=tick_start,
                                                 episode_steps=episode_steps)
                else:
                    if not _exact_state_match(prepared.state, state):
                        raise RuntimeError(f"{agent}: prefetched context/state mismatch")
                    ready_context[agent] = None

                selection = runtime.select(prepared, deterministic=False)
                resolution = adapter.resolve(
                    env=env,
                    agent_name=agent,
                    action_id=selection.requested_action_id,
                    observable_host_scores=host_scores,
                )
                _assert_runtime_resolution(
                    action_id=selection.requested_action_id,
                    family_availability=family_availability,
                    resolution=resolution,
                )
                action_objects = list(env.actions(agent))
                executed_index = int(resolution.executed_index)
                if not 0 <= executed_index < len(action_objects):
                    raise RuntimeError(f"{agent}: invalid executed action index")
                action_object = action_objects[executed_index]
                actual_family = type(action_object).__name__
                if actual_family != str(resolution.executed_action_family):
                    raise RuntimeError(f"{agent}: adapter/action-object family mismatch")
                duration = int(action_object.duration)
                if duration <= 0:
                    raise RuntimeError(f"{agent}: invalid executed action duration")

                index = int(decision_index[agent])
                replay.begin_decision(
                    agent_name=agent,
                    decision_index=index,
                    global_tick_start=tick_start,
                    state=state,
                    resolution=resolution,
                )
                ppo_buffer.begin(
                    episode_seed=seed,
                    agent_name=agent,
                    decision_index=index,
                    candidate_features=prepared.candidate_features,
                    candidate_index=selection.candidate_index,
                    behavior_log_prob=selection.behavior_log_prob,
                    critic_value=selection.critic_value,
                )
                active[agent] = {
                    "ready_at": tick_start + duration,
                    "executed_family": actual_family,
                    "target_host": resolution.target_host,
                    "selection": selection,
                    "prepared": prepared,
                    "policy_version": int(policy_version),
                }
                joint_actions[agent] = executed_index
                requested_counts[selection.requested_action_id] += 1
                executed_counts[actual_family] += 1
                fallback_count += int(bool(resolution.fallback))

            if not replay.open_agents or not ppo_buffer.open_agents:
                raise RuntimeError("provisional scheduler has no open Blue decisions before env.step")

            observations, rewards, terminated, truncated, _info = env.step(actions=joint_actions)
            environment_steps += 1
            tick_end = int(controller.step_count)
            if tick_end != tick_start + 1:
                raise RuntimeError("CC4 must advance exactly one global tick")
            all_lwf = green_local_work_failures(controller)

            for agent in BLUE_AGENTS:
                if agent in done_agents:
                    continue
                if agent not in observations:
                    raise RuntimeError(f"{agent}: missing post-step observation")
                observation = observations[agent]
                tracker = trackers[agent]
                scoped_lwf = [
                    failure for failure in all_lwf if failure.hostname in set(tracker.inventory)
                ]
                accounting = bookkeepers[agent].record_tick(
                    global_tick_end=tick_end,
                    red_presence_after=red_presence_for_hosts(controller, tracker.inventory),
                    local_work_failures=scoped_lwf,
                )
                replay.record_tick(
                    agent_name=agent,
                    global_tick_end=tick_end,
                    official_reward=float(rewards.get(agent, 0.0)),
                    **accounting.to_replay_kwargs(),
                )

                meta = active[agent]
                if meta is None:
                    raise RuntimeError(f"{agent}: open decision missing active metadata")
                ready_at = int(meta["ready_at"])
                if tick_end > ready_at:
                    raise RuntimeError(f"{agent}: scheduler missed decision epoch")
                completed = tick_end == ready_at
                done = bool(terminated.get(agent, False) or truncated.get(agent, False))
                completion_success = observation_success_bool(observation) if completed else None

                tracker.update(
                    observation=observation,
                    global_tick=tick_end,
                    completed_action_family=str(meta["executed_family"]) if completed else None,
                    completed_target_host=meta["target_host"] if completed else None,
                    completed_action_success=completion_success if completed else None,
                )
                current_observation[agent] = observation

                if completed or done:
                    next_state, _next_scores, _next_availability = encode_decision_state(
                        env=env,
                        encoder=encoder,
                        tracker=tracker,
                        observation=observation,
                        agent_name=agent,
                        global_tick=tick_end,
                        episode_steps=episode_steps,
                    )
                    if done:
                        next_prepared = None
                        next_value = 0.0
                    else:
                        next_prepared = prepare_with_cost(next_state, agent, root_tick=tick_end,
                                                         episode_steps=episode_steps)
                        next_value = float(next_prepared.critic_value)

                    transition = replay.end_decision(
                        agent_name=agent,
                        global_tick_end=tick_end,
                        next_state=next_state,
                        done=done,
                        completed_action_success=completion_success if completed else None,
                    )
                    ppo_step = ppo_buffer.complete(
                        agent_name=agent,
                        real_response_reward=float(transition.response_reward),
                        decision_dt=int(transition.decision_dt),
                        done=bool(transition.done),
                        critic_next_value=float(next_value),
                    )
                    pending_steps.append(ppo_step)
                    total_transitions += 1

                    selection = meta["selection"]
                    prepared = meta["prepared"]
                    if int(transition.requested_action_id) != int(selection.requested_action_id):
                        raise RuntimeError("replay requested action differs from selected plan[0]")
                    if int(selection.requested_action_id) != int(selection.selected_plan[0]):
                        raise RuntimeError("selected requested action differs from plan[0]")
                    canonical = canonicalize_requested_action(
                        transition.state,
                        transition.requested_action_id,
                    )
                    expected_executed = EXECUTED_FAMILY_TO_ACTION_ID[str(transition.executed_action_family)]
                    if int(canonical) != int(expected_executed):
                        raise RuntimeError("model-space canonical action differs from executed family")
                    canonical_counts[int(canonical)] += 1

                    record = {
                        "model_alias": spec.experiment_alias,
                        "policy_version": int(meta["policy_version"]),
                        "episode_ordinal": int(episode_ordinal),
                        "episode_seed": int(seed),
                        "agent_name": str(agent),
                        "decision_index": int(transition.decision_index),
                        "global_tick_start": int(transition.global_tick_start),
                        "global_tick_end": int(transition.global_tick_end),
                        "state": np.asarray(transition.state, dtype=np.float32).tolist(),
                        "next_state": np.asarray(transition.next_state, dtype=np.float32).tolist(),
                        "requested_action_id": int(transition.requested_action_id),
                        "canonical_action_id": int(canonical),
                        "executed_action_family": str(transition.executed_action_family),
                        "fallback": bool(transition.fallback),
                        "action_completed": bool(transition.action_completed),
                        "selected_candidate_index": int(selection.candidate_index),
                        "selected_plan": list(selection.selected_plan),
                        "response_reward": float(transition.response_reward),
                        "official_reward": float(transition.official_reward),
                        "decision_dt": int(transition.decision_dt),
                        "done": bool(transition.done),
                        "cache_key": str(prepared.cache_key),
                        "cache_hit": bool(prepared.cache_hit),
                    }
                    validate_probe_record(record)
                    episode_records.append(record)
                    response_reward_total += float(transition.response_reward)
                    official_reward_total += float(transition.official_reward)
                    episode_response_reward += float(transition.response_reward)
                    episode_official_reward += float(transition.official_reward)
                    plan0_match_count += 1
                    alignment_count += 1
                    action_completed_count += int(bool(transition.action_completed))

                    ready_context[agent] = next_prepared
                    active[agent] = None
                    decision_index[agent] += 1

                if done:
                    done_agents.add(agent)
                    ready_context[agent] = None

            if safe_update_barrier(
                len(pending_steps),
                ppo_open_agents=ppo_buffer.open_agents,
                replay_open_agents=replay.open_agents,
                rollout_target=FORMAL_ROLLOUT_TARGET,
            ):
                perform_update(final_flush=False, episode_buffer=ppo_buffer, replay=replay)
                for agent in BLUE_AGENTS:
                    if agent not in done_agents:
                        ready_context[agent] = None

            if environment_steps > episode_steps + 2:
                raise RuntimeError("provisional episode exceeded scenario step bound")

        if replay.open_agents or ppo_buffer.open_agents:
            raise RuntimeError("provisional episode ended with open decisions")
        if len(episode_records) != len(replay.buffer.transitions):
            raise RuntimeError("episode probe record count differs from formal replay")

        append_probe_records(probe_path, episode_records)
        episode_history.append(
            {
                "episode_ordinal": int(episode_ordinal),
                "seed": int(seed),
                "scenario_steps": int(episode_steps),
                "post_reset_env_steps": int(environment_steps),
                "transition_count": int(len(episode_records)),
                "response_reward_total": float(episode_response_reward),
                "official_reward_total": float(episode_official_reward),
                "policy_version_end": int(policy_version),
            }
        )
        next_episode_ordinal += 1

        cumulative_cache_hits = base_cache_hits + runtime.cache_hits
        cumulative_cache_misses = base_cache_misses + runtime.cache_misses
        cumulative_live_calls = base_live_calls + runtime.live_api_calls
        cumulative_cost = base_cost + run_cost
        cumulative_latency = base_latency + run_latency
        checkpoint = _checkpoint_payload(
            model_alias=spec.experiment_alias,
            exact_model_id=spec.exact_model_id,
            registry_sha256=registry_sha,
            world_model_sha256=world_sha,
            reward_model_sha256=reward_sha,
            ppo_seed=ppo_seed,
            policy=policy,
            trainer=trainer,
            total_transitions=total_transitions,
            next_episode_ordinal=next_episode_ordinal,
            policy_version=policy_version,
            pending_steps=pending_steps,
            update_history=update_history,
            episode_history=episode_history,
            cumulative_cache_hits=cumulative_cache_hits,
            cumulative_cache_misses=cumulative_cache_misses,
            cumulative_live_api_calls=cumulative_live_calls,
            cumulative_cost_usd=cumulative_cost,
            cumulative_live_latency_ms=cumulative_latency,
        )
        _atomic_torch_save(checkpoint, checkpoint_path)

    # Stage-final safe flush so continuation starts with a clean on-policy buffer.
    if pending_steps:
        dummy_replay = type("ClosedReplay", (), {"open_agents": ()})()
        dummy_buffer = AsyncPPORolloutBuffer(train_only=True)
        perform_update(final_flush=True, episode_buffer=dummy_buffer, replay=dummy_replay)

    cumulative_cache_hits = base_cache_hits + runtime.cache_hits
    cumulative_cache_misses = base_cache_misses + runtime.cache_misses
    cumulative_live_calls = base_live_calls + runtime.live_api_calls
    cumulative_cost = base_cost + run_cost
    cumulative_latency = base_latency + run_latency
    final_checkpoint = _checkpoint_payload(
        model_alias=spec.experiment_alias,
        exact_model_id=spec.exact_model_id,
        registry_sha256=registry_sha,
        world_model_sha256=world_sha,
        reward_model_sha256=reward_sha,
        ppo_seed=ppo_seed,
        policy=policy,
        trainer=trainer,
        total_transitions=total_transitions,
        next_episode_ordinal=next_episode_ordinal,
        policy_version=policy_version,
        pending_steps=pending_steps,
        update_history=update_history,
        episode_history=episode_history,
        cumulative_cache_hits=cumulative_cache_hits,
        cumulative_cache_misses=cumulative_cache_misses,
        cumulative_live_api_calls=cumulative_live_calls,
        cumulative_cost_usd=cumulative_cost,
        cumulative_live_latency_ms=cumulative_latency,
    )
    _atomic_torch_save(final_checkpoint, checkpoint_path)

    probe_count = count_probe_records(probe_path)
    shortfall = int(target - total_transitions)
    report = {
        "phase": "gate_b4_provisional_ppo",
        "development_only": True,
        "formal_result_eligible": False,
        "model_alias": spec.experiment_alias,
        "exact_model_id": spec.exact_model_id,
        "stage_target": int(target),
        "transition_count": int(total_transitions),
        "stage_shortfall": int(shortfall),
        "max_transition_cap": int(MAX_PROVISIONAL_TRANSITIONS),
        "rollout_target": int(FORMAL_ROLLOUT_TARGET),
        "ppo_seed": int(ppo_seed),
        "episode_count": int(next_episode_ordinal),
        "policy_version": int(policy_version),
        "registry_sha256": registry_sha,
        "world_model_sha256": world_sha,
        "reward_model_sha256": reward_sha,
        "cache_format_version": int(CACHE_FORMAT_VERSION),
        "primary_model_selected_is_false": not bool(registry.primary_model_selected),
        "probe_path": str(probe_path),
        "probe_record_count": int(probe_count),
        "checkpoint_path": str(checkpoint_path),
        "requested_action_counts": {str(i): int(requested_counts.get(i, 0)) for i in range(4)},
        "canonical_action_counts": {str(i): int(canonical_counts.get(i, 0)) for i in range(4)},
        "executed_action_family_counts": dict(sorted(executed_counts.items())),
        "fallback_count_this_run": int(fallback_count),
        "action_completed_count_this_run": int(action_completed_count),
        "plan0_match_count_this_run": int(plan0_match_count),
        "alignment_count_this_run": int(alignment_count),
        "response_reward_total_this_run": float(response_reward_total),
        "official_reward_total_this_run": float(official_reward_total),
        "cache_hits_cumulative": int(cumulative_cache_hits),
        "cache_misses_cumulative": int(cumulative_cache_misses),
        "live_api_calls_cumulative": int(cumulative_live_calls),
        "estimated_cost_usd_cumulative": float(cumulative_cost),
        "live_latency_ms_cumulative": float(cumulative_latency),
        "update_count": len(update_history),
        "update_history": update_history,
        "episode_history": episode_history,
    }
    regular_updates = [item for item in update_history if not bool(item.get("final_flush"))]
    report["safe_update_barriers"] = bool(
        all(
            int(item["batch_size"]) >= FORMAL_ROLLOUT_TARGET
            and int(item["cache_misses_before"]) == int(item["cache_misses_after"])
            and int(item["live_api_calls_before"]) == int(item["live_api_calls_after"])
            and _all_finite_update(item)
            for item in regular_updates
        )
        and all(_all_finite_update(item) for item in update_history)
    )
    report["pass"] = bool(
        0 < total_transitions <= target
        and 0 <= shortfall < len(BLUE_AGENTS) * 5
        and probe_count == total_transitions
        and report["primary_model_selected_is_false"]
        and report["safe_update_barriers"]
        and all(int(item["seed"]) in FORMAL_TRAIN_SEEDS for item in episode_history)
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate B4 staged train-only provisional PPO collection")
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--stage-target", type=int, default=DEFAULT_PROVISIONAL_STAGE_TARGET)
    parser.add_argument("--steps", type=int, default=DEFAULT_PROVISIONAL_SCENARIO_STEPS)
    parser.add_argument("--ppo-seed", type=int, default=DEFAULT_PPO_SEED)
    parser.add_argument("--registry", default=DEFAULT_REGISTRY)
    parser.add_argument("--world-model", default=DEFAULT_WORLD_MODEL)
    parser.add_argument("--reward-model", default=DEFAULT_REWARD_MODEL)
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    report = run_provisional_stage(
        model_alias=args.model_alias,
        stage_target=args.stage_target,
        scenario_steps=args.steps,
        ppo_seed=args.ppo_seed,
        registry_path=args.registry,
        world_model_path=args.world_model,
        reward_model_path=args.reward_model,
        cache_root=args.cache_root,
        out_root=args.out_root,
        device=args.device,
        resume=args.resume,
    )

    print("=" * 80)
    print("[GATE B4 PROVISIONAL PPO STAGE]")
    print("model:", report["model_alias"], report["exact_model_id"])
    print("stage target/completed:", report["stage_target"], report["transition_count"])
    print("stage shortfall:", report["stage_shortfall"])
    print("episodes/updates/policy_version:", report["episode_count"], report["update_count"], report["policy_version"])
    print("canonical actions:", report["canonical_action_counts"])
    print("cache hits/misses:", report["cache_hits_cumulative"], report["cache_misses_cumulative"])
    print("live API calls:", report["live_api_calls_cumulative"])
    print("estimated cost USD:", report["estimated_cost_usd_cumulative"])
    print("safe update barriers:", report["safe_update_barriers"])
    print("primary_model_selected:", not report["primary_model_selected_is_false"])
    print("pass:", report["pass"])
    print("[OK] report:", report_path_for_display(report, args.out_root))
    if not report["pass"]:
        raise SystemExit(1)


def report_path_for_display(report: dict, out_root: str | Path) -> Path:
    return resolve_project_path(out_root) / str(report["model_alias"]) / f"stage_{int(report['stage_target'])}.json"


if __name__ == "__main__":
    main()
