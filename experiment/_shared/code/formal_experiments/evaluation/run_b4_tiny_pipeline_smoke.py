from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
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
from formal_experiments.data_collection.decision_replay import (
    DecisionEpochReplayCollector,
    DecisionEpochTransition,
)
from formal_experiments.data_collection.incident_response import IncidentResponseBookkeeper
from formal_experiments.evaluation.run_b2_multi_model_preflight import make_live_client
from formal_experiments.evaluation.run_b2_prior_quality import (
    DEFAULT_REWARD_MODEL,
    DEFAULT_WORLD_MODEL,
    build_evaluator,
    request_prior_once_logical_transaction,
    resolve_project_path,
)
from formal_experiments.ours.model_registry import DEFAULT_REGISTRY, load_model_registry
from formal_experiments.ours.posterior_features import build_posterior_candidate_features
from formal_experiments.ours.ppo_core import CandidateActorCritic, PPOCoreConfig
from formal_experiments.ours.ppo_training import (
    AsyncPPORolloutBuffer,
    PPOTrainer,
    PPOTrainingConfig,
    RolloutStep,
    assert_formal_train_seed,
    build_training_batch,
)
from formal_experiments.ours.prior_cache import (
    CACHE_FORMAT_VERSION,
    PriorCache,
    exact_state_sha256,
    make_identity,
)
from formal_experiments.ours.ofox_prior_client import ofox_network_mode
from shared.action_contract import get_action
from shared.cyborg_action_adapter import CybORGActionAdapter
from shared.formal_state import BLUE_AGENTS, FormalStateEncoder, ObservableHostEvidenceTracker


THIS = Path(__file__).resolve()
PROJ = THIS.parents[2]
DEFAULT_CACHE_ROOT = "outputs/lwm_rl_v2/prior_cache"
DEFAULT_OUT = "outputs/lwm_rl_v2/b4/tiny_pipeline_smoke.json"
DEFAULT_MODEL_ALIAS = "llm_l_gemini35_flash_lite"
DEFAULT_TRAIN_SEED = 1000
DEFAULT_SCENARIO_STEPS = 20
DEFAULT_PPO_SEED = 20260917


@dataclass(frozen=True)
class CandidateContext:
    agent_name: str
    state: np.ndarray
    state_sha256: str
    plans: np.ndarray
    prior_preferences: np.ndarray
    candidate_features: torch.Tensor
    cache_hit: bool
    cache_key: str
    metadata: dict


@dataclass(frozen=True)
class DecisionAudit:
    episode_seed: int
    agent_name: str
    decision_index: int
    selected_candidate_index: int
    selected_plan: tuple[int, ...]
    requested_action_id: int
    cache_hit: bool
    cache_key: str
    state_sha256: str


def _file_sha256(path: str | Path) -> str:
    return hashlib.sha256(resolve_project_path(path).read_bytes()).hexdigest()


def _finite(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def selected_plan_first_action(plans, candidate_index: int) -> int:
    array = np.asarray(plans, dtype=np.int64)
    if array.shape != (6, 4):
        raise ValueError("formal plans must have shape [6,4]")
    index = int(candidate_index)
    if not 0 <= index < 6:
        raise ValueError("candidate index must lie in [0,5]")
    action = int(array[index, 0])
    if not 0 <= action < 4:
        raise ValueError("selected plan contains invalid first action")
    return action


def assert_runtime_resolution(*, action_id: int, family_availability: dict, resolution) -> None:
    requested = get_action(int(action_id))
    requested_family = str(requested.cyborg_action)
    if requested_family == "Sleep":
        if bool(resolution.fallback):
            raise RuntimeError("requested Sleep must never fallback")
        if str(resolution.executed_action_family) != "Sleep":
            raise RuntimeError("requested Sleep must execute Sleep")
        return

    if requested_family not in family_availability:
        raise RuntimeError(f"missing planner-visible availability for {requested_family}")
    expected_available = bool(family_availability[requested_family])
    if bool(resolution.fallback) == expected_available:
        raise RuntimeError("resolver fallback disagrees with planner-visible availability")
    if expected_available:
        if str(resolution.executed_action_family) != requested_family:
            raise RuntimeError("valid targeted action changed action family")
    elif str(resolution.executed_action_family) != "Sleep":
        raise RuntimeError("unavailable targeted action must fallback to Sleep")


def validate_completed_alignment(
    *,
    transition: DecisionEpochTransition,
    ppo_step: RolloutStep,
    audit: DecisionAudit,
) -> None:
    if int(transition.episode_seed) != int(audit.episode_seed):
        raise RuntimeError("transition/audit seed mismatch")
    if str(transition.agent_name) != str(audit.agent_name):
        raise RuntimeError("transition/audit agent mismatch")
    if int(transition.decision_index) != int(audit.decision_index):
        raise RuntimeError("transition/audit decision index mismatch")
    if int(transition.requested_action_id) != int(audit.requested_action_id):
        raise RuntimeError("requested action differs from selected plan[0]")
    if int(audit.requested_action_id) != int(audit.selected_plan[0]):
        raise RuntimeError("audit requested action differs from selected plan[0]")

    if int(ppo_step.episode_seed) != int(transition.episode_seed):
        raise RuntimeError("PPO/replay seed mismatch")
    if str(ppo_step.agent_name) != str(transition.agent_name):
        raise RuntimeError("PPO/replay agent mismatch")
    if int(ppo_step.decision_index) != int(transition.decision_index):
        raise RuntimeError("PPO/replay decision index mismatch")
    if int(ppo_step.candidate_index) != int(audit.selected_candidate_index):
        raise RuntimeError("PPO selected candidate mismatch")
    if abs(float(ppo_step.real_response_reward) - float(transition.response_reward)) > 1e-7:
        raise RuntimeError("PPO reward differs from formal replay response reward")
    if int(ppo_step.decision_dt) != int(transition.decision_dt):
        raise RuntimeError("PPO/replay decision_dt mismatch")
    if bool(ppo_step.done) != bool(transition.done):
        raise RuntimeError("PPO/replay done mismatch")


class CandidateContextBuilder:
    def __init__(
        self,
        *,
        model_alias: str,
        evaluator,
        cache_root: str | Path,
        registry_path: str | Path = DEFAULT_REGISTRY,
        client_factory: Callable[[], object] = make_live_client,
    ) -> None:
        self.registry_path = resolve_project_path(registry_path)
        self.registry = load_model_registry(self.registry_path)
        if bool(self.registry.primary_model_selected):
            raise RuntimeError("tiny smoke must not run after silently selecting a primary LLM")
        self.spec = self.registry.by_alias(str(model_alias))
        if not bool(self.spec.structured_output_verified):
            raise RuntimeError("tiny-smoke model must have passed structured-output preflight")
        if self.spec.actual_temperature is None:
            raise RuntimeError("tiny-smoke model must have verified actual temperature")

        self.registry_sha256 = hashlib.sha256(self.registry_path.read_bytes()).hexdigest()
        self.evaluator = evaluator
        self.cache = PriorCache(resolve_project_path(cache_root))
        self.client_factory = client_factory
        self._client = None
        self.cache_hits = 0
        self.cache_misses = 0
        self.live_transactions = 0
        self.cost_this_run = 0.0
        self.latency_ms_this_run = 0.0
        self.generated_contexts = 0
        self.generation_config = {
            "temperature": float(self.spec.actual_temperature),
            "structured_output": str(self.spec.structured_output_requested),
            "public_agent_identifier": True,
        }

    def build(self, *, state, agent_name: str) -> CandidateContext:
        state_np = np.asarray(state, dtype=np.float32)
        if state_np.shape != (27,) or not np.isfinite(state_np).all():
            raise ValueError("candidate context requires finite D27 state")
        agent = str(agent_name)
        if agent not in BLUE_AGENTS:
            raise ValueError(f"unsupported Blue agent: {agent}")

        identity = make_identity(
            split="train",
            model_alias=self.spec.experiment_alias,
            provider=self.spec.provider,
            exact_model_id=self.spec.exact_model_id,
            registry_version=self.registry.version,
            registry_sha256=self.registry_sha256,
            prompt_version=self.registry.prompt_version,
            agent_name=agent,
            generation_config=self.generation_config,
            state=state_np,
        )

        def generate():
            if self._client is None:
                self._client = self.client_factory()
            return request_prior_once_logical_transaction(
                spec=self.spec,
                registry=self.registry,
                state=state_np,
                agent_name=agent,
                client=self._client,
            )

        lookup = self.cache.get_or_generate(identity, generate)
        cached = lookup.cached_prior
        if bool(lookup.cache_hit):
            self.cache_hits += 1
        else:
            self.cache_misses += 1
            self.live_transactions += 1
            cost = cached.metadata.get("cost_usd")
            if cost is not None:
                self.cost_this_run += float(cost)
            self.latency_ms_this_run += float(cached.metadata.get("latency_ms", 0.0))

        rollout = self.evaluator.evaluate(state_np, cached.prior.plans)
        features = build_posterior_candidate_features(
            state_np,
            cached.prior.plans,
            cached.prior.prior_preferences,
            rollout,
        ).detach()
        if tuple(features.shape) != (6, 46) or not torch.isfinite(features).all():
            raise RuntimeError("candidate posterior features violate B3 contract")

        self.generated_contexts += 1
        return CandidateContext(
            agent_name=agent,
            state=state_np.copy(),
            state_sha256=exact_state_sha256(state_np),
            plans=np.asarray(cached.prior.plans, dtype=np.int64).copy(),
            prior_preferences=np.asarray(cached.prior.prior_preferences, dtype=np.float32).copy(),
            candidate_features=features,
            cache_hit=bool(lookup.cache_hit),
            cache_key=str(cached.cache_key),
            metadata=dict(cached.metadata),
        )


def _policy_snapshot(policy: CandidateActorCritic) -> torch.Tensor:
    values = [parameter.detach().cpu().reshape(-1) for parameter in policy.parameters()]
    return torch.cat(values, dim=0).clone()


def _context_value(policy: CandidateActorCritic, context: CandidateContext, device) -> float:
    with torch.no_grad():
        _logits, value = policy(context.candidate_features.to(device))
    return float(value.detach().cpu().item())


def _context_matches_state(context: CandidateContext, state) -> bool:
    return str(context.state_sha256) == str(exact_state_sha256(state))


def _report_pass(report: dict) -> bool:
    if int(report.get("cache_format_version", -1)) != CACHE_FORMAT_VERSION:
        return False
    if not bool(report.get("agent_aware_cache_identity")):
        return False
    if int(report.get("transition_count", 0)) <= 0:
        return False
    if int(report.get("transition_count", -1)) != int(report.get("ppo_step_count", -2)):
        return False
    per_agent = report.get("per_agent_transitions", {})
    if set(per_agent) != set(BLUE_AGENTS) or any(int(per_agent[a]) <= 0 for a in BLUE_AGENTS):
        return False
    if int(report.get("plan0_match_count", -1)) != int(report["transition_count"]):
        return False
    if int(report.get("alignment_count", -1)) != int(report["transition_count"]):
        return False
    if int(report.get("cache_misses_before_update", -1)) != int(report.get("cache_misses_after_update", -2)):
        return False
    if int(report.get("live_transactions_before_update", -1)) != int(report.get("live_transactions_after_update", -2)):
        return False
    if not bool(report.get("policy_parameters_changed")):
        return False
    if not bool(report.get("all_agents_done")):
        return False
    if not bool(report.get("primary_model_selected_is_false")):
        return False
    metrics = report.get("update_metrics", {})
    required = (
        "policy_loss_mean",
        "value_loss_mean",
        "entropy_mean",
        "approx_kl_mean",
        "clip_fraction_mean",
        "grad_norm_mean",
        "grad_norm_max",
    )
    return all(_finite(metrics.get(name)) for name in required)


def run_tiny_smoke(
    *,
    model_alias: str = DEFAULT_MODEL_ALIAS,
    seed: int = DEFAULT_TRAIN_SEED,
    scenario_steps: int = DEFAULT_SCENARIO_STEPS,
    ppo_seed: int = DEFAULT_PPO_SEED,
    registry_path: str | Path = DEFAULT_REGISTRY,
    world_model_path: str | Path = DEFAULT_WORLD_MODEL,
    reward_model_path: str | Path = DEFAULT_REWARD_MODEL,
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
    out_path: str | Path = DEFAULT_OUT,
    device: str = "cpu",
    client_factory: Callable[[], object] = make_live_client,
) -> dict:
    train_seed = assert_formal_train_seed(int(seed))
    if int(scenario_steps) < 5:
        raise ValueError("tiny smoke scenario_steps must be >=5")

    torch.manual_seed(int(ppo_seed))
    np.random.seed(int(ppo_seed) % (2**32 - 1))
    torch_device = torch.device(device)

    evaluator = build_evaluator(
        world_model_path=world_model_path,
        reward_model_path=reward_model_path,
        device=device,
    )
    context_builder = CandidateContextBuilder(
        model_alias=model_alias,
        evaluator=evaluator,
        cache_root=cache_root,
        registry_path=registry_path,
        client_factory=client_factory,
    )
    policy = CandidateActorCritic(PPOCoreConfig()).to(torch_device)
    trainer = PPOTrainer(
        policy,
        training_config=PPOTrainingConfig(seed=int(ppo_seed)),
        device=torch_device,
    )
    policy_before = _policy_snapshot(policy)

    env, reset_observations, _reset_info = make_env(
        seed=train_seed,
        steps=int(scenario_steps),
        pad_spaces=False,
    )
    controller = env.env.environment_controller
    adapter = CybORGActionAdapter()
    encoder = FormalStateEncoder()
    replay = DecisionEpochReplayCollector(episode_seed=train_seed)
    ppo_buffer = AsyncPPORolloutBuffer(train_only=True)

    trackers = {}
    bookkeepers = {}
    current_observation = {}
    decision_index = {agent: 0 for agent in BLUE_AGENTS}
    active = {agent: None for agent in BLUE_AGENTS}
    ready_context: dict[str, CandidateContext | None] = {agent: None for agent in BLUE_AGENTS}
    done_agents: set[str] = set()

    requested_counts = Counter()
    executed_counts = Counter()
    per_agent_transitions = Counter()
    fallback_count = 0
    plan0_match_count = 0
    alignment_count = 0
    completed_audits: list[dict] = []
    environment_steps = 0

    for agent in BLUE_AGENTS:
        if agent not in reset_observations:
            raise RuntimeError(f"{agent}: missing reset observation")
        tracker = ObservableHostEvidenceTracker(agent)
        tracker.reset(reset_observations[agent])
        trackers[agent] = tracker
        current_observation[agent] = reset_observations[agent]
        bookkeeper = IncidentResponseBookkeeper(episode_seed=train_seed, agent_name=agent)
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
                episode_steps=int(scenario_steps),
            )

            context = ready_context[agent]
            if context is None:
                context = context_builder.build(state=state, agent_name=agent)
            else:
                if not _context_matches_state(context, state):
                    raise RuntimeError(f"{agent}: prefetched next context does not match next decision state")
                ready_context[agent] = None

            with torch.no_grad():
                selected, log_prob, _entropy, value = policy.act(
                    context.candidate_features.to(torch_device),
                    deterministic=False,
                )
            candidate_index = int(selected.detach().cpu().item())
            requested_action_id = selected_plan_first_action(context.plans, candidate_index)
            selected_plan = tuple(int(x) for x in context.plans[candidate_index].tolist())

            resolution = adapter.resolve(
                env=env,
                agent_name=agent,
                action_id=requested_action_id,
                observable_host_scores=host_scores,
            )
            assert_runtime_resolution(
                action_id=requested_action_id,
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
                raise RuntimeError(f"{agent}: invalid executed duration")

            index = int(decision_index[agent])
            replay.begin_decision(
                agent_name=agent,
                decision_index=index,
                global_tick_start=tick_start,
                state=state,
                resolution=resolution,
            )
            ppo_buffer.begin(
                episode_seed=train_seed,
                agent_name=agent,
                decision_index=index,
                candidate_features=context.candidate_features,
                candidate_index=candidate_index,
                behavior_log_prob=float(log_prob.detach().cpu().item()),
                critic_value=float(value.detach().cpu().item()),
            )

            audit = DecisionAudit(
                episode_seed=train_seed,
                agent_name=agent,
                decision_index=index,
                selected_candidate_index=candidate_index,
                selected_plan=selected_plan,
                requested_action_id=requested_action_id,
                cache_hit=bool(context.cache_hit),
                cache_key=str(context.cache_key),
                state_sha256=str(context.state_sha256),
            )
            active[agent] = {
                "ready_at": tick_start + duration,
                "executed_family": actual_family,
                "target_host": resolution.target_host,
                "audit": audit,
            }
            joint_actions[agent] = executed_index
            requested_counts[requested_action_id] += 1
            executed_counts[actual_family] += 1
            fallback_count += int(bool(resolution.fallback))

        if not replay.open_agents:
            raise RuntimeError("tiny smoke has no open Blue decision before env.step")

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
            if agent not in replay.open_agents:
                raise RuntimeError(f"{agent}: scheduler lost replay decision")
            if agent not in ppo_buffer.open_agents:
                raise RuntimeError(f"{agent}: scheduler lost PPO decision")

            observation = observations[agent]
            tracker = trackers[agent]
            scoped_lwf = [
                failure
                for failure in all_lwf
                if failure.hostname in set(tracker.inventory)
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
                    episode_steps=int(scenario_steps),
                )

                if done:
                    next_value = 0.0
                    ready_context[agent] = None
                else:
                    next_context = context_builder.build(state=next_state, agent_name=agent)
                    next_value = _context_value(policy, next_context, torch_device)
                    ready_context[agent] = next_context

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
                audit = meta["audit"]
                validate_completed_alignment(
                    transition=transition,
                    ppo_step=ppo_step,
                    audit=audit,
                )
                plan0_match_count += 1
                alignment_count += 1
                per_agent_transitions[agent] += 1
                completed_audits.append(
                    {
                        **asdict(audit),
                        "selected_plan": list(audit.selected_plan),
                        "executed_action_family": str(transition.executed_action_family),
                        "fallback": bool(transition.fallback),
                        "response_reward": float(transition.response_reward),
                        "decision_dt": int(transition.decision_dt),
                        "done": bool(transition.done),
                    }
                )
                active[agent] = None
                decision_index[agent] += 1

            if done:
                done_agents.add(agent)
                ready_context[agent] = None

        if environment_steps > int(scenario_steps) + 2:
            raise RuntimeError("tiny smoke exceeded expected scenario step bound")

    if replay.open_agents or ppo_buffer.open_agents:
        raise RuntimeError("tiny smoke ended with unclosed decisions")
    if len(replay.buffer.transitions) != len(ppo_buffer.steps):
        raise RuntimeError("formal replay and PPO rollout counts differ")

    batch = build_training_batch(
        ppo_buffer.steps,
        core_config=policy.config,
        normalize_advantage=True,
    )
    misses_before_update = int(context_builder.cache_misses)
    transactions_before_update = int(context_builder.live_transactions)
    metrics = trainer.update(batch)
    misses_after_update = int(context_builder.cache_misses)
    transactions_after_update = int(context_builder.live_transactions)
    policy_after = _policy_snapshot(policy)
    policy_changed = not torch.equal(policy_before, policy_after)

    transition_count = len(replay.buffer.transitions)
    response_reward_total = float(sum(item.response_reward for item in replay.buffer.transitions))
    official_reward_total = float(sum(item.official_reward for item in replay.buffer.transitions))

    report = {
        "phase": "gate_b4_tiny_pipeline_smoke",
        "development_only": True,
        "formal_result_eligible": False,
        "seed": int(train_seed),
        "scenario_steps": int(scenario_steps),
        "post_reset_env_steps": int(environment_steps),
        "controller_tick_end": int(controller.step_count),
        "ppo_seed": int(ppo_seed),
        "network_mode": ofox_network_mode(),
        "model_alias": context_builder.spec.experiment_alias,
        "exact_model_id": context_builder.spec.exact_model_id,
        "primary_model_selected_is_false": not bool(context_builder.registry.primary_model_selected),
        "registry_sha256": context_builder.registry_sha256,
        "world_model_sha256": _file_sha256(world_model_path),
        "reward_model_sha256": _file_sha256(reward_model_path),
        "cache_format_version": int(CACHE_FORMAT_VERSION),
        "agent_aware_cache_identity": True,
        "transition_count": int(transition_count),
        "ppo_step_count": int(len(ppo_buffer.steps)),
        "per_agent_transitions": {
            agent: int(per_agent_transitions.get(agent, 0)) for agent in BLUE_AGENTS
        },
        "all_agents_done": len(done_agents) == len(BLUE_AGENTS),
        "requested_action_counts": {
            str(index): int(requested_counts.get(index, 0)) for index in range(4)
        },
        "executed_action_family_counts": dict(sorted(executed_counts.items())),
        "fallback_count": int(fallback_count),
        "plan0_match_count": int(plan0_match_count),
        "alignment_count": int(alignment_count),
        "response_reward_total": response_reward_total,
        "official_reward_total": official_reward_total,
        "candidate_context_builds": int(context_builder.generated_contexts),
        "cache_hits": int(context_builder.cache_hits),
        "cache_misses_before_update": misses_before_update,
        "cache_misses_after_update": misses_after_update,
        "live_transactions_before_update": transactions_before_update,
        "live_transactions_after_update": transactions_after_update,
        "estimated_cost_usd_this_run": float(context_builder.cost_this_run),
        "live_latency_ms_this_run": float(context_builder.latency_ms_this_run),
        "training_batch_size": int(batch.size),
        "update_metrics": asdict(metrics),
        "policy_parameters_changed": bool(policy_changed),
        "completed_decisions": completed_audits,
    }
    report["pass"] = _report_pass(report)

    out = resolve_project_path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate B4 one-model tiny end-to-end CC4 PPO pipeline smoke")
    parser.add_argument("--model-alias", default=DEFAULT_MODEL_ALIAS)
    parser.add_argument("--seed", type=int, default=DEFAULT_TRAIN_SEED)
    parser.add_argument("--steps", type=int, default=DEFAULT_SCENARIO_STEPS)
    parser.add_argument("--ppo-seed", type=int, default=DEFAULT_PPO_SEED)
    parser.add_argument("--registry", default=DEFAULT_REGISTRY)
    parser.add_argument("--world-model", default=DEFAULT_WORLD_MODEL)
    parser.add_argument("--reward-model", default=DEFAULT_REWARD_MODEL)
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    report = run_tiny_smoke(
        model_alias=args.model_alias,
        seed=args.seed,
        scenario_steps=args.steps,
        ppo_seed=args.ppo_seed,
        registry_path=args.registry,
        world_model_path=args.world_model,
        reward_model_path=args.reward_model,
        cache_root=args.cache_root,
        out_path=args.out,
        device=args.device,
    )

    print("=" * 80)
    print("[GATE B4 TINY PIPELINE SMOKE]")
    print("development_only:", report["development_only"])
    print("model:", report["model_alias"], report["exact_model_id"])
    print("seed/steps:", report["seed"], report["scenario_steps"])
    print("cache_format/agent_aware:", report["cache_format_version"], report["agent_aware_cache_identity"])
    print("transitions:", report["transition_count"])
    print("per_agent:", report["per_agent_transitions"])
    print("requested_actions:", report["requested_action_counts"])
    print("executed_families:", report["executed_action_family_counts"])
    print("fallback_count:", report["fallback_count"])
    print("plan0_match:", f"{report['plan0_match_count']}/{report['transition_count']}")
    print("alignment:", f"{report['alignment_count']}/{report['transition_count']}")
    print("cache_hits:", report["cache_hits"])
    print("cache_misses/live_calls:", report["cache_misses_before_update"], report["live_transactions_before_update"])
    print("cost_this_run:", report["estimated_cost_usd_this_run"])
    print("response_reward_total:", report["response_reward_total"])
    print("official_reward_total:", report["official_reward_total"])
    print("optimizer_steps:", report["update_metrics"]["optimizer_steps"])
    print("policy_parameters_changed:", report["policy_parameters_changed"])
    print("primary_model_selected:", not report["primary_model_selected_is_false"])
    print("pass:", report["pass"])
    print("[OK] report:", resolve_project_path(args.out))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
