"""Real CC4 1x20 train+eval smoke for RL-Only and WM-RL."""

from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path
import torch

from baselines.rsmbrl_cc4.artifact_preflight import (
    DEFAULT_REWARD_MODEL, DEFAULT_WORLD_MODEL, resolve, run_preflight,
)
from formal_experiments.ours.table2_variants import Table2DecisionRuntime, Table2Variant
from formal_experiments.ours.reward_ablation import RewardMode, tick_reward
from formal_experiments.ours.variant_training import VariantPPOBuffer, VariantTransition, ppo_update
from formal_experiments.training.bootstrap_world_model import BootstrapProbabilisticWorldModel
from formal_experiments.training.response_reward_predictor import ResponseRewardPredictor
from shared.cyborg_action_adapter import CybORGActionAdapter
from shared.formal_state import BLUE_AGENTS, FormalStateEncoder, ObservableHostEvidenceTracker
from shared.d27_projection import D27ProjectionContext
from shared.rollout_evaluator import SharedRolloutConfig, SharedRolloutEvaluator


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _ablation_reward_artifact(mode: RewardMode, *, device: str):
    slug = mode.value.lower().replace("-", "_")
    directory = Path("outputs/formal_v3/table3_reward_models") / slug
    manifest_path = directory / "frozen_manifest.json"
    if not manifest_path.is_file(): raise RuntimeError(f"BLOCKED: missing frozen {mode.value} manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint = Path(manifest["checkpoint"])
    if not manifest.get("frozen") or not manifest.get("quality_gate", {}).get("pass"):
        raise RuntimeError(f"BLOCKED: {mode.value} reward predictor did not pass its frozen gate")
    if not checkpoint.is_file() or sha(checkpoint) != manifest.get("checkpoint_sha256"):
        raise RuntimeError(f"BLOCKED: {mode.value} reward artifact SHA mismatch")
    predictor = ResponseRewardPredictor.load_checkpoint(checkpoint, device=device)
    return predictor, manifest


def build_runtime(variant: Table2Variant, *, seed: int, device: str,
                  reward_mode: RewardMode = RewardMode.FULL_REWARD,
                  offline_prior_artifact: Path | None = None):
    if variant == Table2Variant.RL_ONLY:
        return Table2DecisionRuntime(variant=variant, init_seed=seed, split="train",
                                     device=device), {}
    prior = None; prior_metadata = {}
    if variant in (Table2Variant.LLM_RL, Table2Variant.LWM_RL):
        if offline_prior_artifact is None:
            raise RuntimeError("BLOCKED: LLM variant requires a frozen K6/H4 offline_prior_artifact")
        try:
            from baselines.priorrl_cc4.prototype_retrieval import FrozenPrototypePriorAdapter
            prior = FrozenPrototypePriorAdapter.load_artifact(offline_prior_artifact)
            prior_metadata = {"offline_prior_artifact": str(offline_prior_artifact),
                              "offline_prior_sha256": sha(offline_prior_artifact)}
        except Exception as exc:
            raise RuntimeError(f"BLOCKED: invalid frozen K6/H4 offline prior artifact: {exc}") from exc
    if variant == Table2Variant.LLM_RL:
        return Table2DecisionRuntime(variant=variant, init_seed=seed, offline_prior=prior,
                                     split="train", device=device), prior_metadata
    gate = run_preflight(device=device)
    if not gate["eligible"]: raise RuntimeError("frozen WM/Full-Reward preflight failed: " + "; ".join(gate["errors"]))
    world = BootstrapProbabilisticWorldModel.load_checkpoint(resolve(DEFAULT_WORLD_MODEL), device=device)
    if reward_mode == RewardMode.FULL_REWARD:
        reward = ResponseRewardPredictor.load_checkpoint(resolve(DEFAULT_REWARD_MODEL), device=device)
        reward_sha = sha(resolve(DEFAULT_REWARD_MODEL))
    else:
        reward, reward_manifest = _ablation_reward_artifact(reward_mode, device=device)
        reward_sha = reward_manifest["checkpoint_sha256"]
    frozen_networks = [*world.models, reward.model]
    for model in frozen_networks:
        model.eval()
        for parameter in model.parameters(): parameter.requires_grad_(False)
    evaluator = SharedRolloutEvaluator(world_model=world, reward_predictor=reward,
                                       config=SharedRolloutConfig(horizon=4, gamma_tick=.99))
    evaluator.frozen = True
    runtime = Table2DecisionRuntime(variant=variant, init_seed=seed, offline_prior=prior,
                                    frozen_wm_evaluator=evaluator, split="train", device=device)
    return runtime, {**prior_metadata, "world_model_sha256": sha(resolve(DEFAULT_WORLD_MODEL)),
                     "reward_model_sha256": reward_sha, "reward_mode": reward_mode.value,
                     "ensemble_members": 5, "horizon": 4}


def run_episode(runtime, *, episode_seed: int, steps: int, split: str, train: bool,
                reward_mode: RewardMode = RewardMode.FULL_REWARD,
                return_formal_record: bool = False):
    # Keep CC4 imports local so preflight and contract tests remain usable on hosts
    # that do not have the dedicated environment installed.
    from formal_experiments.data_collection.collect_cc4_formal_replay import (
        encode_decision_state, green_local_work_failures, make_env,
        observation_success_bool, red_presence_for_hosts,
    )
    from formal_experiments.data_collection.incident_response import IncidentResponseBookkeeper

    env, observations, _ = make_env(seed=episode_seed, steps=steps + 1, pad_spaces=False)
    controller = env.env.environment_controller; adapter = CybORGActionAdapter(); encoder = FormalStateEncoder()
    trackers = {}; books = {}; active = {agent: None for agent in BLUE_AGENTS}; done = set()
    buffer = VariantPPOBuffer(split=split); decisions = []
    tick_team_rewards = []; operation_failure_events = []; recovery_actions = []
    for agent in BLUE_AGENTS:
        tracker = ObservableHostEvidenceTracker(agent); tracker.reset(observations[agent]); trackers[agent] = tracker
        book = IncidentResponseBookkeeper(episode_seed=episode_seed, agent_name=agent)
        book.reset(initial_red_presence_by_host=red_presence_for_hosts(controller, tracker.inventory), global_tick=0)
        books[agent] = book
    for _ in range(steps):
        tick = int(controller.step_count); joint = {}
        for agent in BLUE_AGENTS:
            if agent in done or active[agent] is not None: continue
            state, scores, availability = encode_decision_state(
                env=env, encoder=encoder, tracker=trackers[agent], observation=observations[agent],
                agent_name=agent, global_tick=tick, episode_steps=steps,
            )
            mask = [True, availability["Analyse"], availability["Remove"], availability["Restore"]]
            context = D27ProjectionContext.from_root(state, root_tick=tick, episode_steps=steps)
            decision = runtime.decide(state, agent_name=agent, action_mask=mask,
                                      deterministic=not train, projection_context=context)
            resolution = adapter.resolve(env=env, agent_name=agent,
                                         action_id=decision["requested_action"], observable_host_scores=scores)
            action = list(env.actions(agent))[resolution.executed_index]
            active[agent] = {"ready": tick + int(action.duration), "family": resolution.executed_action_family,
                             "target": resolution.target_host, "decision": decision, "reward": 0.0,
                             "start": tick, "fallback": resolution.fallback,
                             "active_incident_before": bool(books[agent].active_events)}
            joint[agent] = resolution.executed_index
            decision_audit = {key: decision[key] for key in (
                "selected_plan", "candidate_index", "candidate_plans_before_dedup",
                "candidate_plans_after_dedup", "candidate_action_mask",
                "candidate_selection_probabilities", "candidate_probability_rule",
            ) if key in decision}
            decisions.append({"tick": tick, "agent": agent, "requested": decision["requested_action"],
                              "executed": resolution.executed_action_family, "mask": mask,
                              "fallback": resolution.fallback, **decision_audit})
        next_obs, _rewards, terminated, truncated, _ = env.step(actions=joint)
        tick_end = int(controller.step_count); all_lwf = green_local_work_failures(controller)
        team_tick_reward = 0.0
        for agent in BLUE_AGENTS:
            if agent in done: continue
            meta = active[agent]; inventory = set(trackers[agent].inventory)
            accounting = books[agent].record_tick(
                global_tick_end=tick_end,
                red_presence_after=red_presence_for_hosts(controller, trackers[agent].inventory),
                local_work_failures=[item for item in all_lwf if item.hostname in inventory],
            )
            reward_at_tick = tick_reward(accounting, reward_mode)
            meta["reward"] += reward_at_tick; team_tick_reward += reward_at_tick
            for failure in all_lwf:
                if failure.hostname in inventory:
                    operation_failure_events.append({"agent_name": agent, "host": failure.hostname,
                                                     "tick": tick_end, "raw_penalty": float(failure.raw_lwf_penalty)})
            completed = tick_end == meta["ready"]
            finished = bool(terminated.get(agent, False) or truncated.get(agent, False))
            success = observation_success_bool(next_obs[agent]) if completed else None
            trackers[agent].update(observation=next_obs[agent], global_tick=tick_end,
                                   completed_action_family=meta["family"] if completed else None,
                                   completed_target_host=meta["target"] if completed else None,
                                   completed_action_success=success if completed else None)
            observations[agent] = next_obs[agent]
            if completed:
                item = meta["decision"]
                buffer.append(VariantTransition(
                    agent_name=agent, policy_input=item["policy_input"], policy_mask=item["policy_mask"],
                    policy_action=item["policy_action"],
                    old_log_prob=item["log_prob"], value=item["value"], next_value=0.0,
                    reward=meta["reward"], duration=tick_end-meta["start"], done=finished, split=split,
                ))
                recovery_actions.append({"agent_name": agent, "tick_start": meta["start"],
                                         "tick_end": tick_end, "executed_action": meta["family"],
                                         "started": True, "completed": True,
                                         "fallback": bool(meta["fallback"]),
                                         "active_incident_before": bool(meta["active_incident_before"])})
                active[agent] = None
            if finished: done.add(agent)
        tick_team_rewards.append(float(team_tick_reward))
    if not return_formal_record: return buffer, decisions
    end_tick = int(controller.step_count)
    incidents = []
    for book in books.values():
        for event in book.all_events:
            value = event.to_jsonable()
            incidents.append({"incident_event_id": value["incident_event_id"],
                              "agent_name": value["agent_name"], "host": value["host_id"],
                              "t_compromise": value["t_compromise"], "t_recovered": value["t_normal"]})
    record = {"schema_version": 1, "protocol_version": "cc4_v3_20260917",
              "episode_seed": int(episode_seed), "tick_count": end_tick, "episode_end_tick": end_tick,
              "tick_team_rewards": tick_team_rewards, "operation_failure_events": operation_failure_events,
              "recovery_actions": recovery_actions, "incidents": incidents}
    return buffer, decisions, record


def policy_hash(policy):
    digest = hashlib.sha256()
    for tensor in policy.state_dict().values(): digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def run_smoke(variant: Table2Variant, *, device="cpu", out: Path | None = None,
              reward_mode: RewardMode = RewardMode.FULL_REWARD,
              offline_prior_artifact: Path | None = None):
    runtime, artifacts = build_runtime(variant, seed=51001, device=device, reward_mode=reward_mode,
                                       offline_prior_artifact=offline_prior_artifact)
    train_buffer, train_decisions = run_episode(runtime, episode_seed=1000, steps=20, split="train", train=True, reward_mode=reward_mode)
    update = ppo_update(runtime, train_buffer)
    before = policy_hash(runtime.policy)
    runtime.split = "validation"
    eval_buffer, eval_decisions = run_episode(runtime, episode_seed=2000, steps=20, split="validation", train=False, reward_mode=reward_mode)
    after = policy_hash(runtime.policy)
    if before != after: raise RuntimeError("evaluation mutated policy")
    report = {"development_only": True, "formal_result_eligible": False, "real_cc4": True,
              "variant": variant.value, "reward_mode": reward_mode.value,
              "reward_semantics_shared_by_real_and_imagined_channels": True,
              "train_seed": 1000, "validation_seed": 2000,
              "test_seed_used": False, "online_llm_calls": 0, "train_transitions": len(train_buffer.transitions),
              "eval_transitions": len(eval_buffer.transitions), "ppo_update": update,
              "evaluation_policy_frozen": True, "artifacts": artifacts,
              "module_audit": {"prior_calls": runtime.audit.prior_calls,
                               "wm_calls": runtime.audit.wm_calls,
                               "policy_calls": runtime.audit.policy_calls,
                               "online_llm_calls": runtime.audit.online_llm_calls,
                               "cache_misses": runtime.audit.cache_misses,
                               "selected_first_action_counts": {
                                   str(action): runtime.audit.selected_first_actions.count(action)
                                   for action in range(4)}},
              "performance_claim": False,
              "note": "Development gate only; PPO loss uses unscaled real response rewards and is not a performance result.",
              "train_decisions": len(train_decisions),
              "eval_decisions": len(eval_decisions)}
    if out: out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--variant", choices=[x.value for x in Table2Variant], required=True)
    parser.add_argument("--reward-mode", choices=[x.value for x in RewardMode], default=RewardMode.FULL_REWARD.value)
    parser.add_argument("--offline-prior-artifact", type=Path)
    parser.add_argument("--device", default="cpu"); parser.add_argument("--out", type=Path, required=True); args=parser.parse_args()
    print(json.dumps(run_smoke(Table2Variant(args.variant), device=args.device, out=args.out,
                               reward_mode=RewardMode(args.reward_mode),
                               offline_prior_artifact=args.offline_prior_artifact), indent=2))


if __name__ == "__main__": main()
