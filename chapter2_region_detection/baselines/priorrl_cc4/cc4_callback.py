"""Real CC4 episode callback for development-only PriorRL runs."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from formal_experiments.data_collection.collect_cc4_formal_replay import (
    encode_decision_state, green_local_work_failures, make_env, observation_success_bool,
    planner_visible_target_availability, red_presence_for_hosts,
)
from shared.action_contract import ACTION_CONTRACTS
from shared.cyborg_action_adapter import CybORGActionAdapter
from shared.formal_state import BLUE_AGENTS, FormalStateEncoder, ObservableHostEvidenceTracker
from .development_pilot import (
    DevelopmentCollector, DevelopmentPilotConfig, FormalRewardChannel,
    make_formal_reward_channel,
)
from .policy import PriorRLActorCritic
from .training import A4PPOConfig, A4PPOTrainer, A4RolloutBuffer, checkpoint_sha256

FAMILY_DURATION = {str(x.cyborg_action): int(x.duration_ticks) for x in ACTION_CONTRACTS}


class UniformFrozenRetriever:
    """Synthetic smoke-only retriever; no cache or API access."""
    artifact_sha256 = hashlib.sha256(b"priorrl_uniform_fake_v1").hexdigest()
    def lookup(self, state, *, agent_name):
        if agent_name not in BLUE_AGENTS or np.asarray(state).shape != (27,):
            raise RuntimeError("invalid synthetic retrieval request")
        return torch.full((4,), .25)


def run_cc4_episode(*, episode_seed: int, policy_seed: int, policy: PriorRLActorCritic,
                    retriever, episode_ticks: int, training: bool,
                    rollout_threshold: int = 32, ppo_config: A4PPOConfig | None = None,
                    trainer: A4PPOTrainer | None = None,
                    buffer: A4RolloutBuffer | None = None,
                    flush_final: bool = True, protocol: str = "development") -> dict:
    allowed = {
        "development": ((range(1000, 1032) if training else range(3200, 3210)), (61001, 61002), 100),
        "alpha_validation": ((range(1000, 1032) if training else range(2000, 2008)), (50999,), 500),
        "formal_train": (range(1000, 1032), (51001, 51002, 51003, 51004, 51005), 500),
    }
    if protocol not in allowed:
        raise ValueError("unknown PriorRL collection protocol")
    seeds, policy_seeds, required_ticks = allowed[protocol]
    if episode_seed not in seeds:
        raise ValueError(f"episode seed violates {protocol} split")
    if policy_seed not in policy_seeds:
        raise ValueError(f"policy seed violates {protocol} split")
    if protocol != "development" and episode_ticks != required_ticks:
        raise ValueError(f"{protocol} requires {required_ticks} ticks")
    if protocol == "formal_train" and not training:
        raise ValueError("formal_train protocol cannot evaluate test episodes")
    if rollout_threshold < 1:
        raise ValueError("rollout_threshold must be positive")
    if not training and (trainer is not None or buffer is not None):
        raise ValueError("evaluation must not receive optimizer state or a rollout buffer")
    env, observations, _ = make_env(seed=episode_seed, steps=episode_ticks + 1, pad_spaces=False)
    controller = env.env.environment_controller
    adapter, encoder = CybORGActionAdapter(), FormalStateEncoder()
    buffer = buffer if buffer is not None else A4RolloutBuffer()
    collector = DevelopmentCollector(policy, retriever, buffer)
    if training:
        trainer = trainer if trainer is not None else A4PPOTrainer(policy, ppo_config)
        if trainer.policy is not policy:
            raise ValueError("trainer and episode must share the same policy instance")
    trackers = {}; channels = {}; active = {agent: None for agent in BLUE_AGENTS}
    for agent in BLUE_AGENTS:
        tracker = ObservableHostEvidenceTracker(agent); tracker.reset(observations[agent])
        trackers[agent] = tracker
        channels[agent] = make_formal_reward_channel(
            episode_seed=episode_seed, agent_name=agent,
            initial_red_presence_by_host=red_presence_for_hosts(controller, tracker.inventory))
    records = []; updates = 0; episode_transitions = 0
    for _ in range(episode_ticks):
        tick = int(controller.step_count); joint = {}
        for agent in BLUE_AGENTS:
            if active[agent] is not None: continue
            state, host_scores, _ = encode_decision_state(
                env=env, encoder=encoder, tracker=trackers[agent], observation=observations[agent],
                agent_name=agent, global_tick=tick, episode_steps=episode_ticks + 1)
            _, family = planner_visible_target_availability(
                env=env, agent_name=agent, observable_host_scores=host_scores)
            mask = (True, bool(family["Analyse"]), bool(family["Remove"]), bool(family["Restore"]))
            decision = collector.decide(state, agent_name=agent, action_mask=mask,
                                        deterministic=not training)
            resolution = adapter.resolve(env=env, agent_name=agent, action_id=decision["action"],
                                         observable_host_scores=host_scores)
            action = list(env.actions(agent))[int(resolution.executed_index)]
            duration = int(action.duration); action_family = type(action).__name__
            if duration != FAMILY_DURATION[action_family]:
                raise RuntimeError("CC4 action duration contract mismatch")
            if resolution.fallback:
                raise RuntimeError("masked PriorRL requested an unavailable action")
            active[agent] = {"start": tick, "ready_at": tick + duration, "family": action_family,
                             "target": resolution.target_host, "decision": decision,
                             "requested": decision["action"], "executed": int(resolution.executed_index)}
            joint[agent] = int(resolution.executed_index)
        new_obs, _official_rewards, terminated, truncated, _ = env.step(actions=joint)
        tick_end = int(controller.step_count)
        if tick_end != tick + 1: raise RuntimeError("CC4 did not advance exactly one tick")
        all_lwf = green_local_work_failures(controller)
        for agent in BLUE_AGENTS:
            tracker = trackers[agent]; inventory = set(tracker.inventory)
            channels[agent].record_hidden_tick(
                global_tick_end=tick_end,
                red_presence_after=red_presence_for_hosts(controller, tracker.inventory),
                local_work_failures=[x for x in all_lwf if x.hostname in inventory])
            meta = active[agent]
            # CybORG has one spare step so that this runner owns the fixed
            # development horizon. Close in-flight long actions as truncated
            # terminal transitions at that exact boundary.
            done = bool(terminated.get(agent, False) or truncated.get(agent, False)
                        or tick_end == episode_ticks)
            completed = tick_end == meta["ready_at"]
            if tick_end > meta["ready_at"]: raise RuntimeError("duration scheduler missed ready_at")
            tracker.update(observation=new_obs[agent], global_tick=tick_end,
                completed_action_family=meta["family"] if completed else None,
                completed_target_host=meta["target"] if completed else None,
                completed_action_success=observation_success_bool(new_obs[agent]) if completed else None)
            observations[agent] = new_obs[agent]
            if completed or done:
                next_state, _, _ = encode_decision_state(
                    env=env, encoder=encoder, tracker=tracker, observation=new_obs[agent],
                    agent_name=agent, global_tick=tick_end, episode_steps=episode_ticks + 1)
                collector.complete(meta["decision"], next_observable_state=next_state,
                    reward_channel=channels[agent], done=done, decision_dt=tick_end-meta["start"],
                    trajectory_id=f"{agent}:{episode_seed}")
                transition = buffer._items[-1]
                records.append({"agent_name": agent, "tick_start": meta["start"],
                    "tick_end": tick_end, "decision_dt": transition.decision_dt,
                    "response_reward": transition.reward, "requested_action_id": meta["requested"],
                    "executed_index": meta["executed"], "executed_action_family": meta["family"],
                    "fallback": False, "action_mask": transition.action_mask.tolist()})
                episode_transitions += 1
                active[agent] = None
        if training and len(buffer) >= rollout_threshold:
            # Each threshold batch is consumed exactly once. Clearing in place
            # lets an externally owned buffer persist across episode calls.
            trainer.optimize(buffer); updates += 1; buffer.clear()
    if any(meta is not None for meta in active.values()):
        raise RuntimeError("episode horizon left an unclosed PriorRL decision")
    if training and flush_final and len(buffer):
        trainer.optimize(buffer); updates += 1; buffer.clear()
    return {"episode_seed": episode_seed, "policy_seed": policy_seed, "training": training,
            "ticks": int(controller.step_count), "decisions": records, "ppo_updates": updates,
            "transition_count": episode_transitions,
            "pending_unoptimized_transitions": len(buffer),
            "checkpoint_sha256": checkpoint_sha256(policy)}


def run_real_wiring_smoke(*, device=None, output=None) -> dict:
    run_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(61001); policy = PriorRLActorCritic().to(run_device)
    retriever = UniformFrozenRetriever()
    train = run_cc4_episode(episode_seed=1000, policy_seed=61001, policy=policy,
                            retriever=retriever, episode_ticks=20, training=True,
                            rollout_threshold=8, ppo_config=A4PPOConfig(epochs=1, minibatch_size=8))
    evaluation = run_cc4_episode(episode_seed=3200, policy_seed=61001, policy=policy,
                                 retriever=retriever, episode_ticks=20, training=False)
    report = {"status": "PASS", "formal_result_eligible": False,
              "smoke": "real_cc4_1x20_train_plus_1x20_eval", "device": str(run_device),
              "retriever": "synthetic_uniform_frozen", "retriever_sha256": retriever.artifact_sha256,
              "world_model_used": False, "online_llm_calls": 0, "normalizer_used": False,
              "train": train, "evaluation": evaluation}
    report["report_sha256"] = hashlib.sha256(json.dumps(report, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()
    if output:
        path = Path(output); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _compact_episode(record: dict) -> dict:
    decisions = record["decisions"]
    serialized = json.dumps(decisions, sort_keys=True, separators=(",", ":")).encode()
    return {
        "episode_seed": record["episode_seed"],
        "policy_seed": record["policy_seed"],
        "training": record["training"],
        "ticks": record["ticks"],
        "transition_count": record["transition_count"],
        "ppo_updates": record["ppo_updates"],
        "pending_unoptimized_transitions": record["pending_unoptimized_transitions"],
        "response_reward_sum": float(sum(x["response_reward"] for x in decisions)),
        "action_counts": dict(sorted(Counter(x["executed_action_family"] for x in decisions).items())),
        "duration_counts": {str(k): v for k, v in sorted(Counter(x["decision_dt"] for x in decisions).items())},
        "fallback_count": int(sum(bool(x["fallback"]) for x in decisions)),
        "last_transition_tick": max((x["tick_end"] for x in decisions), default=None),
        "decisions_sha256": hashlib.sha256(serialized).hexdigest(),
        "checkpoint_sha256": record["checkpoint_sha256"],
    }


def run_real_development_pilot(*, config: DevelopmentPilotConfig, retriever_factory,
                               device=None, output=None, rollout_threshold: int = 128,
                               ppo_config: A4PPOConfig | None = None) -> dict:
    """Run the approved real-CC4 development split with persistent PPO state.

    Each policy seed owns an independently initialized policy, optimizer, rollout
    buffer, and frozen retriever. Evaluation is deterministic and is checked not
    to mutate the trained checkpoint.
    """
    if not isinstance(config, DevelopmentPilotConfig):
        raise TypeError("config must be DevelopmentPilotConfig")
    run_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    report = {
        "schema": "priorrl_cc4_real_development_pilot_v1",
        "status": "PASS",
        "formal_result_eligible": False,
        "device": str(run_device),
        "training_episode_budget": config.training_episode_budget,
        "episode_ticks": config.episode_ticks,
        "policy_seeds": list(config.policy_seeds),
        "train_seeds": list(config.train_seeds()),
        "eval_seed_map": {str(k): list(v) for k, v in config.eval_seed_map().items()},
        "rollout_threshold": int(rollout_threshold),
        "world_model_used": False,
        "online_llm_calls": 0,
        "policies": {},
    }
    for policy_seed in config.policy_seeds:
        torch.manual_seed(policy_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(policy_seed)
        policy = PriorRLActorCritic().to(run_device)
        policy.train()
        retriever = retriever_factory(policy_seed)
        trainer = A4PPOTrainer(policy, ppo_config)
        buffer = A4RolloutBuffer()
        train_rows = []
        train_seeds = config.train_seeds()
        for index, episode_seed in enumerate(train_seeds):
            row = run_cc4_episode(
                episode_seed=episode_seed, policy_seed=policy_seed, policy=policy,
                retriever=retriever, episode_ticks=config.episode_ticks, training=True,
                rollout_threshold=rollout_threshold, trainer=trainer, buffer=buffer,
                flush_final=index == len(train_seeds) - 1,
            )
            train_rows.append(_compact_episode(row))
        if len(buffer):
            raise RuntimeError("final training rollout was not consumed")
        trained_sha = checkpoint_sha256(policy)
        policy.eval()
        eval_rows = []
        with torch.no_grad():
            for episode_seed in config.eval_seed_map()[policy_seed]:
                row = run_cc4_episode(
                    episode_seed=episode_seed, policy_seed=policy_seed, policy=policy,
                    retriever=retriever, episode_ticks=config.episode_ticks, training=False,
                    rollout_threshold=rollout_threshold,
                )
                eval_rows.append(_compact_episode(row))
                if checkpoint_sha256(policy) != trained_sha:
                    raise RuntimeError("evaluation mutated the PriorRL policy")
        report["policies"][str(policy_seed)] = {
            "trained_checkpoint_sha256": trained_sha,
            "train": train_rows,
            "evaluation": eval_rows,
        }
    report["report_sha256"] = hashlib.sha256(json.dumps(
        report, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if output:
        path = Path(output); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
