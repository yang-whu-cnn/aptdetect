import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from formal_experiments.data_collection.decision_replay import DecisionEpochTransition
from formal_experiments.evaluation.run_b4_tiny_pipeline_smoke import (
    DEFAULT_MODEL_ALIAS,
    DEFAULT_TRAIN_SEED,
    CandidateContextBuilder,
    DecisionAudit,
    _context_matches_state,
    _report_pass,
    assert_runtime_resolution,
    selected_plan_first_action,
    validate_completed_alignment,
)
from formal_experiments.ours.ppo_training import RolloutStep
from formal_experiments.ours.prior_cache import CACHE_FORMAT_VERSION


class _FakeEvaluator:
    def evaluate(self, state, plans):
        del state
        plans = np.asarray(plans, dtype=np.int64)
        base = -plans.sum(axis=1).astype(np.float32)
        members = np.stack([base + shift for shift in (-0.2, -0.1, 0.0, 0.1, 0.2)], axis=1)
        return SimpleNamespace(
            expected_return=torch.tensor(members.mean(axis=1), dtype=torch.float32),
            member_returns=torch.tensor(members, dtype=torch.float32),
        )


class _FakeUsage:
    prompt_tokens = 10
    completion_tokens = 10
    total_tokens = 20


class _FakeResponse:
    def __init__(self, text):
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=text))]
        self.usage = _FakeUsage()
        self.model = "fake"


class _FakeCompletions:
    def __init__(self, owner):
        self.owner = owner

    def create(self, **kwargs):
        del kwargs
        self.owner.calls += 1
        candidates = []
        plans = [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [2, 0, 0, 0],
            [3, 0, 0, 0],
            [1, 2, 0, 0],
            [3, 2, 1, 0],
        ]
        for index, plan in enumerate(plans):
            candidates.append(
                {
                    "actions": [
                        ("no_op", "analyse", "remove", "restore")[action]
                        for action in plan
                    ],
                    "prior_score": float(0.9 - index * 0.1),
                    "reason": f"candidate-{index}",
                }
            )
        return _FakeResponse(json.dumps({"candidates": candidates}))


class _FakeClient:
    def __init__(self):
        self.calls = 0
        self.chat = SimpleNamespace(completions=_FakeCompletions(self))


def _transition(*, reward=-3.0, dt=2, done=False, requested=1):
    return DecisionEpochTransition(
        episode_seed=1000,
        agent_name="blue_agent_0",
        decision_index=0,
        global_tick_start=0,
        global_tick_end=dt,
        decision_dt=dt,
        state=np.zeros(27, dtype=np.float32),
        next_state=np.ones(27, dtype=np.float32),
        requested_action_id=requested,
        requested_action_name=("no_op", "analyse", "remove", "restore")[requested],
        requested_cyborg_family=("Sleep", "Analyse", "Remove", "Restore")[requested],
        requested_duration_ticks=(1, 2, 3, 5)[requested],
        executed_index=0,
        executed_label="dummy",
        executed_action_family="Analyse" if requested == 1 else "Sleep",
        executed_duration_ticks=dt,
        target_host=None,
        fallback=False,
        fallback_reason=None,
        action_completed=True,
        completed_action_success=True,
        incident_event_ids=(),
        incident_host_ids=(),
        incident_active_ticks=3,
        incident_host_lwf_count=0,
        incident_host_lwf_raw_penalty=0.0,
        response_reward=reward,
        official_reward=0.0,
        done=done,
    )


def _ppo_step(*, reward=-3.0, dt=2, done=False, candidate=2):
    return RolloutStep(
        episode_seed=1000,
        agent_name="blue_agent_0",
        decision_index=0,
        candidate_features=torch.zeros(6, 46),
        candidate_index=candidate,
        behavior_log_prob=-1.0,
        critic_value=0.5,
        critic_next_value=0.0 if done else 0.25,
        real_response_reward=reward,
        decision_dt=dt,
        done=done,
    )


def _audit(*, requested=1, candidate=2):
    return DecisionAudit(
        episode_seed=1000,
        agent_name="blue_agent_0",
        decision_index=0,
        selected_candidate_index=candidate,
        selected_plan=(requested, 0, 0, 0),
        requested_action_id=requested,
        cache_hit=False,
        cache_key="a" * 64,
        state_sha256="b" * 64,
    )


def _valid_report():
    return {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "agent_aware_cache_identity": True,
        "transition_count": 10,
        "ppo_step_count": 10,
        "per_agent_transitions": {f"blue_agent_{i}": 2 for i in range(5)},
        "plan0_match_count": 10,
        "alignment_count": 10,
        "cache_misses_before_update": 4,
        "cache_misses_after_update": 4,
        "live_transactions_before_update": 4,
        "live_transactions_after_update": 4,
        "policy_parameters_changed": True,
        "all_agents_done": True,
        "primary_model_selected_is_false": True,
        "update_metrics": {
            "policy_loss_mean": 0.1,
            "value_loss_mean": 1.0,
            "entropy_mean": 1.5,
            "approx_kl_mean": 0.01,
            "clip_fraction_mean": 0.0,
            "grad_norm_mean": 0.5,
            "grad_norm_max": 1.0,
        },
    }


class TestGateB4TinyPipelineSmoke(unittest.TestCase):
    def test_default_smoke_is_train_only_low_cost_engineering_model(self):
        self.assertEqual(DEFAULT_TRAIN_SEED, 1000)
        self.assertEqual(DEFAULT_MODEL_ALIAS, "llm_l_gemini35_flash_lite")

    def test_selected_plan_first_action_contract(self):
        plans = np.asarray(
            [
                [0, 0, 0, 0],
                [1, 2, 3, 0],
                [2, 1, 0, 0],
                [3, 3, 0, 0],
                [1, 1, 1, 1],
                [2, 2, 2, 2],
            ],
            dtype=np.int64,
        )
        self.assertEqual(selected_plan_first_action(plans, 3), 3)
        with self.assertRaises(ValueError):
            selected_plan_first_action(plans, 6)

    def test_runtime_resolution_contract_for_available_and_fallback(self):
        available = {"Analyse": True, "Remove": True, "Restore": True}
        assert_runtime_resolution(
            action_id=1,
            family_availability=available,
            resolution=SimpleNamespace(fallback=False, executed_action_family="Analyse"),
        )
        unavailable = {"Analyse": False, "Remove": False, "Restore": False}
        assert_runtime_resolution(
            action_id=3,
            family_availability=unavailable,
            resolution=SimpleNamespace(fallback=True, executed_action_family="Sleep"),
        )
        with self.assertRaises(RuntimeError):
            assert_runtime_resolution(
                action_id=2,
                family_availability=unavailable,
                resolution=SimpleNamespace(fallback=False, executed_action_family="Remove"),
            )

    def test_completed_alignment_accepts_exact_replay_ppo_match(self):
        validate_completed_alignment(
            transition=_transition(),
            ppo_step=_ppo_step(),
            audit=_audit(),
        )

    def test_completed_alignment_rejects_reward_dt_and_plan_mismatch(self):
        with self.assertRaises(RuntimeError):
            validate_completed_alignment(
                transition=_transition(),
                ppo_step=_ppo_step(reward=-4.0),
                audit=_audit(),
            )
        with self.assertRaises(RuntimeError):
            validate_completed_alignment(
                transition=_transition(),
                ppo_step=_ppo_step(dt=1),
                audit=_audit(),
            )
        with self.assertRaises(RuntimeError):
            validate_completed_alignment(
                transition=_transition(requested=1),
                ppo_step=_ppo_step(),
                audit=_audit(requested=2),
            )

    def test_context_prefetch_exact_state_hash_guard(self):
        fake = SimpleNamespace(state_sha256=None)
        state = np.arange(27, dtype=np.float32)
        from formal_experiments.ours.prior_cache import exact_state_sha256

        fake.state_sha256 = exact_state_sha256(state)
        self.assertTrue(_context_matches_state(fake, state.copy()))
        changed = state.copy()
        changed[0] += 1.0
        self.assertFalse(_context_matches_state(fake, changed))

    def test_candidate_context_builder_uses_agent_aware_train_cache_and_hits_second_time(self):
        fake_client = _FakeClient()
        with tempfile.TemporaryDirectory() as tmp:
            builder = CandidateContextBuilder(
                model_alias=DEFAULT_MODEL_ALIAS,
                evaluator=_FakeEvaluator(),
                cache_root=tmp,
                client_factory=lambda: fake_client,
            )
            state = np.linspace(0.0, 1.0, 27, dtype=np.float32)
            first = builder.build(state=state, agent_name="blue_agent_0")
            second = builder.build(state=state, agent_name="blue_agent_0")

            self.assertFalse(first.cache_hit)
            self.assertTrue(second.cache_hit)
            self.assertEqual(fake_client.calls, 1)
            self.assertEqual(builder.cache_misses, 1)
            self.assertEqual(builder.live_transactions, 1)
            self.assertEqual(builder.cache_hits, 1)
            self.assertEqual(tuple(first.candidate_features.shape), (6, 46))
            self.assertEqual(first.cache_key, second.cache_key)
            paths = list(Path(tmp).rglob("*.json"))
            self.assertEqual(len(paths), 1)
            self.assertIn("train", paths[0].parts)
            self.assertIn(DEFAULT_MODEL_ALIAS, paths[0].parts)
            self.assertIn("blue_agent_0", paths[0].parts)

    def test_same_d27_different_agent_forces_second_provider_transaction(self):
        fake_client = _FakeClient()
        with tempfile.TemporaryDirectory() as tmp:
            builder = CandidateContextBuilder(
                model_alias=DEFAULT_MODEL_ALIAS,
                evaluator=_FakeEvaluator(),
                cache_root=tmp,
                client_factory=lambda: fake_client,
            )
            state = np.linspace(0.0, 1.0, 27, dtype=np.float32)
            first = builder.build(state=state, agent_name="blue_agent_0")
            second = builder.build(state=state, agent_name="blue_agent_1")
            self.assertNotEqual(first.cache_key, second.cache_key)
            self.assertEqual(fake_client.calls, 2)
            self.assertEqual(builder.cache_misses, 2)
            self.assertEqual(builder.live_transactions, 2)

    def test_report_gate_requires_agent_cache_alignment_no_optimizer_api_and_all_agents(self):
        report = _valid_report()
        self.assertTrue(_report_pass(report))

        broken = dict(report)
        broken["live_transactions_after_update"] = 5
        self.assertFalse(_report_pass(broken))

        broken = dict(report)
        broken["agent_aware_cache_identity"] = False
        self.assertFalse(_report_pass(broken))

        broken = dict(report)
        broken["cache_format_version"] = 1
        self.assertFalse(_report_pass(broken))

    def test_report_gate_rejects_missing_agent_or_unchanged_policy(self):
        base = _valid_report()
        base["transition_count"] = 5
        base["ppo_step_count"] = 5
        base["per_agent_transitions"] = {f"blue_agent_{i}": 1 for i in range(5)}
        base["plan0_match_count"] = 5
        base["alignment_count"] = 5
        missing = dict(base)
        missing["per_agent_transitions"] = {"blue_agent_0": 5}
        self.assertFalse(_report_pass(missing))
        unchanged = dict(base)
        unchanged["policy_parameters_changed"] = False
        self.assertFalse(_report_pass(unchanged))


if __name__ == "__main__":
    unittest.main()
