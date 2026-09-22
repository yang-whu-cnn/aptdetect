import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from formal_experiments.ours.llm_prior_v2 import PriorBatch
from formal_experiments.ours.lwm_runtime import LWMDecisionRuntime
from formal_experiments.ours.model_registry import load_model_registry
from formal_experiments.ours.ppo_core import CandidateActorCritic, PPOCoreConfig
from formal_experiments.ours.prior_cache import PriorCache, make_identity
from shared.formal_state import FORMAL_STATE_DIM


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "configs" / "llm_model_registry_v1.yaml"
REGISTRY_SHA = hashlib.sha256(REGISTRY_PATH.read_bytes()).hexdigest()


def sample_prior():
    plans = np.asarray(
        [
            [0, 1, 2, 3],
            [1, 0, 2, 3],
            [2, 1, 0, 3],
            [3, 1, 2, 0],
            [1, 1, 1, 1],
            [2, 2, 2, 2],
        ],
        dtype=np.int64,
    )
    scores = np.asarray([0.9, 0.8, 0.7, 0.6, 0.5, 0.4], dtype=np.float32)
    return PriorBatch(
        plans=plans,
        raw_prior_scores=scores,
        prior_preferences=(scores / scores.sum()).astype(np.float32),
        sources=("llm",) * 6,
    )


def metadata():
    return {
        "api_success": True,
        "fallback_count": 0,
        "prompt_sha256": "1" * 64,
        "response_sha256": "2" * 64,
        "latency_ms": 1.0,
        "retry_count": 0,
    }


class FakeEvaluator:
    def __init__(self):
        self.calls = 0

    def evaluate(self, state, plans, *, projection_context=None):
        self.calls += 1
        self.last_projection_context = projection_context
        self.last_state = np.asarray(state, dtype=np.float32).copy()
        self.last_plans = np.asarray(plans, dtype=np.int64).copy()
        member_returns = torch.tensor(
            [
                [-1.0, -1.2, -0.8, -1.1, -0.9],
                [-2.0, -2.2, -1.8, -2.1, -1.9],
                [-3.0, -3.2, -2.8, -3.1, -2.9],
                [-4.0, -4.2, -3.8, -4.1, -3.9],
                [-5.0, -5.2, -4.8, -5.1, -4.9],
                [-6.0, -6.2, -5.8, -6.1, -5.9],
            ],
            dtype=torch.float32,
        )
        return SimpleNamespace(
            member_returns=member_returns,
            expected_return=member_returns.mean(dim=1),
        )


class TestGateB4LWMRuntime(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260917)
        self.registry = load_model_registry(REGISTRY_PATH)
        self.spec = self.registry.by_alias("llm_l_gemini35_flash_lite")
        self.policy = CandidateActorCritic(PPOCoreConfig())
        self.evaluator = FakeEvaluator()
        self.generator_calls = []

    def make_runtime(self, cache_root, *, split="train"):
        def generate(state, agent_name):
            self.generator_calls.append((agent_name, np.asarray(state).copy()))
            return sample_prior(), metadata()

        return LWMDecisionRuntime(
            registry=self.registry,
            model_spec=self.spec,
            registry_sha256=REGISTRY_SHA,
            evaluator=self.evaluator,
            prior_cache=PriorCache(cache_root),
            policy=self.policy,
            live_prior_generator=generate,
            split=split,
        )

    def test_prepare_produces_frozen_6x46_features_and_scalar_critic(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime = self.make_runtime(temp)
            state = np.zeros(FORMAL_STATE_DIM, dtype=np.float32)
            prepared = runtime.prepare(state, agent_name="blue_agent_0")
            self.assertEqual(tuple(prepared.candidate_features.shape), (6, 46))
            self.assertTrue(torch.isfinite(prepared.candidate_features).all())
            self.assertTrue(np.isfinite(prepared.critic_value))
            self.assertFalse(prepared.cache_hit)
            self.assertEqual(runtime.cache_misses, 1)
            self.assertEqual(runtime.live_api_calls, 1)

    def test_same_agent_state_hits_cache_without_second_generator_call(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime = self.make_runtime(temp)
            state = np.zeros(FORMAL_STATE_DIM, dtype=np.float32)
            first = runtime.prepare(state, agent_name="blue_agent_0")
            second = runtime.prepare(state, agent_name="blue_agent_0")
            self.assertFalse(first.cache_hit)
            self.assertTrue(second.cache_hit)
            self.assertEqual(first.cache_key, second.cache_key)
            self.assertEqual(len(self.generator_calls), 1)
            self.assertEqual(runtime.cache_hits, 1)
            self.assertEqual(runtime.cache_misses, 1)

    def test_identical_d27_different_agents_get_different_cache_keys(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime = self.make_runtime(temp)
            state = np.zeros(FORMAL_STATE_DIM, dtype=np.float32)
            a0 = runtime.prepare(state, agent_name="blue_agent_0")
            a1 = runtime.prepare(state, agent_name="blue_agent_1")
            self.assertNotEqual(a0.cache_key, a1.cache_key)
            self.assertEqual(len(self.generator_calls), 2)
            self.assertEqual(runtime.cache_misses, 2)

    def test_train_identity_missing_agent_fails_closed(self):
        state = np.zeros(FORMAL_STATE_DIM, dtype=np.float32)
        with self.assertRaises(ValueError):
            make_identity(
                split="train",
                model_alias=self.spec.experiment_alias,
                provider=self.spec.provider,
                exact_model_id=self.spec.exact_model_id,
                registry_version=self.registry.version,
                registry_sha256=REGISTRY_SHA,
                prompt_version=self.registry.prompt_version,
                generation_config={"temperature": 0.2},
                state=state,
            )

    def test_historical_validation_identity_can_use_explicit_legacy_sentinel_path(self):
        state = np.zeros(FORMAL_STATE_DIM, dtype=np.float32)
        identity = make_identity(
            split="validation",
            model_alias=self.spec.experiment_alias,
            provider=self.spec.provider,
            exact_model_id=self.spec.exact_model_id,
            registry_version=self.registry.version,
            registry_sha256=REGISTRY_SHA,
            prompt_version=self.registry.prompt_version,
            generation_config={"temperature": 0.2},
            state=state,
        )
        self.assertEqual(identity.agent_name, "legacy_validation_unique_bank")

    def test_select_returns_candidate_index_and_exact_plan_zero_action(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime = self.make_runtime(temp)
            prepared = runtime.prepare(
                np.zeros(FORMAL_STATE_DIM, dtype=np.float32),
                agent_name="blue_agent_0",
            )
            selected = runtime.select(prepared, deterministic=True)
            self.assertGreaterEqual(selected.candidate_index, 0)
            self.assertLess(selected.candidate_index, 6)
            self.assertEqual(
                selected.requested_action_id,
                int(prepared.prior.plans[selected.candidate_index, 0]),
            )
            self.assertEqual(selected.requested_action_id, selected.selected_plan[0])
            self.assertEqual(len(selected.selected_plan), 4)
            self.assertTrue(np.isfinite(selected.behavior_log_prob))
            self.assertTrue(np.isfinite(selected.entropy))

    def test_runtime_generation_config_records_public_agent_context(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime = self.make_runtime(temp)
            self.assertTrue(runtime.generation_config["public_agent_identifier"])
            self.assertEqual(runtime.generation_config["temperature"], 0.2)

    def test_runtime_rejects_unknown_agent_and_bad_state(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime = self.make_runtime(temp)
            with self.assertRaises(ValueError):
                runtime.prepare(np.zeros(27, dtype=np.float32), agent_name="red_agent_0")
            with self.assertRaises(ValueError):
                runtime.prepare(np.zeros(26, dtype=np.float32), agent_name="blue_agent_0")
            bad = np.zeros(27, dtype=np.float32)
            bad[0] = np.nan
            with self.assertRaises(ValueError):
                runtime.prepare(bad, agent_name="blue_agent_0")


if __name__ == "__main__":
    unittest.main()
