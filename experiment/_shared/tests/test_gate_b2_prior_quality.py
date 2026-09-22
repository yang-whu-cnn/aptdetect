import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from formal_experiments.evaluation.run_b2_prior_quality import (
    estimate_cost_usd,
    evaluate_prior,
    evaluate_uniform_baseline,
    request_prior_once_logical_transaction,
    run_model,
    spearman,
    summarize_model_records,
    uniform_prior,
)
from formal_experiments.ours.llm_prior_v2 import PriorBatch
from formal_experiments.ours.model_registry import load_model_registry
from formal_experiments.ours.prior_cache import PriorCache
from shared.formal_state import FORMAL_STATE_DIM
from shared.rollout_evaluator import SharedRolloutResult


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "configs" / "llm_model_registry_v1.yaml"
REGISTRY_SHA = "a" * 64


def valid_text():
    return json.dumps(
        {
            "candidates": [
                {"actions": ["no_op", "analyse", "remove", "restore"], "prior_score": 0.9, "reason": "a"},
                {"actions": ["analyse", "analyse", "remove", "restore"], "prior_score": 0.8, "reason": "b"},
                {"actions": ["remove", "analyse", "remove", "restore"], "prior_score": 0.7, "reason": "c"},
                {"actions": ["restore", "analyse", "remove", "restore"], "prior_score": 0.6, "reason": "d"},
                {"actions": ["no_op", "no_op", "no_op", "no_op"], "prior_score": 0.5, "reason": "e"},
                {"actions": ["restore", "restore", "restore", "restore"], "prior_score": 0.4, "reason": "f"},
            ]
        }
    )


def response(text=None):
    return SimpleNamespace(
        model="openai/gpt-5.6-sol",
        choices=[SimpleNamespace(message=SimpleNamespace(content=valid_text() if text is None else text))],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50, total_tokens=150),
    )


class FakeCompletions:
    def __init__(self, sequence):
        self.sequence = list(sequence)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.sequence.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeClient:
    def __init__(self, sequence):
        self.chat = SimpleNamespace(completions=FakeCompletions(sequence))


class FakeEvaluator:
    def evaluate(self, state, plans):
        plans = np.asarray(plans, dtype=np.int64)
        n = plans.shape[0]
        values = torch.arange(1, n + 1, dtype=torch.float32)
        members = torch.stack([values + offset for offset in (0.0, 0.1, -0.1, 0.2, -0.2)], dim=1)
        next_states = torch.zeros((4, n, 5, FORMAL_STATE_DIM), dtype=torch.float32)
        return SharedRolloutResult(
            next_states=next_states,
            member_returns=members,
            expected_return=members.mean(dim=1),
        )


def state_row(index=0):
    state = np.zeros(FORMAL_STATE_DIM, dtype=np.float32)
    state[0] = np.float32(index / 100.0)
    import hashlib
    digest = hashlib.sha256(np.asarray(state, dtype="<f4").tobytes()).hexdigest()
    return {
        "state": state.tolist(),
        "state_sha256": digest,
        "episode_seed": 2000,
        "agent_name": "blue_agent_0",
        "decision_index": index,
    }


class TestGateB2PriorQuality(unittest.TestCase):
    def setUp(self):
        self.registry = load_model_registry(REGISTRY_PATH)
        self.spec = self.registry.get("tier_h")

    def test_registry_is_live_preflight_verified_but_primary_unselected(self):
        self.assertFalse(self.registry.primary_model_selected)
        for spec in self.registry.models.values():
            self.assertTrue(spec.structured_output_verified)
            self.assertEqual(spec.actual_temperature, 0.2)

    def test_cost_estimate_uses_registry_snapshot_and_usage(self):
        usage = {"prompt_tokens": 1000, "completion_tokens": 2000}
        expected = (
            1000 * self.spec.pricing_snapshot.input
            + 2000 * self.spec.pricing_snapshot.output
        ) / 1_000_000.0
        self.assertAlmostEqual(estimate_cost_usd(self.spec, usage), expected)
        self.assertIsNone(estimate_cost_usd(self.spec, {"prompt_tokens": None, "completion_tokens": 2}))

    def test_spearman_handles_order_and_constant_prior(self):
        self.assertAlmostEqual(spearman([1, 2, 3], [10, 20, 30]), 1.0)
        self.assertAlmostEqual(spearman([1, 2, 3], [30, 20, 10]), -1.0)
        self.assertIsNone(spearman([1, 1, 1], [1, 2, 3]))

    def test_provider_transaction_records_raw_contract_usage_cost_and_no_raw_text(self):
        client = FakeClient([response()])
        state = np.zeros(FORMAL_STATE_DIM, dtype=np.float32)
        prior, metadata = request_prior_once_logical_transaction(
            spec=self.spec,
            registry=self.registry,
            state=state,
            agent_name="blue_agent_0",
            client=client,
            sleep_fn=lambda _: None,
        )
        self.assertEqual(prior.plans.shape, (6, 4))
        self.assertTrue(metadata["api_success"])
        self.assertTrue(metadata["schema_valid"])
        self.assertEqual(metadata["semantic_valid_candidate_count"], 6)
        self.assertEqual(metadata["fallback_count"], 0)
        self.assertEqual(metadata["usage"]["total_tokens"], 150)
        self.assertIsNotNone(metadata["cost_usd"])
        self.assertNotIn("raw_response", metadata)
        call = client.chat.completions.calls[0]
        self.assertEqual(call["temperature"], 0.2)
        self.assertEqual(call["response_format"]["type"], "json_schema")

    def test_schema_invalid_success_is_measured_and_parser_fallback_remains_auditable(self):
        client = FakeClient([response(text="not-json")])
        prior, metadata = request_prior_once_logical_transaction(
            spec=self.spec,
            registry=self.registry,
            state=np.zeros(FORMAL_STATE_DIM, dtype=np.float32),
            agent_name="blue_agent_0",
            client=client,
            sleep_fn=lambda _: None,
        )
        self.assertTrue(metadata["api_success"])
        self.assertFalse(metadata["schema_valid"])
        self.assertEqual(metadata["semantic_valid_candidate_count"], 0)
        self.assertEqual(metadata["fallback_count"], 6)
        self.assertEqual(sum(source != "llm" for source in prior.sources), 6)

    def test_retryable_provider_failure_is_bounded(self):
        client = FakeClient([RuntimeError("503 unavailable"), response()])
        _, metadata = request_prior_once_logical_transaction(
            spec=self.spec,
            registry=self.registry,
            state=np.zeros(FORMAL_STATE_DIM, dtype=np.float32),
            agent_name="blue_agent_0",
            client=client,
            sleep_fn=lambda _: None,
        )
        self.assertEqual(metadata["attempts"], 2)
        self.assertEqual(metadata["retry_count"], 1)

    def test_evaluate_prior_reports_value_uncertainty_rank_and_unique_plans(self):
        prior = uniform_prior()
        result = evaluate_prior(
            np.zeros(FORMAL_STATE_DIM, dtype=np.float32),
            prior,
            FakeEvaluator(),
        )
        self.assertEqual(result["best_of_k_value"], 6.0)
        self.assertEqual(result["unique_plan_ratio"], 1.0)
        self.assertEqual(len(result["predicted_values"]), 6)
        self.assertEqual(len(result["predictive_uncertainty"]), 6)
        self.assertIsNone(result["prior_value_spearman"])

    def test_cache_hit_does_not_construct_provider_client(self):
        row = state_row()
        evaluator = FakeEvaluator()
        calls = {"factory": 0}

        def factory():
            calls["factory"] += 1
            return FakeClient([response()])

        with tempfile.TemporaryDirectory() as temp:
            cache = PriorCache(temp)
            first = run_model(
                spec=self.spec,
                registry=self.registry,
                registry_sha256=REGISTRY_SHA,
                rows=[row],
                evaluator=evaluator,
                cache=cache,
                client_factory=factory,
            )
            second = run_model(
                spec=self.spec,
                registry=self.registry,
                registry_sha256=REGISTRY_SHA,
                rows=[row],
                evaluator=evaluator,
                cache=cache,
                client_factory=factory,
            )
            self.assertTrue(first["pass"])
            self.assertTrue(second["pass"])
            self.assertEqual(calls["factory"], 1)
            self.assertEqual(first["summary"]["cache_misses_this_run"], 1)
            self.assertEqual(second["summary"]["cache_hits_this_run"], 1)
            self.assertEqual(second["summary"]["live_api_calls_this_run"], 0)

    def test_uniform_baseline_is_deterministic_and_zero_api(self):
        rows = [state_row(0), state_row(1)]
        first = evaluate_uniform_baseline(rows, FakeEvaluator())
        second = evaluate_uniform_baseline(rows, FakeEvaluator())
        self.assertEqual(first["fixed_plans"], second["fixed_plans"])
        self.assertEqual(first["summary"]["live_api_calls_this_run"], 0)
        self.assertTrue(first["pass"])

    def test_summary_keeps_raw_schema_and_fallback_metrics_separate(self):
        quality = evaluate_prior(
            np.zeros(FORMAL_STATE_DIM, dtype=np.float32),
            uniform_prior(),
            FakeEvaluator(),
        )
        records = [
            {
                "status": "ok",
                "cache_hit": False,
                "metadata": {
                    "schema_valid": False,
                    "semantic_valid_candidate_count": 0,
                    "duplicate_candidate_count": 0,
                    "fallback_count": 6,
                    "retry_count": 0,
                    "latency_ms": 1.0,
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                    "cost_usd": 0.0,
                },
                "quality": quality,
            }
        ]
        summary = summarize_model_records(records, expected_states=1)
        self.assertEqual(summary["schema_valid_response_rate"], 0.0)
        self.assertEqual(summary["fallback_candidate_rate"], 1.0)
        self.assertTrue(summary["pass"])


if __name__ == "__main__":
    unittest.main()
