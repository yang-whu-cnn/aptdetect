import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from formal_experiments.evaluation.build_llm_validation_state_bank import state_sha256
from formal_experiments.evaluation.run_b2_repeatability import (
    analyze_state_replicates,
    make_repeatability_identity,
    run_model_repeatability,
    select_repeatability_subset,
    summarize_repeatability,
)
from formal_experiments.ours.model_registry import load_model_registry
from formal_experiments.ours.prior_cache import PriorCache
from shared.formal_state import BLUE_AGENTS, FORMAL_STATE_DIM
from shared.rollout_evaluator import SharedRolloutResult


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "configs" / "llm_model_registry_v1.yaml"
REGISTRY_SHA = "a" * 64


def valid_text(offset=0):
    plans = [
        ["no_op", "analyse", "remove", "restore"],
        ["analyse", "analyse", "remove", "restore"],
        ["remove", "analyse", "remove", "restore"],
        ["restore", "analyse", "remove", "restore"],
        ["no_op", "no_op", "no_op", "no_op"],
        ["restore", "restore", "restore", "restore"],
    ]
    if offset:
        plans = plans[offset:] + plans[:offset]
    return json.dumps(
        {
            "candidates": [
                {
                    "actions": plan,
                    "prior_score": 0.9 - index * 0.1,
                    "reason": f"r{index}",
                }
                for index, plan in enumerate(plans)
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
        base = plans.sum(axis=1).astype(np.float32)
        values = torch.as_tensor(base, dtype=torch.float32)
        members = torch.stack(
            [values + offset for offset in (0.0, 0.1, -0.1, 0.2, -0.2)],
            dim=1,
        )
        next_states = torch.zeros((4, n, 5, FORMAL_STATE_DIM), dtype=torch.float32)
        return SharedRolloutResult(
            next_states=next_states,
            member_returns=members,
            expected_return=members.mean(dim=1),
        )


def build_bank_rows():
    rows = []
    counter = 1
    for seed in range(2000, 2008):
        for agent_index, agent in enumerate(BLUE_AGENTS):
            for local_index in range(6):
                state = np.zeros(FORMAL_STATE_DIM, dtype=np.float32)
                state[0] = np.float32(counter / 10000.0)
                state[1] = np.float32(agent_index / 10.0)
                state[17] = np.float32(0 if local_index < 3 else 1)
                rows.append(
                    {
                        "bank_format_version": 1,
                        "split": "validation",
                        "source_replay": "validation.jsonl",
                        "selection_config_sha256": "b" * 64,
                        "episode_seed": seed,
                        "agent_name": agent,
                        "decision_index": local_index,
                        "global_tick_start": local_index * 10,
                        "feature17_any_valid_observable_target": int(state[17]),
                        "state_dtype": "float32_le",
                        "state_sha256": state_sha256(state),
                        "state": state.tolist(),
                    }
                )
                counter += 1
    return rows


def one_state_row():
    return build_bank_rows()[0]


def quality_record(replicate, plans, prefs, best_value):
    return {
        "status": "ok",
        "replicate": replicate,
        "quality": {
            "plans": plans,
            "prior_preferences": prefs,
            "top_prior_index": int(np.argmax(np.asarray(prefs))),
            "best_of_k_value": float(best_value),
        },
    }


class TestGateB2Repeatability(unittest.TestCase):
    def setUp(self):
        self.registry = load_model_registry(REGISTRY_PATH)
        self.spec = self.registry.get("tier_h")

    def test_subset_is_deterministic_balanced_and_unique(self):
        rows = build_bank_rows()
        first, first_summary = select_repeatability_subset(rows)
        second, second_summary = select_repeatability_subset(rows)
        self.assertEqual(first, second)
        self.assertEqual(first_summary["subset_sha256"], second_summary["subset_sha256"])
        self.assertEqual(len(first), 30)
        self.assertEqual(len({row["state_sha256"] for row in first}), 30)
        self.assertEqual(set(first_summary["per_agent"].values()), {6})
        self.assertEqual(first_summary["feature17_counts"], {"0": 15, "1": 15})
        self.assertEqual(sorted(first_summary["per_seed"].values()), [3, 3, 4, 4, 4, 4, 4, 4])

    def test_subset_rejects_missing_feature17_in_selected_cell(self):
        rows = build_bank_rows()
        for row in rows:
            if row["episode_seed"] == 2000 and row["agent_name"] == BLUE_AGENTS[0]:
                row["feature17_any_valid_observable_target"] = 1
        with self.assertRaises(ValueError):
            select_repeatability_subset(rows)

    def test_replicate_identity_isolated_by_replicate(self):
        state = np.asarray(one_state_row()["state"], dtype=np.float32)
        identities = [
            make_repeatability_identity(
                spec=self.spec,
                registry=self.registry,
                registry_sha256=REGISTRY_SHA,
                state=state,
                replicate=index,
            )
            for index in range(3)
        ]
        self.assertEqual(len({identity.cache_key for identity in identities}), 3)
        self.assertEqual(
            [identity.generation_config["replicate"] for identity in identities],
            [0, 1, 2],
        )

    def test_identical_replicates_have_perfect_overlap_and_zero_value_variance(self):
        plans = [[0, 1, 2, 3], [1, 1, 2, 3], [2, 1, 2, 3], [3, 1, 2, 3], [0, 0, 0, 0], [3, 3, 3, 3]]
        prefs = [0.3, 0.2, 0.15, 0.15, 0.1, 0.1]
        records = [quality_record(i, plans, prefs, -2.0) for i in range(3)]
        result = analyze_state_replicates(records)
        self.assertAlmostEqual(result["candidate_set_jaccard_mean"], 1.0)
        self.assertAlmostEqual(result["top_prior_pair_agreement_mean"], 1.0)
        self.assertTrue(result["top_prior_all_three_agree"])
        self.assertAlmostEqual(result["best_of_k_value_variance"], 0.0)

    def test_repeatability_metrics_detect_changed_sets_and_values(self):
        plans_a = [[0, 0, 0, x] for x in range(4)] + [[1, 0, 0, 0], [2, 0, 0, 0]]
        plans_b = [[3, 3, 3, x] for x in range(4)] + [[1, 3, 3, 3], [2, 3, 3, 3]]
        plans_c = [[0, 1, 0, x] for x in range(4)] + [[1, 1, 0, 0], [2, 1, 0, 0]]
        prefs = [0.4, 0.2, 0.1, 0.1, 0.1, 0.1]
        records = [
            quality_record(0, plans_a, prefs, -1.0),
            quality_record(1, plans_b, prefs, -2.0),
            quality_record(2, plans_c, prefs, -3.0),
        ]
        result = analyze_state_replicates(records)
        self.assertLess(result["candidate_set_jaccard_mean"], 1.0)
        self.assertFalse(result["top_prior_all_three_agree"])
        self.assertGreater(result["best_of_k_value_variance"], 0.0)

    def test_three_replicates_are_three_distinct_live_transactions_then_cache_hits(self):
        row = one_state_row()
        calls = {"factory": 0}

        def factory():
            calls["factory"] += 1
            return FakeClient([response(), response(), response()])

        with tempfile.TemporaryDirectory() as temp:
            cache = PriorCache(temp)
            first = run_model_repeatability(
                spec=self.spec,
                registry=self.registry,
                registry_sha256=REGISTRY_SHA,
                rows=[row],
                evaluator=FakeEvaluator(),
                cache=cache,
                client_factory=factory,
            )
            second = run_model_repeatability(
                spec=self.spec,
                registry=self.registry,
                registry_sha256=REGISTRY_SHA,
                rows=[row],
                evaluator=FakeEvaluator(),
                cache=cache,
                client_factory=factory,
            )
        self.assertTrue(first["pass"])
        self.assertTrue(second["pass"])
        self.assertEqual(calls["factory"], 1)
        self.assertEqual(first["summary"]["live_api_calls_this_run"], 3)
        self.assertEqual(second["summary"]["live_api_calls_this_run"], 0)
        self.assertEqual(second["summary"]["cache_hits_this_run"], 3)

    def test_schema_invalid_success_remains_visible_in_repeatability_summary(self):
        metadata = {
            "schema_valid": False,
            "semantic_valid_candidate_count": 0,
            "fallback_count": 6,
            "duplicate_candidate_count": 0,
            "retry_count": 0,
            "latency_ms": 1.0,
            "cost_usd": 0.0,
        }
        quality = quality_record(0, [[0, 0, 0, 0]] * 6, [1 / 6] * 6, -1.0)["quality"]
        generations = [
            {
                "status": "ok",
                "cache_hit": False,
                "metadata": dict(metadata),
                "quality": quality,
            }
            for _ in range(3)
        ]
        analysis = {
            "candidate_set_jaccard_mean": 1.0,
            "candidate_overlap_count_mean": 1.0,
            "top_prior_pair_agreement_mean": 1.0,
            "top_prior_all_three_agree": True,
            "top_prior_preference_variance": 0.0,
            "matched_plan_preference_variance_mean": 0.0,
            "best_of_k_value_variance": 0.0,
            "best_of_k_value_range": 0.0,
        }
        summary = summarize_repeatability(
            [{"status": "ok", "repeatability": analysis}],
            generations,
            expected_states=1,
        )
        self.assertEqual(summary["schema_valid_generation_rate"], 0.0)
        self.assertEqual(summary["fallback_candidate_rate"], 1.0)
        self.assertTrue(summary["pass"])

    def test_failed_generation_makes_gate_fail(self):
        summary = summarize_repeatability(
            [],
            [{"status": "failed"}],
            expected_states=1,
        )
        self.assertFalse(summary["pass"])
        self.assertEqual(summary["failed_generations"], 1)

    def test_provider_failure_report_does_not_serialize_exception_message(self):
        row = one_state_row()

        def factory():
            return FakeClient([RuntimeError("secret-provider-message")])

        with tempfile.TemporaryDirectory() as temp:
            report = run_model_repeatability(
                spec=self.spec,
                registry=self.registry,
                registry_sha256=REGISTRY_SHA,
                rows=[row],
                evaluator=FakeEvaluator(),
                cache=PriorCache(temp),
                client_factory=factory,
            )
        dumped = json.dumps(report)
        self.assertFalse(report["pass"])
        self.assertNotIn("secret-provider-message", dumped)

    def test_primary_model_remains_unselected(self):
        self.assertFalse(self.registry.primary_model_selected)


if __name__ == "__main__":
    unittest.main()
