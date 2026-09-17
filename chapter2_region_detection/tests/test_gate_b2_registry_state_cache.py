import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from formal_experiments.data_collection.collect_cc4_formal_replay import DEFAULT_SPLIT_SEEDS
from formal_experiments.data_collection.decision_replay import DecisionEpochTransition
from formal_experiments.evaluation.build_llm_validation_state_bank import (
    FEATURE17_INDEX,
    TARGET_STATES_PER_CELL,
    build_state_bank,
    select_cell,
)
from formal_experiments.ours.llm_prior_v2 import PriorBatch
from formal_experiments.ours.model_registry import load_model_registry
from formal_experiments.ours.prior_cache import (
    PriorCache,
    PriorCacheCorruptError,
    exact_state_sha256,
    make_identity,
)
from shared.formal_state import BLUE_AGENTS, FORMAL_STATE_DIM


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs" / "lwm_rl_llm_model_registry_v2_3.yaml"


def transition(seed: int, agent: str, decision: int, feature17: int, *, completed: bool = True):
    state = np.zeros(FORMAL_STATE_DIM, dtype=np.float32)
    state[0] = np.float32((decision + 1) / 100.0)
    state[FEATURE17_INDEX] = np.float32(feature17)
    next_state = state.copy()
    next_state[1] = np.float32((decision + 2) / 100.0)
    return DecisionEpochTransition(
        episode_seed=int(seed),
        agent_name=str(agent),
        decision_index=int(decision),
        global_tick_start=int(decision),
        global_tick_end=int(decision + 1),
        decision_dt=1,
        state=state,
        next_state=next_state,
        requested_action_id=0,
        requested_action_name="no_op",
        requested_cyborg_family="Sleep",
        requested_duration_ticks=1,
        executed_index=0,
        executed_label="Sleep",
        executed_action_family="Sleep",
        executed_duration_ticks=1,
        target_host=None,
        fallback=False,
        fallback_reason=None,
        action_completed=bool(completed),
        completed_action_success=True if completed else None,
        incident_event_ids=tuple(),
        incident_host_ids=tuple(),
        incident_active_ticks=0,
        incident_host_lwf_count=0,
        incident_host_lwf_raw_penalty=0.0,
        response_reward=0.0,
        official_reward=0.0,
        done=False,
    )


def full_validation_transitions():
    rows = []
    for seed in DEFAULT_SPLIT_SEEDS["validation"]:
        for agent in BLUE_AGENTS:
            for decision in range(8):
                rows.append(transition(seed, agent, decision, decision % 2))
    return tuple(rows)


def sample_prior():
    plans = np.asarray(
        [
            [0, 1, 2, 3],
            [1, 1, 2, 3],
            [2, 1, 2, 3],
            [3, 1, 2, 3],
            [0, 0, 0, 0],
            [3, 3, 3, 3],
        ],
        dtype=np.int64,
    )
    scores = np.asarray([0.9, 0.8, 0.7, 0.6, 0.5, 0.4], dtype=np.float32)
    prefs = scores / scores.sum()
    return PriorBatch(
        plans=plans,
        raw_prior_scores=scores,
        prior_preferences=prefs.astype(np.float32),
        sources=("llm",) * 6,
    )


class TestGateB2RegistryStateCache(unittest.TestCase):
    def test_registry_has_exact_three_candidate_tiers_and_no_primary(self):
        registry = load_model_registry(REGISTRY)
        self.assertEqual(tuple(registry.models.keys()), ("tier_h", "tier_m", "tier_l"))
        self.assertFalse(registry.primary_model_selected)
        self.assertEqual(registry.get("tier_h").exact_model_id, "openai/gpt-5.6-sol")
        self.assertEqual(registry.get("tier_m").exact_model_id, "openai/gpt-5.4-mini")
        self.assertEqual(registry.get("tier_l").exact_model_id, "google/gemini-3.5-flash-lite")
        self.assertEqual(registry.base_url, "https://api.ofox.ai/v1")

    def test_registry_converts_each_model_to_same_formal_prior_contract(self):
        registry = load_model_registry(REGISTRY)
        for spec in registry.models.values():
            cfg = spec.to_prior_config(
                prompt_version=registry.prompt_version,
                k_candidates=registry.k_candidates,
                horizon=registry.horizon,
            )
            self.assertEqual(cfg.k_candidates, 6)
            self.assertEqual(cfg.horizon, 4)
            self.assertEqual(cfg.temperature, 0.2)
            self.assertEqual(cfg.api_model, spec.exact_model_id)

    def test_registry_pricing_forms_strict_cost_gradient(self):
        registry = load_model_registry(REGISTRY)
        high = registry.get("tier_h").pricing_snapshot
        medium = registry.get("tier_m").pricing_snapshot
        low = registry.get("tier_l").pricing_snapshot
        self.assertGreater(high.input, medium.input)
        self.assertGreater(medium.input, low.input)
        self.assertGreater(high.output, medium.output)
        self.assertGreater(medium.output, low.output)

    def test_select_cell_covers_both_feature17_values(self):
        rows = [transition(2000, "blue_agent_0", index, index % 2) for index in range(12)]
        selected = select_cell(rows)
        self.assertEqual(len(selected), TARGET_STATES_PER_CELL)
        self.assertEqual({int(item.state[FEATURE17_INDEX]) for item in selected}, {0, 1})

    def test_select_cell_single_feature_still_returns_exact_six(self):
        rows = [transition(2000, "blue_agent_0", index, 1) for index in range(10)]
        selected = select_cell(rows)
        self.assertEqual(len(selected), 6)
        self.assertEqual({int(item.state[FEATURE17_INDEX]) for item in selected}, {1})

    def test_state_bank_is_exact_240_balanced_and_planner_visible_only(self):
        records, summary = build_state_bank(full_validation_transitions())
        self.assertEqual(len(records), 240)
        self.assertEqual(summary["record_count"], 240)
        self.assertTrue(summary["pass"])
        self.assertFalse(summary["hidden_truth_fields"])
        self.assertFalse(summary["reward_fields"])
        self.assertFalse(summary["future_state_fields"])
        self.assertEqual(set(summary["per_seed"].values()), {30})
        self.assertEqual(set(summary["per_agent"].values()), {48})
        self.assertEqual(set(summary["per_cell"].values()), {6})
        allowed = {
            "split",
            "episode_seed",
            "agent_name",
            "decision_index",
            "global_tick_start",
            "feature17_any_valid_observable_target",
            "state_sha256",
            "state",
        }
        self.assertTrue(all(set(row) == allowed for row in records))
        self.assertTrue(all(len(row["state"]) == FORMAL_STATE_DIM for row in records))

    def test_state_bank_is_deterministic(self):
        rows = full_validation_transitions()
        first, first_summary = build_state_bank(rows)
        second, second_summary = build_state_bank(tuple(reversed(rows)))
        self.assertEqual(first, second)
        self.assertEqual(first_summary["bank_sha256"], second_summary["bank_sha256"])

    def test_state_hash_is_exact_float32_not_quantized(self):
        left = np.zeros(FORMAL_STATE_DIM, dtype=np.float32)
        right = left.copy()
        right[0] = np.nextafter(np.float32(0.0), np.float32(1.0), dtype=np.float32)
        self.assertNotEqual(exact_state_sha256(left), exact_state_sha256(right))

    def test_cache_key_isolated_by_split_model_and_prompt(self):
        state = np.zeros(FORMAL_STATE_DIM, dtype=np.float32)
        base = dict(
            provider="ofox",
            exact_model_id="openai/gpt-5.6-sol",
            prompt_version="p1",
            generation_config={"temperature": 0.2, "structured_output": "json_schema"},
            state=state,
        )
        train = make_identity(split="train", model_alias="m1", **base)
        validation = make_identity(split="validation", model_alias="m1", **base)
        other_model = make_identity(split="train", model_alias="m2", **{**base, "exact_model_id": "openai/gpt-5.4-mini"})
        other_prompt = make_identity(split="train", model_alias="m1", **{**base, "prompt_version": "p2"})
        self.assertEqual(len({train.cache_key, validation.cache_key, other_model.cache_key, other_prompt.cache_key}), 4)

    def test_cache_miss_calls_generator_once_and_hit_calls_zero_times(self):
        state = np.zeros(FORMAL_STATE_DIM, dtype=np.float32)
        identity = make_identity(
            split="train",
            model_alias="m1",
            provider="ofox",
            exact_model_id="openai/gpt-5.6-sol",
            prompt_version="p1",
            generation_config={"temperature": 0.2},
            state=state,
        )
        calls = {"n": 0}
        def generator():
            calls["n"] += 1
            return sample_prior(), {"latency_ms": 123.0, "usage": {"input_tokens": 10, "output_tokens": 20}}
        with tempfile.TemporaryDirectory() as temp:
            cache = PriorCache(temp)
            first = cache.get_or_generate(identity, generator)
            second = cache.get_or_generate(identity, generator)
            self.assertFalse(first.cache_hit)
            self.assertTrue(second.cache_hit)
            self.assertEqual(calls["n"], 1)
            np.testing.assert_array_equal(first.cached_prior.prior.plans, second.cached_prior.prior.plans)

    def test_cache_rejects_sensitive_metadata(self):
        state = np.zeros(FORMAL_STATE_DIM, dtype=np.float32)
        identity = make_identity(
            split="validation",
            model_alias="m1",
            provider="ofox",
            exact_model_id="openai/gpt-5.6-sol",
            prompt_version="p1",
            generation_config={"temperature": 0.2},
            state=state,
        )
        with tempfile.TemporaryDirectory() as temp:
            cache = PriorCache(temp)
            with self.assertRaises(ValueError):
                cache.store(identity, sample_prior(), metadata={"api_key": "secret"})

    def test_corrupt_cache_is_hard_failure_not_silent_miss(self):
        state = np.zeros(FORMAL_STATE_DIM, dtype=np.float32)
        identity = make_identity(
            split="test",
            model_alias="m1",
            provider="ofox",
            exact_model_id="openai/gpt-5.6-sol",
            prompt_version="p1",
            generation_config={"temperature": 0.2},
            state=state,
        )
        with tempfile.TemporaryDirectory() as temp:
            cache = PriorCache(temp)
            path = cache.path_for(identity)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{not-json", encoding="utf-8")
            with self.assertRaises(PriorCacheCorruptError):
                cache.load(identity)

    def test_cache_atomic_store_leaves_no_temp_file(self):
        state = np.zeros(FORMAL_STATE_DIM, dtype=np.float32)
        identity = make_identity(
            split="train",
            model_alias="m1",
            provider="ofox",
            exact_model_id="openai/gpt-5.6-sol",
            prompt_version="p1",
            generation_config={"temperature": 0.2},
            state=state,
        )
        with tempfile.TemporaryDirectory() as temp:
            cache = PriorCache(temp)
            cached = cache.store(identity, sample_prior(), metadata={"latency_ms": 1.0})
            self.assertTrue(cached.path.exists())
            leftovers = list(cached.path.parent.glob("*.tmp")) + list(cached.path.parent.glob(".*.tmp"))
            self.assertEqual(leftovers, [])
            payload = json.loads(cached.path.read_text(encoding="utf-8"))
            self.assertEqual(payload["identity"]["split"], "train")
            self.assertNotIn("api_key", json.dumps(payload).lower())


if __name__ == "__main__":
    unittest.main()
