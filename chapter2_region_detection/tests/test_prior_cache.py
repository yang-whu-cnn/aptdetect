import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from formal_experiments.ours.llm_prior_v2 import PriorBatch
from formal_experiments.ours.prior_cache import (
    CACHE_FORMAT_VERSION,
    PriorCache,
    PriorCacheCorruptError,
    exact_state_sha256,
    make_identity,
)
from shared.formal_state import FORMAL_STATE_DIM


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs" / "llm_model_registry_v1.yaml"
REGISTRY_SHA256 = hashlib.sha256(REGISTRY.read_bytes()).hexdigest()


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


def sample_metadata():
    return {
        "api_success": True,
        "fallback_count": 0,
        "prompt_sha256": "1" * 64,
        "response_sha256": "2" * 64,
        "latency_ms": 123.0,
        "retry_count": 0,
        "usage": {"input_tokens": 10, "output_tokens": 20},
        "cost_usd": 0.001,
    }


def identity_for(
    state,
    *,
    split="train",
    alias="m1",
    model="openai/gpt-5.6-sol",
    prompt="p1",
    temperature=0.2,
    registry_sha=REGISTRY_SHA256,
):
    return make_identity(
        split=split,
        model_alias=alias,
        provider="ofox",
        exact_model_id=model,
        registry_version=1,
        registry_sha256=registry_sha,
        prompt_version=prompt,
        generation_config={"temperature": temperature, "structured_output": "json_schema"},
        state=state,
    )


class TestPriorCache(unittest.TestCase):
    def test_exact_state_hash_is_not_quantized(self):
        left = np.zeros(FORMAL_STATE_DIM, dtype=np.float32)
        right = left.copy()
        right[0] = np.nextafter(np.float32(0.0), np.float32(1.0))
        self.assertNotEqual(exact_state_sha256(left), exact_state_sha256(right))

    def test_cache_key_isolated_by_split_model_prompt_config_and_registry(self):
        state = np.zeros(FORMAL_STATE_DIM, dtype=np.float32)
        train = identity_for(state)
        validation = identity_for(state, split="validation")
        other_model = identity_for(state, alias="m2", model="openai/gpt-5.4-mini")
        other_prompt = identity_for(state, prompt="p2")
        other_config = identity_for(state, temperature=0.3)
        other_registry = identity_for(state, registry_sha="a" * 64)
        keys = {
            train.cache_key,
            validation.cache_key,
            other_model.cache_key,
            other_prompt.cache_key,
            other_config.cache_key,
            other_registry.cache_key,
        }
        self.assertEqual(len(keys), 6)
        payload = train.canonical_payload()
        self.assertEqual(payload["cache_format_version"], CACHE_FORMAT_VERSION)
        self.assertEqual(payload["n_actions"], 4)
        self.assertEqual(payload["state_shape"], [27])
        self.assertEqual(payload["state_dtype"], "float32_le")
        self.assertEqual(payload["registry_sha256"], REGISTRY_SHA256)

    def test_cache_miss_calls_generator_once_and_hit_calls_zero_times(self):
        identity = identity_for(np.zeros(FORMAL_STATE_DIM, dtype=np.float32))
        calls = {"n": 0}

        def generator():
            calls["n"] += 1
            return sample_prior(), sample_metadata()

        with tempfile.TemporaryDirectory() as temp:
            cache = PriorCache(temp)
            first = cache.get_or_generate(identity, generator)
            second = cache.get_or_generate(identity, generator)
            self.assertFalse(first.cache_hit)
            self.assertTrue(second.cache_hit)
            self.assertEqual(calls["n"], 1)
            np.testing.assert_array_equal(
                first.cached_prior.prior.plans,
                second.cached_prior.prior.plans,
            )

    def test_cache_rejects_sensitive_metadata(self):
        identity = identity_for(
            np.zeros(FORMAL_STATE_DIM, dtype=np.float32),
            split="validation",
        )
        metadata = sample_metadata()
        metadata["api_key"] = "secret"
        with tempfile.TemporaryDirectory() as temp:
            cache = PriorCache(temp)
            with self.assertRaises(ValueError):
                cache.store(identity, sample_prior(), metadata=metadata)

    def test_cache_rejects_failed_provider_transaction(self):
        identity = identity_for(np.zeros(FORMAL_STATE_DIM, dtype=np.float32))
        metadata = sample_metadata()
        metadata["api_success"] = False
        with tempfile.TemporaryDirectory() as temp:
            cache = PriorCache(temp)
            with self.assertRaises(ValueError):
                cache.store(identity, sample_prior(), metadata=metadata)
            self.assertFalse(cache.path_for(identity).exists())

    def test_corrupt_cache_is_quarantined_and_miss_recovers_once(self):
        identity = identity_for(
            np.zeros(FORMAL_STATE_DIM, dtype=np.float32),
            split="test",
        )
        calls = {"n": 0}

        def generator():
            calls["n"] += 1
            return sample_prior(), sample_metadata()

        with tempfile.TemporaryDirectory() as temp:
            cache = PriorCache(temp)
            path = cache.path_for(identity)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{not-json", encoding="utf-8")
            with self.assertRaises(PriorCacheCorruptError):
                cache.load(identity)
            quarantined = list(path.parent.glob(path.name + ".corrupt.*"))
            self.assertEqual(len(quarantined), 1)
            result = cache.get_or_generate(identity, generator)
            self.assertFalse(result.cache_hit)
            self.assertEqual(calls["n"], 1)
            self.assertTrue(path.exists())

    def test_atomic_store_checksum_roundtrip_and_no_temp_or_lock(self):
        identity = identity_for(np.zeros(FORMAL_STATE_DIM, dtype=np.float32))
        with tempfile.TemporaryDirectory() as temp:
            cache = PriorCache(temp)
            cached = cache.store(identity, sample_prior(), metadata=sample_metadata())
            self.assertTrue(cached.path.exists())
            self.assertEqual(list(cached.path.parent.glob("*.tmp")), [])
            self.assertEqual(list(cached.path.parent.glob(".*.tmp")), [])
            self.assertFalse(cache.lock_path_for(identity).exists())
            payload = json.loads(cached.path.read_text(encoding="utf-8"))
            self.assertEqual(payload["identity"]["split"], "train")
            self.assertEqual(len(payload["entry_sha256"]), 64)
            self.assertNotIn("api_key", json.dumps(payload).lower())
            np.testing.assert_array_equal(cached.prior.plans, sample_prior().plans)


if __name__ == "__main__":
    unittest.main()
