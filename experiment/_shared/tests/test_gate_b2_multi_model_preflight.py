import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from formal_experiments.evaluation.run_b2_multi_model_preflight import (
    classify_preflight_exception,
    run_model_preflight,
    validate_raw_prior_contract,
)
from formal_experiments.ours.model_registry import load_model_registry
from shared.formal_state import FORMAL_STATE_DIM


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "configs" / "llm_model_registry_v1.yaml"


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


def response(text=None, model="openai/gpt-5.6-sol"):
    return SimpleNamespace(
        model=model,
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


def probe():
    state = np.zeros(FORMAL_STATE_DIM, dtype=np.float32)
    return {
        "state": state.tolist(),
        "state_sha256": "0" * 64,
        "episode_seed": 2000,
        "agent_name": "blue_agent_0",
    }


class TestGateB2MultiModelPreflight(unittest.TestCase):
    def setUp(self):
        self.registry = load_model_registry(REGISTRY_PATH)
        self.spec = self.registry.get("tier_h")

    def test_raw_contract_accepts_exact_six_valid_candidates(self):
        result = validate_raw_prior_contract(valid_text(), k=6, h=4)
        self.assertTrue(result["json_valid"])
        self.assertTrue(result["schema_valid"])
        self.assertEqual(result["semantic_valid_candidate_count"], 6)
        self.assertEqual(result["duplicate_candidate_count"], 0)

    def test_raw_contract_rejects_malformed_json_before_fallback(self):
        result = validate_raw_prior_contract("not-json", k=6, h=4)
        self.assertFalse(result["json_valid"])
        self.assertFalse(result["schema_valid"])

    def test_raw_contract_rejects_wrong_count_and_invalid_action(self):
        obj = json.loads(valid_text())
        obj["candidates"] = obj["candidates"][:5]
        self.assertFalse(validate_raw_prior_contract(json.dumps(obj), k=6, h=4)["schema_valid"])
        obj = json.loads(valid_text())
        obj["candidates"][0]["actions"][0] = "control_traffic"
        self.assertFalse(validate_raw_prior_contract(json.dumps(obj), k=6, h=4)["schema_valid"])

    def test_raw_contract_rejects_additional_properties(self):
        obj = json.loads(valid_text())
        obj["extra"] = 1
        self.assertFalse(validate_raw_prior_contract(json.dumps(obj), k=6, h=4)["schema_valid"])
        obj = json.loads(valid_text())
        obj["candidates"][0]["extra"] = "not allowed"
        self.assertFalse(validate_raw_prior_contract(json.dumps(obj), k=6, h=4)["schema_valid"])

    def test_exception_classification(self):
        self.assertEqual(classify_preflight_exception(RuntimeError("429 rate limit")), "rate_limit")
        self.assertEqual(classify_preflight_exception(RuntimeError("503 service unavailable")), "server_5xx")
        self.assertEqual(classify_preflight_exception(RuntimeError("SSL ConnectError")), "network_transient")
        self.assertEqual(classify_preflight_exception(RuntimeError("401 unauthorized")), "auth")
        self.assertEqual(classify_preflight_exception(RuntimeError("404 model not found")), "model_not_found")
        self.assertEqual(classify_preflight_exception(RuntimeError("400 unsupported response_format")), "invalid_request")

    def test_success_uses_registry_model_temp_schema_and_usage(self):
        client = FakeClient([response()])
        result = run_model_preflight(
            spec=self.spec,
            registry=self.registry,
            probe=probe(),
            client=client,
            sleep_fn=lambda _: None,
        )
        self.assertTrue(result["pass"])
        self.assertTrue(result["structured_output_verified"])
        self.assertTrue(result["temperature_parameter_accepted"])
        self.assertEqual(result["fallback_count"], 0)
        self.assertEqual(result["usage"]["total_tokens"], 150)
        call = client.chat.completions.calls[0]
        self.assertEqual(call["model"], self.spec.exact_model_id)
        self.assertEqual(call["temperature"], 0.2)
        self.assertEqual(call["response_format"]["type"], "json_schema")

    def test_retryable_failure_retries_then_succeeds(self):
        client = FakeClient([RuntimeError("503 service unavailable"), response()])
        result = run_model_preflight(
            spec=self.spec,
            registry=self.registry,
            probe=probe(),
            client=client,
            sleep_fn=lambda _: None,
        )
        self.assertTrue(result["pass"])
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(result["retry_count"], 1)

    def test_nonretryable_invalid_request_stops_immediately(self):
        client = FakeClient([RuntimeError("400 unsupported response_format")])
        result = run_model_preflight(
            spec=self.spec,
            registry=self.registry,
            probe=probe(),
            client=client,
            sleep_fn=lambda _: None,
        )
        self.assertFalse(result["pass"])
        self.assertEqual(result["failure_class"], "invalid_request")
        self.assertEqual(result["attempts"], 1)

    def test_parser_fallback_cannot_mask_raw_contract_failure(self):
        client = FakeClient([response(text="not-json")])
        result = run_model_preflight(
            spec=self.spec,
            registry=self.registry,
            probe=probe(),
            client=client,
            sleep_fn=lambda _: None,
        )
        self.assertTrue(result["api_success"])
        self.assertFalse(result["structured_output_verified"])
        self.assertTrue(result["final_prior_valid"])
        self.assertEqual(result["fallback_count"], 6)
        self.assertFalse(result["pass"])


if __name__ == "__main__":
    unittest.main()
