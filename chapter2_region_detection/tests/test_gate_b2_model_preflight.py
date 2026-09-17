import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from formal_experiments.evaluation.build_llm_validation_state_bank import state_sha256
from formal_experiments.evaluation.run_gate_b2_model_preflight import (
    EXPECTED_BANK_COUNT,
    estimate_cost_usd,
    extract_usage,
    inspect_raw_prior_response,
    load_frozen_state_bank,
    run_model_preflight,
    run_preflight,
    select_preflight_state,
)
from formal_experiments.ours.model_registry import load_model_registry
from shared.formal_state import FORMAL_STATE_DIM


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "configs" / "llm_model_registry_v1.yaml"


def valid_payload(*, duplicate=False):
    candidates = [
        {"actions": ["no_op", "analyse", "remove", "restore"], "prior_score": 0.90, "reason": "a"},
        {"actions": ["analyse", "analyse", "remove", "restore"], "prior_score": 0.80, "reason": "b"},
        {"actions": ["remove", "analyse", "remove", "restore"], "prior_score": 0.70, "reason": "c"},
        {"actions": ["restore", "analyse", "remove", "restore"], "prior_score": 0.60, "reason": "d"},
        {"actions": ["no_op", "no_op", "no_op", "no_op"], "prior_score": 0.50, "reason": "e"},
        {"actions": ["restore", "restore", "restore", "restore"], "prior_score": 0.40, "reason": "f"},
    ]
    if duplicate:
        candidates[-1] = dict(candidates[0])
        candidates[-1]["prior_score"] = 0.30
        candidates[-1]["reason"] = "duplicate"
    return {"candidates": candidates}


def response_for(model, *, payload=None, usage=True):
    if payload is None:
        payload = valid_payload()
    usage_obj = None
    if usage:
        usage_obj = SimpleNamespace(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    return SimpleNamespace(
        model=f"served::{model}",
        usage=usage_obj,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=json.dumps(payload)),
                finish_reason="stop",
            )
        ],
    )


class FakeCompletions:
    def __init__(self, responder):
        self.responder = responder
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responder(kwargs)


class FakeClient:
    def __init__(self, responder):
        self.chat = SimpleNamespace(completions=FakeCompletions(responder))


def write_bank(root: Path):
    selection_hash = "a" * 64
    rows = []
    for index in range(EXPECTED_BANK_COUNT):
        state = np.zeros(FORMAL_STATE_DIM, dtype=np.float32)
        state[0] = np.float32(index / 1000.0)
        state[1] = np.float32((index + 1) / 2000.0)
        row = {
            "bank_format_version": 1,
            "split": "validation",
            "source_replay": "outputs/formal_replay_v2/validation.jsonl",
            "selection_config_sha256": selection_hash,
            "episode_seed": 2000 + (index // 30),
            "agent_name": f"blue_agent_{(index // 6) % 5}",
            "decision_index": index % 6,
            "global_tick_start": index,
            "feature17_any_valid_observable_target": index % 2,
            "state_dtype": "float32_le",
            "state_sha256": state_sha256(state),
            "state": [float(x) for x in state.tolist()],
        }
        rows.append(row)

    bank_path = root / "state_bank.jsonl"
    canonical_lines = [json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) for row in rows]
    bank_hash = __import__("hashlib").sha256(("\n".join(canonical_lines) + "\n").encode("utf-8")).hexdigest()
    with bank_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    summary = {
        "status": "PASS",
        "pass": True,
        "split": "validation",
        "record_count": EXPECTED_BANK_COUNT,
        "unique_exact_state_count": EXPECTED_BANK_COUNT,
        "selection_config_sha256": selection_hash,
        "bank_sha256": bank_hash,
    }
    summary_path = root / "state_bank_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return bank_path, summary_path, rows, summary


class TestGateB2ModelPreflight(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = load_model_registry(REGISTRY_PATH)

    def test_raw_inspection_accepts_exact_schema(self):
        report = inspect_raw_prior_response(json.dumps(valid_payload()))
        self.assertTrue(report["json_object_valid"])
        self.assertTrue(report["schema_valid"])
        self.assertEqual(report["candidate_count"], 6)
        self.assertEqual(report["semantic_valid_candidate_count"], 6)
        self.assertEqual(report["unique_valid_plan_count"], 6)
        self.assertEqual(report["duplicate_valid_plan_count"], 0)

    def test_raw_inspection_rejects_unknown_action_even_if_parser_can_fallback(self):
        payload = valid_payload()
        payload["candidates"][0]["actions"][-1] = "control_traffic"
        report = inspect_raw_prior_response(json.dumps(payload))
        self.assertFalse(report["schema_valid"])
        self.assertEqual(report["semantic_valid_candidate_count"], 5)

    def test_duplicate_schema_response_is_diagnostic_not_transport_failure(self):
        state_record = {
            "agent_name": "blue_agent_0",
            "state_sha256": state_sha256(np.zeros(27, dtype=np.float32)),
            "state": np.zeros(27, dtype=np.float32).tolist(),
        }
        client = FakeClient(lambda kwargs: response_for(kwargs["model"], payload=valid_payload(duplicate=True)))
        report = run_model_preflight(
            registry=self.registry,
            tier="tier_h",
            state_record=state_record,
            client=client,
        )
        self.assertTrue(report["api_success"])
        self.assertTrue(report["raw_inspection"]["schema_valid"])
        self.assertEqual(report["raw_inspection"]["duplicate_valid_plan_count"], 1)
        self.assertEqual(report["fallback_count"], 1)
        self.assertTrue(report["final_prior_valid"])
        self.assertTrue(report["pass"])

    def test_usage_and_cost_are_extracted_from_same_live_response(self):
        response = response_for("openai/gpt-5.6-sol")
        usage = extract_usage(response)
        self.assertEqual(usage["input_tokens"], 100)
        self.assertEqual(usage["output_tokens"], 50)
        self.assertEqual(usage["total_tokens"], 150)
        self.assertTrue(usage["available"])
        cost = estimate_cost_usd(self.registry.get("tier_h"), usage)
        self.assertAlmostEqual(cost, (100 * 5.0 + 50 * 30.0) / 1_000_000.0)

    def test_state_bank_integrity_and_preflight_state_are_deterministic(self):
        with tempfile.TemporaryDirectory() as temp:
            bank, summary, rows, _ = write_bank(Path(temp))
            loaded, loaded_summary = load_frozen_state_bank(bank, summary)
            self.assertEqual(len(loaded), 240)
            self.assertEqual(loaded_summary["record_count"], 240)
            first = select_preflight_state(loaded)
            second = select_preflight_state(list(reversed(loaded)))
            self.assertEqual(first, second)
            self.assertEqual(first["episode_seed"], 2000)
            self.assertEqual(first["agent_name"], "blue_agent_0")

    def test_all_three_models_use_same_state_exact_ids_temperature_and_json_schema(self):
        fake = FakeClient(lambda kwargs: response_for(kwargs["model"]))
        with tempfile.TemporaryDirectory() as temp:
            bank, summary, _, _ = write_bank(Path(temp))
            report = run_preflight(
                registry_path=REGISTRY_PATH,
                bank_path=bank,
                summary_path=summary,
                client_factory=lambda registry: fake,
                require_env_key=False,
            )
        self.assertTrue(report["pass"])
        self.assertEqual(report["live_call_count"], 3)
        self.assertEqual(len(fake.chat.completions.calls), 3)
        requested = [call["model"] for call in fake.chat.completions.calls]
        self.assertEqual(
            requested,
            [
                "openai/gpt-5.6-sol",
                "openai/gpt-5.4-mini",
                "google/gemini-3.5-flash-lite",
            ],
        )
        for call in fake.chat.completions.calls:
            self.assertEqual(call["temperature"], 0.2)
            self.assertEqual(call["response_format"]["type"], "json_schema")
        self.assertEqual(len({item["state_sha256"] for item in report["models"]}), 1)
        self.assertFalse(report["primary_model_selected"])

    def test_missing_usage_metadata_fails_capability_preflight(self):
        state = np.zeros(27, dtype=np.float32)
        state_record = {
            "agent_name": "blue_agent_0",
            "state_sha256": state_sha256(state),
            "state": state.tolist(),
        }
        fake = FakeClient(lambda kwargs: response_for(kwargs["model"], usage=False))
        report = run_model_preflight(
            registry=self.registry,
            tier="tier_m",
            state_record=state_record,
            client=fake,
        )
        self.assertTrue(report["api_success"])
        self.assertTrue(report["raw_inspection"]["schema_valid"])
        self.assertFalse(report["usage"]["available"])
        self.assertFalse(report["pass"])

    def test_invalid_raw_schema_fails_even_when_final_prior_is_fallback_valid(self):
        payload = valid_payload()
        payload["candidates"][0]["actions"] = ["invalid", "analyse", "remove", "restore"]
        state = np.zeros(27, dtype=np.float32)
        state_record = {
            "agent_name": "blue_agent_0",
            "state_sha256": state_sha256(state),
            "state": state.tolist(),
        }
        fake = FakeClient(lambda kwargs: response_for(kwargs["model"], payload=payload))
        report = run_model_preflight(
            registry=self.registry,
            tier="tier_l",
            state_record=state_record,
            client=fake,
        )
        self.assertTrue(report["api_success"])
        self.assertFalse(report["raw_inspection"]["schema_valid"])
        self.assertTrue(report["final_prior_valid"])
        self.assertEqual(report["fallback_count"], 1)
        self.assertFalse(report["pass"])

    def test_provider_exception_records_only_safe_class_and_type(self):
        class SecretFailure(RuntimeError):
            pass

        def fail(kwargs):
            raise SecretFailure("429 secret-key-value should never enter report")

        state = np.zeros(27, dtype=np.float32)
        state_record = {
            "agent_name": "blue_agent_0",
            "state_sha256": state_sha256(state),
            "state": state.tolist(),
        }
        report = run_model_preflight(
            registry=self.registry,
            tier="tier_h",
            state_record=state_record,
            client=FakeClient(fail),
        )
        self.assertFalse(report["api_success"])
        self.assertEqual(report["failure_class"], "quota_blocked")
        self.assertEqual(report["exception_type"], "SecretFailure")
        dumped = json.dumps(report)
        self.assertNotIn("secret-key-value", dumped)
        self.assertFalse(report["pass"])

    def test_report_never_records_raw_prompt_response_or_key_value(self):
        fake = FakeClient(lambda kwargs: response_for(kwargs["model"]))
        with tempfile.TemporaryDirectory() as temp:
            bank, summary, _, _ = write_bank(Path(temp))
            report = run_preflight(
                registry_path=REGISTRY_PATH,
                bank_path=bank,
                summary_path=summary,
                client_factory=lambda registry: fake,
                require_env_key=False,
            )
        self.assertFalse(report["raw_prompt_recorded"])
        self.assertFalse(report["raw_response_recorded"])
        self.assertFalse(report["api_key_value_recorded"])
        self.assertFalse(report["cache_used"])
        dumped = json.dumps(report).lower()
        self.assertNotIn("api_key", dumped.replace("api_key_value_recorded", ""))
        self.assertNotIn("candidates\":", dumped)


if __name__ == "__main__":
    unittest.main()
