import hashlib
import json
import tempfile
import threading
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from formal_experiments.evaluation.build_final_prior_cache import (
    StateBankRecord, build_cache, coverage, deterministic_final_manifest_sha256, load_frozen_registry,
    load_primary_model_selection, run_probe,
    select_representative_states, validate_live_result,
)
from formal_experiments.ours.llm_prior_v2 import PriorBatch
from shared.formal_state import BLUE_AGENTS, FORMAL_STATE_DIM


def record(index, agent, action, evidence):
    state = np.zeros(FORMAL_STATE_DIM, dtype=np.float32)
    state[0], state[17] = index / 1000.0, evidence
    digest = hashlib.sha256(np.asarray(state, dtype="<f4").tobytes()).hexdigest()
    return StateBankRecord(index, 1000 + index, agent, action, evidence, state, digest)


def sample_prior():
    plans = np.asarray([[i % 4] * 4 for i in range(6)], dtype=np.int64)
    scores = np.asarray([.9, .8, .7, .6, .5, .4], dtype=np.float32)
    return PriorBatch(plans, scores, scores / scores.sum(), ("llm",) * 6)


def sample_metadata():
    return {"api_success": True, "fallback_count": 0,
            "prompt_sha256": "1" * 64, "response_sha256": "2" * 64,
            "latency_ms": 1.0, "retry_count": 0}


class TestBuildFinalPriorCache(unittest.TestCase):
    def test_registry_mismatch_fails_closed(self):
        registry = Path(__file__).resolve().parents[1] / "configs" / "llm_model_registry_v1.yaml"
        payload = yaml.safe_load(registry.read_text(encoding="utf-8"))
        payload["models"]["tier_h"]["exact_model_id"] = "wrong/model"
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "registry.yaml"
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact_model_id"):
                load_frozen_registry(path)

    def test_frozen_registry_matches_builder(self):
        registry = Path(__file__).resolve().parents[1] / "configs" / "llm_model_registry_v1.yaml"
        payload, digest = load_frozen_registry(registry)
        self.assertEqual(payload["registry"]["version"], 1)
        self.assertEqual(len(digest), 64)

    def test_primary_selection_matches_registry_and_mismatch_fails_closed(self):
        root = Path(__file__).resolve().parents[1]
        _, registry_sha = load_frozen_registry(root / "configs" / "llm_model_registry_v1.yaml")
        payload, digest = load_primary_model_selection(
            root / "configs" / "formal_v3" / "primary_model_selection.yaml",
            registry_sha256=registry_sha)
        self.assertTrue(payload["selection"]["selected"])
        self.assertEqual(len(digest), 64)
        payload["selection"]["selected"] = False
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "selection.yaml"
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "selected"):
                load_primary_model_selection(path, registry_sha256=registry_sha)

    def test_live_result_model_or_temperature_mismatch_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "unexpected model"):
            validate_live_result(SimpleNamespace(model="other/model", temperature=0.2))
        with self.assertRaisesRegex(RuntimeError, "temperature"):
            validate_live_result(SimpleNamespace(model="openai/gpt-5.6-sol", temperature=0.3))
        validate_live_result(SimpleNamespace(model="openai/gpt-5.6-sol", temperature=0.2))

    def test_selection_is_deterministic_and_balanced(self):
        rows = [record(i * 100 + action * 10 + evidence * 3 + j, agent, action, evidence)
                for i, agent in enumerate(BLUE_AGENTS) for action in range(4)
                for evidence in range(2) for j in range(3)]
        first = select_representative_states(rows, 80)
        second = select_representative_states(reversed(rows), 80)
        self.assertEqual([x.state_sha256 for x in first], [x.state_sha256 for x in second])
        self.assertEqual(set(coverage(first)["per_agent"]), set(BLUE_AGENTS))
        self.assertEqual(set(coverage(first)["per_action"]), {"0", "1", "2", "3"})

    def test_cache_uses_agent_identity_checksum_and_resume(self):
        rows = [record(1, "blue_agent_0", 0, 0), record(2, "blue_agent_1", 1, 1)]
        calls = []
        def generate(state, agent_name):
            calls.append(agent_name)
            return sample_prior(), sample_metadata()
        with tempfile.TemporaryDirectory() as temp:
            first = build_cache(selected=rows, output_root=temp,
                                registry_sha256="a" * 64, generator=generate)
            second = build_cache(selected=rows, output_root=temp,
                                 registry_sha256="a" * 64,
                                 generator=lambda *_: self.fail("provider called on resume"))
            self.assertEqual(first["generated"], 2)
            self.assertEqual(second["cache_hits"], 2)
            self.assertEqual(calls, ["blue_agent_0", "blue_agent_1"])
            payloads = [json.loads(path.read_text()) for path in Path(temp).rglob("*.json")]
            self.assertEqual({x["identity"]["agent_name"] for x in payloads},
                             {"blue_agent_0", "blue_agent_1"})
            self.assertTrue(all(len(x["entry_sha256"]) == 64 for x in payloads))

    def test_insufficient_bank_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "insufficient unique train states"):
            select_representative_states([record(1, "blue_agent_0", 0, 0)], 2)

    def test_duplicate_cache_key_fails_closed_before_generation(self):
        row = record(1, "blue_agent_0", 0, 0)
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "unique cache keys"):
                build_cache(selected=[row, row], output_root=temp,
                            registry_sha256="a"*64,
                            generator=lambda *_: self.fail("provider called"))

    def test_bounded_workers_create_thread_local_generators(self):
        rows = [record(10, "blue_agent_0", 0, 0), record(11, "blue_agent_1", 1, 1)]
        barrier = threading.Barrier(2); created = []; lock = threading.Lock()
        def factory():
            with lock:
                created.append(object())
            def generate(*_):
                barrier.wait(timeout=5)
                return sample_prior(), sample_metadata()
            return generate
        with tempfile.TemporaryDirectory() as temp:
            result = build_cache(selected=rows, output_root=temp, registry_sha256="a"*64,
                                 generator_factory=factory, workers=2, retry_backoff_seconds=0)
        self.assertEqual(result["generated"], 2)
        self.assertEqual(len(created), 2)

    def test_retryable_three_attempts_nonretryable_partial_and_resume(self):
        rows = [record(20, "blue_agent_0", 0, 0), record(21, "blue_agent_1", 1, 1)]
        calls = Counter()
        def classify(exc): return str(exc)
        def first(state, agent):
            calls[agent] += 1
            if agent == "blue_agent_0" and calls[agent] < 3:
                raise RuntimeError("network_blocked")
            if agent == "blue_agent_1":
                raise RuntimeError("auth_blocked")
            return sample_prior(), sample_metadata()
        with tempfile.TemporaryDirectory() as temp:
            progress = Path(temp) / "progress.jsonl"
            partial = build_cache(selected=rows, output_root=temp, registry_sha256="a"*64,
                                  generator=first, max_attempts=3, retry_backoff_seconds=0,
                                  classify_exception=classify, sleep_fn=lambda _: None,
                                  progress_path=progress)
            self.assertEqual((partial["verified_entries"], partial["failed_entries"]), (1, 1))
            self.assertEqual(calls["blue_agent_0"], 3)
            self.assertEqual(calls["blue_agent_1"], 1)
            resumed = build_cache(selected=rows, output_root=temp, registry_sha256="a"*64,
                                  generator=lambda *_: (sample_prior(), sample_metadata()),
                                  classify_exception=classify, retry_backoff_seconds=0)
            self.assertEqual((resumed["cache_hits"], resumed["generated"], resumed["failed_entries"]), (1, 1, 0))
            events = [json.loads(x) for x in progress.read_text().splitlines()]
            self.assertFalse(any("secret" in json.dumps(x).lower() for x in events))

    def test_verified_resume_keeps_deterministic_entry_set_manifest(self):
        rows = [record(30, "blue_agent_0", 0, 0), record(31, "blue_agent_1", 1, 1)]
        with tempfile.TemporaryDirectory() as temp:
            first = build_cache(selected=rows, output_root=temp, registry_sha256="a"*64,
                                generator=lambda *_: (sample_prior(), sample_metadata()), workers=2)
            second = build_cache(selected=list(reversed(rows)), output_root=temp, registry_sha256="a"*64,
                                 generator=lambda *_: self.fail("resume generated"), workers=1)
            self.assertEqual(first["entry_sha256_set_sha256"], second["entry_sha256_set_sha256"])
            first_hash = deterministic_final_manifest_sha256(
                target_size=2, registry_sha256="a"*64, selection_sha256="b"*64,
                primary_model_selection_sha256="c"*64,
                result=first, status="PASS")
            second_hash = deterministic_final_manifest_sha256(
                target_size=2, registry_sha256="a"*64, selection_sha256="b"*64,
                primary_model_selection_sha256="c"*64,
                result=second, status="PASS")
            self.assertEqual(first_hash, second_hash)

    def test_probe_success_reload_and_model_drift_fail_closed(self):
        rows = [record(1000 + i, BLUE_AGENTS[i % len(BLUE_AGENTS)], i % 4, i % 2)
                for i in range(2000)]
        good = sample_metadata() | {
            "actual_model": "openai/gpt-5.6-sol", "actual_temperature": 0.2,
            "structured_response_verified": True,
        }
        with tempfile.TemporaryDirectory() as temp:
            report = run_probe(
                selected=rows, output_root=temp, registry_sha256="a"*64,
                primary_selection_sha256="b"*64, state_selection_sha256="c"*64,
                generator_factory=lambda: (lambda *_: (sample_prior(), good)),
                retry_backoff_seconds=0)
            self.assertEqual(report["status"], "PROBE_PASS")
            self.assertTrue((Path(temp) / "probe_manifest.json").is_file())
            self.assertFalse((Path(temp) / "manifest.json").exists())
        bad = good | {"actual_model": "silent/model-drift"}
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(RuntimeError, "did not produce"):
                run_probe(
                    selected=rows, output_root=temp, registry_sha256="a"*64,
                    primary_selection_sha256="b"*64, state_selection_sha256="c"*64,
                    generator_factory=lambda: (lambda *_: (sample_prior(), bad)),
                    max_attempts=1, retry_backoff_seconds=0)

    def test_probe_and_execute_live_are_mutually_exclusive(self):
        from unittest.mock import patch
        with patch("sys.stderr"):
            with self.assertRaises(SystemExit) as raised:
                from formal_experiments.evaluation.build_final_prior_cache import main
                main(["--probe-live", "--execute-live"])
        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
