import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from formal_experiments.common.run_manifest import (
    FINAL_TEST_SEEDS,
    POLICY_SEEDS,
    validate_episode_schema,
    validate_manifest,
)
from formal_experiments.evaluation.validate_formal_run import validate_run_directory
from formal_experiments.evaluation.metrics_v3 import aggregate_repeat


def manifest():
    return {
        "schema_version": 1,
        "protocol_version": "cc4_v3_20260917",
        "method": "golden-test",
        "repeat_index": 1,
        "training_seed": POLICY_SEEDS[0],
        "test_episode_seeds": list(FINAL_TEST_SEEDS),
        "episode_ticks": 500,
        "code_commit": "abc123",
        "config_sha256": "a" * 64,
        "paper_sha256": "b" * 64,
        "upstream_commit": None,
        "python_version": "3.test",
        "dependencies": {"python": "test", "torch": "test", "numpy": "test",
                         "CybORG": {"source_sha256": "d" * 64}},
        "hardware": {"cpu": "test", "gpu": [], "selected_device": "cpu"},
        "model_sha256": None,
        "run_mode": "formal",
        "formal_result_eligible": True,
        "method_slug": "golden-test",
        "git_dirty": False,
        "git_diff_sha256": "c" * 64,
        "method_artifacts": {},
        "artifact_sha256": {
            "config.resolved.yaml": "a" * 64,
            "policy_spec.json": "a" * 64,
            "episodes.jsonl": "a" * 64,
            "decisions.jsonl.zst": "a" * 64,
            "metrics.json": "a" * 64,
        },
    }


def episode(seed):
    return {
        "schema_version": 1,
        "protocol_version": "cc4_v3_20260917",
        "episode_seed": seed,
        "tick_count": 500,
        "episode_end_tick": 500,
        "tick_team_rewards": [0.0] * 500,
        "operation_failure_events": [],
        "recovery_actions": [
            {"executed_action": "remove", "started": True, "completed": True,
             "fallback": False, "active_incident_before": True}
        ],
        "incidents": [{"t_compromise": 100, "t_recovered": 120}],
    }


def bind_artifacts(root: Path, value: dict) -> None:
    value["artifact_sha256"] = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ("config.resolved.yaml", "policy_spec.json", "episodes.jsonl",
                     "decisions.jsonl.zst", "metrics.json")
    }
    value["config_sha256"] = value["artifact_sha256"]["config.resolved.yaml"]
    value["model_sha256"] = value["artifact_sha256"]["policy_spec.json"]
    (root / "manifest.json").write_text(json.dumps(value), encoding="utf-8")


class TestFormalManifest(unittest.TestCase):
    def test_golden_manifest_and_episode_schema(self):
        self.assertEqual(validate_manifest(manifest()), [])
        self.assertEqual(validate_episode_schema(episode(4000)), [])

    def test_manifest_rejects_short_test_seed_set_and_bad_hash(self):
        value = manifest()
        value["test_episode_seeds"] = [4000]
        value["config_sha256"] = "not-a-hash"
        errors = validate_manifest(value)
        self.assertTrue(any("4000..4099" in error for error in errors))
        self.assertTrue(any("config_sha256" in error for error in errors))

    def test_manifest_rejects_repeat_training_seed_mismatch(self):
        value = manifest()
        value["repeat_index"] = 2
        value["training_seed"] = POLICY_SEEDS[0]
        errors = validate_manifest(value)
        self.assertTrue(any("repeat 2" in error and "51002" in error for error in errors))

    def test_uamcts_manifest_requires_complete_frozen_provenance_chain(self):
        value = manifest()
        value.update({"method": "UAMCTS-CC4 (adapted)", "method_slug": "uamcts_cc4"})
        fields = ("policy_spec_sha256", "world_model_sha256", "reward_model_sha256",
                  "progress_model_sha256", "prototype_prior_sha256",
                  "prior_entropy_sha256", "normalizer_sha256")
        value["method_artifacts"] = {field: "d" * 64 for field in fields}
        self.assertEqual(validate_manifest(value), [])
        del value["method_artifacts"]["prior_entropy_sha256"]
        self.assertTrue(any("prior_entropy_sha256" in error
                            for error in validate_manifest(value)))

    def test_learned_baseline_manifests_require_training_provenance_files(self):
        cases = (
            ("terla_a4", "TERLA-A4", "training_manifest_sha256", "training_manifest.json"),
            ("carl_cc4", "CARL-CC4 (adapted)", "validation_selection_sha256",
             "validation_selection.json"),
        )
        for slug, name, provenance_field, bound_name in cases:
            with self.subTest(method=slug):
                value = manifest()
                value.update({"method": name, "method_slug": slug})
                value["method_artifacts"] = {
                    "policy_spec_sha256": "d" * 64,
                    "checkpoint_sha256": "e" * 64,
                    provenance_field: "f" * 64,
                }
                value["artifact_sha256"]["checkpoint.pt"] = "e" * 64
                value["artifact_sha256"][bound_name] = "f" * 64
                self.assertEqual(validate_manifest(value), [])
                del value["artifact_sha256"][bound_name]
                self.assertTrue(any(bound_name in error for error in validate_manifest(value)))

    def test_failure_event_requires_auditable_blue_scope(self):
        value = episode(4000)
        value["operation_failure_events"] = [{"raw_penalty": -1.0}]
        errors = validate_episode_schema(value)
        self.assertTrue(any("agent_name+host" in error for error in errors))
        value["operation_failure_events"] = [{"raw_penalty": -1.0, "scope": "all_blue"}]
        self.assertEqual(validate_episode_schema(value), [])

    def test_excluded_non_blue_failure_events_are_auxiliary(self):
        value = episode(4000)
        value["excluded_non_blue_failure_events"] = [{
            "host": "contractor_network", "tick": 10, "raw_penalty": -1.0,
            "reason": "outside_blue_jurisdiction",
        }]
        self.assertEqual(validate_episode_schema(value), [])
        self.assertEqual(aggregate_repeat([value]).operation_failure_penalty, 0.0)

    def test_malformed_manifest_types_return_errors_without_crashing(self):
        for field, bad in (
            ("episode_ticks", {"bad": 500}),
            ("training_seed", [51001]),
            ("test_episode_seeds", "4000,4001"),
        ):
            value = manifest()
            value[field] = bad
            errors = validate_manifest(value)
            self.assertTrue(errors, field)

    def test_malformed_episode_tick_types_return_errors_without_crashing(self):
        for field, bad in (("tick_count", []), ("episode_end_tick", {"bad": 500})):
            value = episode(4000)
            value[field] = bad
            errors = validate_episode_schema(value)
            self.assertTrue(any(field in error for error in errors))

    def test_dev_20_tick_schema_is_ineligible_and_formal_rejects_manifest(self):
        value = episode(39000)
        value["tick_count"] = value["episode_end_tick"] = 20
        value["tick_team_rewards"] = [0.0] * 20
        self.assertEqual(validate_episode_schema(value, expected_ticks=20), [])
        dev_manifest = manifest()
        dev_manifest.update({
            "run_mode": "dev", "formal_result_eligible": False,
            "episode_ticks": 20, "training_seed": 0,
            "test_episode_seeds": [39000, 39001],
        })
        self.assertEqual(validate_manifest(dev_manifest, formal=False), [])
        self.assertTrue(validate_manifest(dev_manifest, formal=True))

    def test_formal_run_gate_recomputes_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            value = manifest()
            episodes = [episode(seed) for seed in FINAL_TEST_SEEDS]
            with (root / "episodes.jsonl").open("w", encoding="utf-8") as handle:
                for row in episodes:
                    handle.write(json.dumps(row) + "\n")
            (root / "metrics.json").write_text(
                json.dumps(aggregate_repeat(episodes).to_dict()), encoding="utf-8"
            )
            for filename in ("config.resolved.yaml", "decisions.jsonl.zst", "stdout.log", "policy_spec.json"):
                (root / filename).write_bytes(b"test")
            bind_artifacts(root, value)
            result = validate_run_directory(root)
            self.assertTrue(result["passed"], result["errors"])
            self.assertEqual(result["metrics"]["episode_count"], 100)
            self.assertEqual(result["metrics"]["recovery_precision"], 1.0)

    def test_formal_run_gate_rejects_incomplete_episode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            value = manifest()
            bad = episode(4000)
            bad["tick_count"] = 499
            (root / "episodes.jsonl").write_text(json.dumps(bad) + "\n", encoding="utf-8")
            (root / "metrics.json").write_text("{}", encoding="utf-8")
            for filename in ("config.resolved.yaml", "decisions.jsonl.zst", "policy_spec.json"):
                (root / filename).write_bytes(b"test")
            bind_artifacts(root, value)
            result = validate_run_directory(root, formal=False)
            self.assertFalse(result["passed"])
            self.assertTrue(any("expected 100" in error for error in result["errors"]))
            self.assertTrue(any("tick_count" in error for error in result["errors"]))

    def test_dev_run_allows_na_precision_with_warning(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            value = manifest()
            value.update({
                "run_mode": "dev", "formal_result_eligible": False,
                "episode_ticks": 20, "training_seed": 0, "test_episode_seeds": [39000],
            })
            row = episode(39000)
            row["tick_count"] = row["episode_end_tick"] = 20
            row["tick_team_rewards"] = [0.0] * 20
            row["recovery_actions"] = []
            row["incidents"] = []
            (root / "manifest.json").write_text(json.dumps(value), encoding="utf-8")
            (root / "episodes.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            (root / "metrics.json").write_text(
                json.dumps(aggregate_repeat([row], expected_ticks=20).to_dict()),
                encoding="utf-8",
            )
            for filename in ("config.resolved.yaml", "decisions.jsonl.zst", "policy_spec.json"):
                (root / filename).write_bytes(b"test")
            bind_artifacts(root, value)
            result = validate_run_directory(root, formal=False)
            self.assertTrue(result["passed"], result["errors"])
            self.assertTrue(any("precision is NA" in warning for warning in result["warnings"]))

    def test_formal_integrity_rejects_tampered_bound_artifacts_and_writes_fail_report(self):
        for target in ("episodes.jsonl", "metrics.json", "decisions.jsonl.zst",
                       "config.resolved.yaml", "policy_spec.json"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); value = manifest()
                episodes = [episode(seed) for seed in FINAL_TEST_SEEDS]
                (root / "episodes.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in episodes), encoding="utf-8")
                (root / "metrics.json").write_text(
                    json.dumps(aggregate_repeat(episodes).to_dict()), encoding="utf-8")
                for filename in ("config.resolved.yaml", "decisions.jsonl.zst", "stdout.log", "policy_spec.json"):
                    (root / filename).write_bytes(b"test")
                bind_artifacts(root, value)
                (root / target).write_bytes((root / target).read_bytes() + b"tamper")
                result = validate_run_directory(root)
                self.assertFalse(result["passed"])
                self.assertTrue(any("SHA256 mismatch" in error for error in result["errors"]))
                report = json.loads((root / "eligibility_report.json").read_text())
                self.assertEqual(report["status"], "FAIL")

    def test_formal_integrity_rejects_path_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"; root.mkdir(); value = manifest()
            value["artifact_sha256"]["../outside.pt"] = "a" * 64
            (root / "manifest.json").write_text(json.dumps(value), encoding="utf-8")
            (root / "episodes.jsonl").write_text("", encoding="utf-8")
            for filename in ("config.resolved.yaml", "decisions.jsonl.zst", "metrics.json", "policy_spec.json", "stdout.log"):
                (root / filename).write_bytes(b"test")
            result = validate_run_directory(root)
            self.assertFalse(result["passed"])
            self.assertTrue(any("escapes run directory" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
