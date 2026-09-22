import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from formal_experiments.evaluation.generate_paper_figures import discover_formal_runs
from formal_experiments.evaluation.metrics_v3 import aggregate_repeat
from formal_experiments.evaluation.true_ppo_admission import (
    AdmissionError,
    canonical_sha256,
    create_admission_bridge,
    validate_admission_directory,
)
from formal_experiments.evaluation.validate_formal_run import validate_run_directory
from tests.test_formal_manifest import episode as production_episode


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestTruePPOAdmission(unittest.TestCase):
    def make_source(self, root: Path, *, row_id: str = "rl_only") -> Path:
        source = root / "table2" / f"{row_id}_trueppo" / "repeat_1"
        source.mkdir(parents=True)
        episodes = []
        decisions = []
        for seed in range(4000, 4100):
            row = production_episode(seed)
            row["decision_count"] = 1
            episodes.append(row)
            decisions.append({"episode_seed": seed, "duration": 1})
        write_jsonl(source / "episodes.jsonl", episodes)
        write_jsonl(source / "decisions.jsonl", decisions)
        metrics = {
            "schema": "cc4_v3_true_ppo_metrics_v1",
            "episode_count": 100,
            "mean": aggregate_repeat(episodes).to_dict(),
            "std": {},
            "per_episode": [],
        }
        write_json(source / "metrics.json", metrics)
        write_jsonl(source / "training_curve.jsonl", [
            {"schema": "cc4_v3_training_curve_v1", "split": "train",
             "environment_steps": 500, "training_objective_reward": 0.0},
            {"schema": "cc4_v3_training_curve_v1", "split": "train",
             "environment_steps": 1000, "training_objective_reward": 1.0},
        ])
        for name in (
            "normalizer.json", "calibration_manifest.json", "train_summary.json",
            "validation_records.json", "journal_index.json", "prepare_test_authorization.json",
            "test_summary.json",
        ):
            write_json(source / name, {"fixture": name})
        frozen = {}
        input_names = (
            "training_curve.jsonl", "episodes.jsonl", "decisions.jsonl", "metrics.json",
            "normalizer.json", "calibration_manifest.json", "train_summary.json",
            "validation_records.json", "journal_index.json", "prepare_test_authorization.json",
        )
        eligibility = {
            "schema": "cc4_v3_true_ppo_eligibility_v1",
            "passed": True,
            "episode_count": 100,
            "expected_episode_count": 100,
            "expected_ticks": 500,
            "provider_calls": 0,
            "cache_misses": 0,
            "test_time_updates": 0,
            "metrics_recomputed": True,
            "train_seeds": list(range(1000, 1032)),
            "calibration_seeds": list(range(3000, 3008)),
            "validation_seeds": list(range(2000, 2008)),
            "test_seeds": list(range(4000, 4100)),
            "frozen_artifact_sha256_before": frozen,
            "frozen_artifact_sha256_after": frozen,
            "input_sha256": {name: sha(source / name) for name in input_names},
            "training_curve": {
                "schema": "cc4_v3_training_curve_v1", "split": "train",
                "x_field": "environment_steps", "y_field": "training_objective_reward",
                "record_count": 2, "relative_path": "training_curve.jsonl",
                "sha256": sha(source / "training_curve.jsonl"),
            },
        }
        eligibility["input_sha256"]["frozen_artifact_bundle"] = canonical_sha256(frozen)
        write_json(source / "eligibility_report.json", eligibility)
        artifact_names = (
            "training_curve.jsonl", "train_summary.json", "validation_records.json",
            "test_summary.json", "episodes.jsonl", "decisions.jsonl", "metrics.json",
            "eligibility_report.json", "prepare_test_authorization.json", "journal_index.json",
        )
        manifest = {
            "schema": "cc4_v3_true_ppo_manifest_v1",
            "protocol": "cc4_v3_true_on_policy_clipped_ppo_v1",
            "variant": row_id,
            "reward_mode": "Full-Reward",
            "policy_seed": 51001,
            "table_id": "table2",
            "row_id": row_id,
            "test_seeds": list(range(4000, 4100)),
            "episode_ticks": 500,
            "code_commit": "a" * 40,
            "git_dirty": False,
            "formal_result_eligible": True,
            "artifact_sha256": {name: sha(source / name) for name in artifact_names},
            "metrics_sha256": sha(source / "metrics.json"),
            "eligibility_report_sha256": sha(source / "eligibility_report.json"),
            "figure_artifacts": {"training_curve": {
                **eligibility["training_curve"],
            }},
        }
        write_json(source / "manifest.json", manifest)
        write_json(source / "aggregate.json", {
            "schema": "cc4_v3_true_ppo_aggregate_v1",
            "status": "PASS",
            "source_manifest_sha256": sha(source / "manifest.json"),
            "source_eligibility_sha256": sha(source / "eligibility_report.json"),
        })
        write_json(source / ".true_ppo_writer.lock.json", {"status": "RELEASED", "pid": 123})
        return source

    def test_bridge_is_non_destructive_provisional_and_discoverable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "formal_v3"
            source = self.make_source(root)
            original = (source / "eligibility_report.json").read_bytes()
            destination = root / "admission_derived" / "table2" / "rl_only" / "repeat_1"
            result = create_admission_bridge(source, destination)
            self.assertTrue(result["passed"])
            self.assertEqual((source / "eligibility_report.json").read_bytes(), original)
            self.assertEqual({item.name for item in destination.iterdir()}, {"manifest.json", "admission_report.json"})
            bridge = json.loads((destination / "manifest.json").read_text())
            self.assertEqual(bridge["status"], "PROVISIONAL")
            self.assertFalse(bridge["paper_row_eligible"])
            accepted, rejected = discover_formal_runs(root)
            self.assertEqual(len(accepted), 1)
            self.assertEqual(accepted[0]["manifest"]["repeat_index"], 1)
            self.assertTrue(any(row.get("source_only") for row in rejected))

    def test_generic_validator_never_overwrites_raw_true_ppo_eligibility(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = self.make_source(Path(temporary) / "formal_v3")
            original = (source / "eligibility_report.json").read_bytes()
            result = validate_run_directory(source, formal=True)
            self.assertFalse(result["passed"])
            self.assertIn("read-only", result["errors"][0])
            self.assertEqual((source / "eligibility_report.json").read_bytes(), original)

    def test_source_tamper_invalidates_existing_bridge(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "formal_v3"
            source = self.make_source(root)
            destination = root / "admission_derived" / "table2" / "rl_only" / "repeat_1"
            create_admission_bridge(source, destination)
            with (source / "metrics.json").open("a", encoding="utf-8") as handle:
                handle.write(" ")
            with self.assertRaises(AdmissionError):
                validate_admission_directory(destination)

    def test_active_source_and_destination_overwrite_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "formal_v3"
            source = self.make_source(root)
            destination = root / "admission_derived" / "table2" / "rl_only" / "repeat_1"
            write_json(source / ".true_ppo_writer.lock.json", {"status": "ACTIVE", "pid": 123})
            with self.assertRaises(AdmissionError):
                create_admission_bridge(source, destination)
            self.assertFalse(destination.exists())
            write_json(source / ".true_ppo_writer.lock.json", {"status": "RELEASED", "pid": 123})
            create_admission_bridge(source, destination)
            with self.assertRaises(AdmissionError):
                create_admission_bridge(source, destination)


if __name__ == "__main__":
    unittest.main()
