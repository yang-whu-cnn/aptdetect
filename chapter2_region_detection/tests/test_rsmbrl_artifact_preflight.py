import tempfile
import unittest
from unittest import mock
from pathlib import Path
import json

import torch

from baselines.rsmbrl_cc4.artifact_preflight import (
    SIDECAR_SCHEMA, sha256_file, validate_frozen_normalizer,
    validate_normalizer_sidecar,
)
from baselines.rsmbrl_cc4.calibrate_normalizers import audit_calibration_states
from baselines.rsmbrl_cc4 import calibrate_normalizers as rebuild
from baselines.rsmbrl_cc4.run_pilot import _episode_passed
from shared.formal_state import BLUE_AGENTS
from shared.d27_projection import PROJECTION_SHA256, PROJECTION_VERSION


def valid_bundle():
    return {
        "format_version": 2, "method": "RSMBRL-CC4", "source_split": "calibration",
        "calibration_seeds": list(range(3000, 3008)), "horizon": 4, "state_dim": 27,
        "n_actions": 4, "ensemble_size": 5, "freeze_after_calibration": True,
        "online_updates_after_calibration": False, "world_model_sha256": "a" * 64,
        "projection_version": PROJECTION_VERSION, "projection_sha256": PROJECTION_SHA256,
        "beta": 0.1, "beta_source": "paper_fixed_no_validation_tuning",
        "planner_profile": {"population_size": 200, "num_iterations": 5,
                            "elite_ratio": 0.3, "alpha": 0.1, "beta": 0.1},
        "coverage": {agent: {"planner_calls": 100, "seeds": list(range(3000, 3008)),
                              "plans_per_call": 200} for agent in BLUE_AGENTS},
        "provenance": {"states_sha256": "b" * 64, "summary_sha256": "c" * 64,
                       "record_count": 2301},
        "agents": {agent: {"obs_mean": torch.zeros(27), "obs_std": torch.ones(27),
                            "horizon_std": torch.ones(4)} for agent in BLUE_AGENTS},
    }


def valid_sidecar(root: Path, normalizer_sha="a" * 64, world_sha="b" * 64,
                  reward_sha="c" * 64):
    train = root / "train.jsonl"
    validation = root / "validation.jsonl"
    train.write_text("".join(json.dumps({"episode_seed": seed}) + "\n"
                             for seed in range(1000, 1032)), encoding="utf-8")
    validation.write_text("".join(json.dumps({"episode_seed": seed}) + "\n"
                                  for seed in range(2000, 2008)), encoding="utf-8")
    return {
        "schema": SIDECAR_SCHEMA,
        "method": "RSMBRL-CC4",
        "normalizer_sha256": normalizer_sha,
        "world_model_sha256": world_sha,
        "reward_model_sha256": reward_sha,
        "formal_result_eligible": False,
        "normalizer_schema": {"format_version": 2, "source_split": "calibration",
                              "state_dim": 27, "horizon": 4},
        "test_seeds_used": False,
        "train": {"split": "train", "seeds": list(range(1000, 1032)),
                  "replay_path": str(train), "replay_sha256": sha256_file(train)},
        "validation": {"split": "validation", "seeds": list(range(2000, 2008)),
                       "replay_path": str(validation),
                       "replay_sha256": sha256_file(validation)},
        "git": {"code_commit": "d" * 40, "git_dirty": False,
                "git_diff_sha256":
                    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
    }


class TestRSMBRLArtifactPreflight(unittest.TestCase):
    def test_rebuild_dirty_tree_fails_before_generation(self):
        with mock.patch.object(rebuild, "inspect_clean_git",
                               side_effect=RuntimeError("clean Git snapshot")), \
                mock.patch.object(rebuild, "_calibrate_to_path") as generate:
            with self.assertRaisesRegex(RuntimeError, "clean Git snapshot"):
                rebuild.calibrate()
            generate.assert_not_called()

    def test_replay_audit_rejects_test_seed_before_generation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            replay, summary = root / "train.jsonl", root / "train_summary.json"
            row = {key: None for key in rebuild.REPLAY_FIELDS}
            row.update(episode_seed=4000, state=[0.0] * 27, next_state=[0.0] * 27)
            replay.write_text(json.dumps(row) + "\n", encoding="utf-8")
            summary.write_text(json.dumps({"split": "train", "seeds": list(range(1000, 1032)),
                                           "state_dim": 27, "n_actions": 4,
                                           "episode_steps": 500,
                                           "transition_count": 1}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "seed coverage mismatch"):
                rebuild._audit_replay("train", replay, summary, tuple(range(1000, 1032)))

    def test_transactional_rebuild_promotes_only_after_preflight(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output, sidecar = root / "normalizer.pt", root / "normalizer.sidecar.json"
            audited = {
                "git": {"code_commit": "d" * 40, "git_dirty": False,
                        "git_diff_sha256":
                            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
                "train": {"split": "train", "seeds": list(range(1000, 1032)),
                          "replay_path": "train.jsonl", "replay_sha256": "1" * 64},
                "validation": {"split": "validation", "seeds": list(range(2000, 2008)),
                               "replay_path": "validation.jsonl", "replay_sha256": "2" * 64},
                "calibration": {"states_sha256": "3" * 64},
                "replay_manifest_sha256": "4" * 64,
                "model_manifest_sha256": "5" * 64,
            }
            def fake_generate(**kwargs):
                destination = Path(kwargs["output_path"])
                destination.write_bytes(b"new-normalizer")
                return {"status": "PASS", "output": str(destination),
                        "output_sha256": rebuild.sha256_file(destination),
                        "world_model_sha256": "6" * 64,
                        "reward_model_sha256": "7" * 64,
                        "provenance": {}, "coverage": {}}
            with mock.patch.object(rebuild, "audit_rebuild_inputs", return_value=audited), \
                    mock.patch.object(rebuild, "_calibrate_to_path", side_effect=fake_generate), \
                    mock.patch.object(rebuild, "run_preflight",
                                      return_value={"eligible": True, "errors": []}):
                report = rebuild.calibrate(output_path=output, sidecar_path=sidecar)
            self.assertEqual(output.read_bytes(), b"new-normalizer")
            self.assertTrue(sidecar.is_file())
            self.assertEqual(report["sidecar_sha256"], rebuild.sha256_file(sidecar))

    def test_failed_temporary_preflight_preserves_existing_pair(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output, sidecar = root / "normalizer.pt", root / "normalizer.sidecar.json"
            output.write_bytes(b"old-normalizer")
            sidecar.write_text("old-sidecar", encoding="utf-8")
            audited = {"git": {}, "train": {}, "validation": {}, "calibration": {},
                       "replay_manifest_sha256": "4" * 64,
                       "model_manifest_sha256": "5" * 64}
            def fake_generate(**kwargs):
                destination = Path(kwargs["output_path"])
                destination.write_bytes(b"new-normalizer")
                return {"output_sha256": rebuild.sha256_file(destination),
                        "world_model_sha256": "6" * 64, "reward_model_sha256": "7" * 64}
            with mock.patch.object(rebuild, "audit_rebuild_inputs", return_value=audited), \
                    mock.patch.object(rebuild, "_calibrate_to_path", side_effect=fake_generate), \
                    mock.patch.object(rebuild, "run_preflight",
                                      return_value={"eligible": False, "errors": ["bad SHA"]}):
                with self.assertRaisesRegex(RuntimeError, "bad SHA"):
                    rebuild.calibrate(output_path=output, sidecar_path=sidecar)
            self.assertEqual(output.read_bytes(), b"old-normalizer")
            self.assertEqual(sidecar.read_text(encoding="utf-8"), "old-sidecar")

    def test_sidecar_missing_fails_closed(self):
        payload, errors = validate_normalizer_sidecar(
            Path("does-not-exist.sidecar.json"), normalizer_sha256="a" * 64,
            world_sha256="b" * 64, reward_sha256="c" * 64,
        )
        self.assertIsNone(payload)
        self.assertTrue(any("missing frozen normalizer sidecar" in item for item in errors))

    def test_sidecar_binds_artifacts_replays_splits_and_git(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sidecar = valid_sidecar(root)
            path = root / "normalizer.sidecar.json"
            path.write_text(json.dumps(sidecar), encoding="utf-8")
            _, errors = validate_normalizer_sidecar(
                path, normalizer_sha256="a" * 64,
                world_sha256="b" * 64, reward_sha256="c" * 64,
            )
            self.assertEqual(errors, [])

            mutations = (
                ("normalizer", lambda x: x.update(normalizer_sha256="f" * 64)),
                ("world", lambda x: x.update(world_model_sha256="f" * 64)),
                ("train seeds", lambda x: x["train"].update(seeds=list(range(4000, 4100)))),
                ("replay", lambda x: x["validation"].update(replay_sha256="f" * 64)),
                ("git", lambda x: x["git"].update(git_dirty=True)),
                ("eligibility", lambda x: x.update(formal_result_eligible=True)),
            )
            for label, mutate in mutations:
                with self.subTest(label=label):
                    candidate = json.loads(json.dumps(sidecar))
                    mutate(candidate)
                    path.write_text(json.dumps(candidate), encoding="utf-8")
                    _, errors = validate_normalizer_sidecar(
                        path, normalizer_sha256="a" * 64,
                        world_sha256="b" * 64, reward_sha256="c" * 64,
                    )
                    self.assertTrue(errors)

    def test_development_pilot_exact_transition_gate(self):
        good = {"requested_transitions": 100, "environment_steps": 100,
                "controller_tick_end": 100, "all_agents_done": True, "errors": []}
        self.assertTrue(_episode_passed(good))
        for key in ("environment_steps", "controller_tick_end"):
            bad = dict(good); bad[key] = 99
            self.assertFalse(_episode_passed(bad))

    def test_calibration_state_audit_rejects_extra_fields(self):
        row = {"split": "calibration", "episode_seed": 3000,
               "agent_name": "blue_agent_0", "decision_index": 0,
               "global_tick_start": 0, "state": [0.0] * 27, "reward": 1.0}
        summary = {"split": "calibration", "seeds": list(range(3000, 3008)),
                   "episode_steps": 100,
                   "state_dim": 27, "contains_hidden_truth": False,
                   "contains_reward_labels": False,
                   "record_fields": [key for key in row if key != "reward"],
                   "record_count": 1, "per_agent_counts": {"blue_agent_0": 1},
                   "per_seed_counts": {"3000": 1}}
        with tempfile.TemporaryDirectory() as temp:
            states, meta = Path(temp) / "states.jsonl", Path(temp) / "summary.json"
            states.write_text(json.dumps(row) + "\n", encoding="utf-8")
            meta.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "extra fields"):
                audit_calibration_states(states, meta)

    def test_missing_bundle_has_actionable_failure(self):
        payload, errors = validate_frozen_normalizer(Path("does-not-exist.pt"), world_sha256="a" * 64)
        self.assertIsNone(payload)
        self.assertTrue(any("calibration seeds 3000..3007" in item for item in errors))

    def test_valid_frozen_bundle_passes_and_online_bundle_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bundle.pt"
            torch.save(valid_bundle(), path)
            _, errors = validate_frozen_normalizer(path, world_sha256="a" * 64)
            self.assertEqual(errors, [])
            bad = valid_bundle()
            bad["online_updates_after_calibration"] = True
            torch.save(bad, path)
            _, errors = validate_frozen_normalizer(path, world_sha256="a" * 64)
            self.assertTrue(any("online_updates_after_calibration" in item for item in errors))

    def test_wrong_hash_and_shape_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bundle.pt"
            bundle = valid_bundle()
            bundle["agents"]["blue_agent_0"]["horizon_std"] = torch.ones(3)
            torch.save(bundle, path)
            _, errors = validate_frozen_normalizer(path, world_sha256="b" * 64)
            self.assertTrue(any("world_model_sha256" in item for item in errors))
            self.assertTrue(any("expected shape (4,)" in item for item in errors))

    def test_legacy_or_wrong_projection_bundle_fails_closed(self):
        for label, mutate in (
            ("legacy", lambda x: x.update(format_version=1)),
            ("projection", lambda x: x.update(projection_sha256="0" * 64)),
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "bundle.pt"
                bundle = valid_bundle(); mutate(bundle); torch.save(bundle, path)
                _, errors = validate_frozen_normalizer(path, world_sha256="a" * 64)
                self.assertTrue(errors)

    def test_reward_profile_provenance_and_coverage_tampering_fail(self):
        mutations = (
            ("reward", lambda x: x.update(reward_model_sha256="d" * 64)),
            ("profile", lambda x: x["planner_profile"].update(population_size=64)),
            ("provenance", lambda x: x["provenance"].update(states_sha256="bad")),
            ("coverage", lambda x: x["coverage"]["blue_agent_0"].update(planner_calls=99)),
        )
        for label, mutate in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temp:
                bundle = valid_bundle()
                bundle["reward_model_sha256"] = "e" * 64
                mutate(bundle)
                path = Path(temp) / "bundle.pt"
                torch.save(bundle, path)
                _, errors = validate_frozen_normalizer(
                    path, world_sha256="a" * 64, reward_sha256="e" * 64
                )
                self.assertTrue(errors)


if __name__ == "__main__":
    unittest.main()
