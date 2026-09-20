import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from baselines.uamcts_cc4.artifacts import (
    ENSEMBLE_SIZE,
    PROGRESS_DERIVATION_VERSION,
    PROGRESS_LABEL,
    PROGRESS_SIDECAR_SCHEMA,
    validate_normalizer_sidecar,
    validate_prior_entropy_sidecar,
    validate_progress_sidecar,
)
from formal_experiments.evaluation.run_method_repeat import run_repeat


def _fake_runner(*, method, seed, ticks, run_mode, device, policy_seed):
    episode = {
        "schema_version": 1, "protocol_version": "cc4_v3_20260917",
        "episode_seed": seed, "tick_count": ticks, "episode_end_tick": ticks,
        "tick_team_rewards": [0.0] * ticks, "operation_failure_events": [],
        "recovery_actions": [], "incidents": [], "method": method,
        "policy_seed": policy_seed, "run_mode": run_mode,
        "formal_result_eligible": run_mode == "formal",
    }
    return episode, []


class TestUAMCTSTrustChain(unittest.TestCase):
    def _sidecar(self, root: Path, *, progress_sha="a" * 64):
        path = root / "progress.sidecar.json"
        path.write_text(json.dumps({
            "schema": PROGRESS_SIDECAR_SCHEMA,
            "checkpoint_sha256": progress_sha,
            "progress_sha256": progress_sha,
            "canonical_train_replay_sha256": "b" * 64,
            "canonical_train_record_count": 43297,
            "canonical_train_seeds": list(range(1000, 1032)),
            "validation_replay_sha256": "c" * 64,
            "validation_record_count": 8,
            "validation_seeds": list(range(2000, 2008)),
            "label_contract": {"label": PROGRESS_LABEL,
                               "derivation_version": PROGRESS_DERIVATION_VERSION},
            "ensemble_size": ENSEMBLE_SIZE,
            "validation_rmse": 0.091226,
            "constant_baseline_rmse": 0.128235,
            "validation_gate_pass": True,
            "provider_calls": 0,
            "test_seeds_used": False,
        }, sort_keys=True), encoding="utf-8")
        return path

    def test_complete_progress_sidecar_accepts_canonical_train_binding(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sidecar = self._sidecar(root)
            _, errors = validate_progress_sidecar(
                sidecar, progress_sha256="a" * 64,
                canonical_train_replay_sha256="b" * 64,
                canonical_train_record_count=43297,
                validation_replay_sha256="c" * 64,
            )
            self.assertEqual(errors, [])

    def test_progress_sidecar_missing_and_tampered_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sidecar = self._sidecar(root)
            _, errors = validate_progress_sidecar(
                root / "missing.json", progress_sha256="a" * 64,
                canonical_train_replay_sha256="b" * 64,
                canonical_train_record_count=43297,
            )
            self.assertTrue(any("missing" in item for item in errors))
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            payload["checkpoint_sha256"] = "d" * 64
            sidecar.write_text(json.dumps(payload), encoding="utf-8")
            _, errors = validate_progress_sidecar(
                sidecar, progress_sha256="a" * 64,
                canonical_train_replay_sha256="b" * 64,
                canonical_train_record_count=43297,
            )
            self.assertTrue(any("checkpoint_sha256" in item for item in errors))

    def test_entropy_and_normalizer_sidecars_bind_all_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            entropy = root / "entropy.sidecar.json"
            entropy.write_text(json.dumps({
                "schema": "uamcts_validation_prior_entropy_sidecar_v1",
                "artifact_sha256": "e" * 64,
                "prototype_artifact_sha256": "p" * 64,
                "prototype_sha256": "q" * 64,
                "split": "validation", "validation_seeds": list(range(2000, 2008)),
                "validation_replay_sha256": "v" * 64,
                "aligned_unique_records": 10, "provider_calls": 0,
                "test_seeds_used": False, "hits": 10, "misses": 0,
            }), encoding="utf-8")
            _, errors = validate_prior_entropy_sidecar(
                entropy, entropy_sha256="e" * 64,
                prototype_artifact_sha256="p" * 64, prototype_sha256="q" * 64,
                validation_replay_sha256="v" * 64,
            )
            self.assertFalse(errors, errors)
            normalizer = root / "normalizer.sidecar.json"
            normalizer.write_text(json.dumps({
                "schema": "uamcts_uncertainty_normalizers_sidecar_v1",
                "artifact_sha256": "n" * 64, "world_model_sha256": "w" * 64,
                "reward_model_sha256": "r" * 64, "progress_sha256": "p" * 64,
                "prior_entropy_sha256": "e" * 64, "split": "validation",
                "validation_seeds": list(range(2000, 2008)),
                "validation_replay_sha256": "v" * 64,
                "validation_aligned_total": 10,
                "uncertainty_sources": ["world_model", "progress", "prior_entropy"],
                "test_updates_allowed": False, "test_seeds_used": False,
                "provider_calls": 0,
            }), encoding="utf-8")
            _, errors = validate_normalizer_sidecar(
                normalizer, normalizer_sha256="n" * 64,
                world_model_sha256="w" * 64, reward_model_sha256="r" * 64,
                progress_sha256="p" * 64, prior_entropy_sha256="e" * 64,
                validation_replay_sha256="v" * 64,
            )
            self.assertFalse(errors, errors)
            payload = json.loads(normalizer.read_text(encoding="utf-8"))
            payload["provider_calls"] = 1
            normalizer.write_text(json.dumps(payload), encoding="utf-8")
            _, errors = validate_normalizer_sidecar(
                normalizer, normalizer_sha256="n" * 64,
                world_model_sha256="w" * 64, reward_model_sha256="r" * 64,
                progress_sha256="p" * 64, prior_entropy_sha256="e" * 64,
                validation_replay_sha256="v" * 64,
            )
            self.assertTrue(any("provider_calls" in item for item in errors))

    def test_resume_rejects_commit_or_artifact_drift_and_snapshot_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            state = {"code_commit": "a" * 40, "git_dirty": False,
                     "git_diff_sha256": "b" * 64}
            gate = ({"schema": "uamcts_real_gate_v2", "eligible": True},
                    {"world_model_sha256": "w" * 64})
            with patch("formal_experiments.evaluation.run_method_repeat.git_state",
                       return_value=state), \
                 patch("formal_experiments.evaluation.run_method_repeat._uamcts_preflight",
                       return_value=gate), \
                 patch("formal_experiments.evaluation.run_method_repeat.artifact_provenance",
                       return_value=({"policy_spec_sha256": "p" * 64}, "w" * 64)), \
                 patch("formal_experiments.evaluation.run_method_repeat.dependency_and_hardware",
                       return_value=({}, {})), \
                 patch("formal_experiments.evaluation.run_method_repeat.validate_run_directory",
                       return_value={"passed": True, "errors": [], "warnings": []}):
                run_repeat(method="uamcts_cc4", run_mode="dev", repeat_index=1,
                           policy_seed=61001, episode_seeds=[39000], ticks=20,
                           output=root, episode_runner=_fake_runner)
                snapshot = root / "preflight_snapshot.json"
                before = snapshot.read_bytes()
                snapshot.write_text("tampered", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "snapshot overwrite refused"):
                    run_repeat(method="uamcts_cc4", run_mode="dev", repeat_index=1,
                               policy_seed=61001, episode_seeds=[39000], ticks=20,
                               output=root, resume=True, episode_runner=_fake_runner)
                self.assertEqual(snapshot.read_text(encoding="utf-8"), "tampered")
                snapshot.write_bytes(before)
            with patch("formal_experiments.evaluation.run_method_repeat.git_state",
                       return_value={"code_commit": "c" * 40, "git_dirty": False,
                                     "git_diff_sha256": "b" * 64}), \
                 patch("formal_experiments.evaluation.run_method_repeat._uamcts_preflight",
                       return_value=gate):
                with self.assertRaisesRegex(RuntimeError, "(config overwrite refused|resume identity drift)"):
                    run_repeat(method="uamcts_cc4", run_mode="dev", repeat_index=1,
                               policy_seed=61001, episode_seeds=[39000], ticks=20,
                               output=root, resume=True, episode_runner=_fake_runner)

    def test_non_uamcts_keeps_legacy_identity_and_has_no_preflight_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            observed = []

            def runner(**kwargs):
                observed.append(((root / "config.resolved.yaml").exists(),
                                 (root / "preflight_snapshot.json").exists()))
                return _fake_runner(**kwargs)

            with patch("formal_experiments.evaluation.run_method_repeat.git_state",
                       return_value={"code_commit": "a" * 40, "git_dirty": False,
                                     "git_diff_sha256": "b" * 64}):
                result = run_repeat(method="dca_cc4", run_mode="dev", repeat_index=1,
                                    policy_seed=61001, episode_seeds=[39000], ticks=20,
                                    output=root, episode_runner=runner)
            self.assertEqual(observed, [(False, False)])
            self.assertEqual(
                json.loads((root / "resume_index.json").read_text(encoding="utf-8"))["identity"],
                {"method": "dca_cc4", "run_mode": "dev", "repeat_index": 1,
                 "policy_seed": 61001, "episode_seeds": [39000], "ticks": 20},
            )
            self.assertNotIn("preflight_snapshot_sha256", result["manifest"])
            self.assertNotIn("frozen_artifact_sha256", result["manifest"])
            self.assertNotIn("provider_calls", result["manifest"])

    def test_uamcts_blocked_preflight_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "blocked"
            called = False

            def runner(**kwargs):
                nonlocal called
                called = True
                return _fake_runner(**kwargs)

            with patch("formal_experiments.evaluation.run_method_repeat._uamcts_preflight",
                       side_effect=RuntimeError("UAMCTS artifact preflight failed")):
                with self.assertRaisesRegex(RuntimeError, "artifact preflight"):
                    run_repeat(method="uamcts_cc4", run_mode="dev", repeat_index=1,
                               policy_seed=61001, episode_seeds=[39000], ticks=20,
                               output=root, episode_runner=runner)
            self.assertFalse(root.exists())
            self.assertFalse(called)


if __name__ == "__main__":
    unittest.main()
