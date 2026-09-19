import tempfile
import unittest
from pathlib import Path
import json

import torch

from baselines.rsmbrl_cc4.artifact_preflight import validate_frozen_normalizer
from baselines.rsmbrl_cc4.calibrate_normalizers import audit_calibration_states
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


class TestRSMBRLArtifactPreflight(unittest.TestCase):
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
