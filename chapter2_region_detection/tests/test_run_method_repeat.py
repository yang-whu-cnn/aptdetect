import json
import tempfile
import unittest
from pathlib import Path

from formal_experiments.evaluation.run_method_repeat import run_repeat, validate_repeat_request
from unittest.mock import patch


def fake_runner(*, method, seed, ticks, run_mode, device, policy_seed):
    episode = {
        "schema_version": 1, "protocol_version": "cc4_v3_20260917", "episode_seed": seed,
        "tick_count": ticks, "episode_end_tick": ticks,
        "tick_team_rewards": [0.0] * ticks, "operation_failure_events": [],
        "recovery_actions": [{"executed_action": "remove", "started": True, "completed": True,
                              "fallback": False, "active_incident_before": True}],
        "incidents": [{"t_compromise": 0, "t_recovered": 1}], "method": method,
        "policy_seed": policy_seed, "run_mode": run_mode,
        "formal_result_eligible": run_mode == "formal",
    }
    return episode, [{"episode_seed": seed, "method": method, "policy_seed": policy_seed}]


class TestRunMethodRepeat(unittest.TestCase):
    def test_formal_contract_locks_repeat_seed_episodes_and_ticks(self):
        validate_repeat_request(run_mode="formal", repeat_index=1, policy_seed=51001,
                                episode_seeds=list(range(4000, 4100)), ticks=500)
        with self.assertRaises(ValueError):
            validate_repeat_request(run_mode="formal", repeat_index=1, policy_seed=51002,
                                    episode_seeds=list(range(4000, 4100)), ticks=500)
        with self.assertRaises(ValueError):
            validate_repeat_request(run_mode="formal", repeat_index=1, policy_seed=51001,
                                    episode_seeds=[4000], ticks=20)

    def test_dev_repeat_writes_manifest_and_validates(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = run_repeat(method="dca_cc4", run_mode="dev", repeat_index=1,
                                policy_seed=61001, episode_seeds=[39000, 39001], ticks=20,
                                output=Path(temporary), episode_runner=fake_runner)
            self.assertTrue(result["validation"]["passed"])
            self.assertFalse(result["manifest"]["formal_result_eligible"])
            self.assertEqual(result["manifest"]["method"], "DCA-CC4 (adapted)")
            self.assertEqual(result["manifest"]["method_slug"], "dca_cc4")
            self.assertEqual(len(result["manifest"]["paper_sha256"]), 64)
            self.assertIn("provenance_audit_sha256", result["manifest"]["method_artifacts"])
            self.assertEqual(result["manifest"]["test_episode_seeds"], [39000, 39001])
            report = json.loads((Path(temporary) / "eligibility_report.json").read_text())
            self.assertEqual(report["status"], "PASS")
            self.assertNotIn("eligibility_report.json", result["manifest"]["artifact_sha256"])
            for name in ("config.resolved.yaml", "policy_spec.json", "episodes.jsonl",
                         "decisions.jsonl", "metrics.json"):
                self.assertIn(name, result["manifest"]["artifact_sha256"])

    def test_resume_rejects_polluted_episode_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_repeat(method="dca_cc4", run_mode="dev", repeat_index=1,
                       policy_seed=61001, episode_seeds=[39000], ticks=20,
                       output=root, episode_runner=fake_runner)
            path = root / "episode_parts" / "episode_39000.json"
            path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                run_repeat(method="dca_cc4", run_mode="dev", repeat_index=1,
                           policy_seed=61001, episode_seeds=[39000], ticks=20,
                           output=root, resume=True, episode_runner=fake_runner)

    @patch("formal_experiments.evaluation.run_method_repeat.git_state")
    def test_formal_dirty_snapshot_fails_before_episode_execution(self, state):
        state.return_value = {"code_commit": "a" * 40, "git_dirty": True,
                              "git_diff_sha256": "b" * 64}
        called = False
        def runner(**kwargs):
            nonlocal called
            called = True
            return fake_runner(**kwargs)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "clean git"):
                run_repeat(method="dca_cc4", run_mode="formal", repeat_index=1,
                           policy_seed=51001, episode_seeds=list(range(4000, 4100)), ticks=500,
                           output=Path(temporary), episode_runner=runner)
        self.assertFalse(called)


if __name__ == "__main__":
    unittest.main()
