import json
import tempfile
import unittest
from pathlib import Path

from formal_experiments.ours.run_table23_formal import (
    FORMAL_EPISODE_TICKS, POLICY_SEEDS, TEST_SEEDS, TRAIN_SEEDS,
    VALIDATION_SEEDS, FormalStageRunner, ResumeJournal, RunProfile,
    assert_immutable, physical_jobs, preflight, resolved_repeat_metadata,
    row_specs, select_validation_checkpoint,
)


class TestTable23FormalRunner(unittest.TestCase):
    def test_frozen_profile_and_shared_lwm_full_identity(self):
        profile = RunProfile()
        self.assertEqual((profile.train_seeds, profile.validation_seeds, profile.test_seeds),
                         (TRAIN_SEEDS, VALIDATION_SEEDS, TEST_SEEDS))
        self.assertEqual(profile.episode_ticks, FORMAL_EPISODE_TICKS)
        self.assertEqual(len(row_specs()), 7); self.assertEqual(len(physical_jobs()), 6)
        shared = [x for x in row_specs() if x.canonical_id == "lwm_full"]
        self.assertEqual({(x.table_id, x.row_id) for x in shared},
                         {("table2", "lwm_rl"), ("table3", "full_reward")})
        with self.assertRaisesRegex(ValueError, "formal profile"):
            RunProfile("formal", (1000,), VALIDATION_SEEDS, TEST_SEEDS, 500)

    def test_dev_is_never_eligible_and_repeat_seed_is_frozen(self):
        profile = RunProfile("dev", (1000,), (2000,), (3200,), 20)
        meta = resolved_repeat_metadata(row_specs()[0], repeat_index=2, profile=profile)
        self.assertFalse(meta["formal_result_eligible"])
        self.assertEqual(meta["training_seed"], POLICY_SEEDS[1])

    def test_validation_selection_rejects_test_data_and_tie_breaks(self):
        sha_a, sha_b = "a" * 64, "b" * 64
        selected = select_validation_checkpoint([
            {"epoch": 2, "validation_score": 1, "checkpoint_sha256": sha_a},
            {"epoch": 1, "validation_score": 1, "checkpoint_sha256": sha_b},
        ])
        self.assertEqual(selected["epoch"], 1)
        with self.assertRaisesRegex(ValueError, "test fields"):
            select_validation_checkpoint([{"epoch": 1, "validation_score": 1,
                                           "checkpoint_sha256": sha_a, "test_score": 9}])

    def test_fail_only_gate_blocks_current_formal_preflight(self):
        root = Path(__file__).resolve().parents[1]
        report = preflight(project_root=root, profile=RunProfile())
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("Fail-Only" in item and "FAIL" in item for item in report["errors"]))

    def test_stage_runner_split_isolation_resume_and_immutable_test(self):
        calls = []
        hashes = {"policy": "a" * 64, "world": "b" * 64, "reward": "c" * 64, "scaler": "d" * 64}
        callbacks = {
            "preflight": lambda: {"status": "PASS"},
            "train": lambda seeds: calls.append(("train", tuple(seeds))) or {"checkpoint_count": 2},
            "validation": lambda seeds: calls.append(("validation", tuple(seeds))) or [
                {"epoch": 1, "validation_score": 2.0, "checkpoint_sha256": "e" * 64}],
            "test": lambda seeds: calls.append(("test", tuple(seeds))) or {"decisions_audited": True},
            "hashes": lambda: hashes,
            "integrity": lambda _: {"passed": True},
        }
        with tempfile.TemporaryDirectory() as temporary:
            runner = FormalStageRunner(spec=row_specs()[0], repeat_index=1,
                                       profile=RunProfile(), run_dir=Path(temporary), callbacks=callbacks)
            result = runner.run(); runner.run()
            self.assertEqual([x[0] for x in calls], ["train", "validation", "test"])
            self.assertEqual(calls[0][1], TRAIN_SEEDS); self.assertEqual(calls[1][1], VALIDATION_SEEDS)
            self.assertEqual(calls[2][1], TEST_SEEDS)
            self.assertEqual(set(result["completed"]), set(ResumeJournal.STAGES))

    def test_test_hash_mutation_fails(self):
        with self.assertRaisesRegex(RuntimeError, "mutated"):
            assert_immutable({"policy": "a"}, {"policy": "b"})


if __name__ == "__main__": unittest.main()
