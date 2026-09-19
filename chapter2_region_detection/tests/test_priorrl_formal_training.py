from pathlib import Path
import unittest

from baselines.priorrl_cc4.formal_training import (
    ALPHA_GRID, POLICY_SEEDS, TRAIN_SEEDS, VALIDATION_SEEDS,
    validate_alpha_selection, validate_checkpoint_metadata,
    validate_prototype_provenance, validate_training_manifest,
)
from formal_experiments.evaluation.run_method_episode import METHODS


class TestPriorRLFormalTraining(unittest.TestCase):
    def test_frozen_validation_split_matches_protocol(self):
        self.assertEqual(VALIDATION_SEEDS, tuple(range(2000, 2008)))

    def payload(self):
        candidates = [{"alpha_kl": alpha, "validation_response_reward_mean": -2.0}
                      for alpha in ALPHA_GRID]
        candidates[1]["validation_response_reward_mean"] = 3.0
        return {"schema": "priorrl_alpha_selection_v1", "train_seeds": list(TRAIN_SEEDS),
                "validation_seeds": list(VALIDATION_SEEDS), "test_seeds_used": False,
                "online_llm_calls": 0, "prototype_sha256": "a" * 64,
                "prototype_file_sha256": "b" * 64,
                "prototype_coverage_sha256": "c" * 64,
                "prototype_provenance_sha256": "d" * 64,
                "code_commit": "e" * 40, "git_dirty": False,
                "git_diff_sha256": "f" * 64,
                "candidates": candidates, "selected_alpha_kl": .05}

    def test_alpha_selection_contract_and_runner_registration(self):
        self.assertEqual(validate_alpha_selection(
            self.payload(), prototype_sha256="a" * 64,
            prototype_file_sha256="b" * 64,
            prototype_coverage_sha256="c" * 64,
            prototype_provenance_sha256="d" * 64), .05)
        self.assertIn("priorrl_ppo_cc4", METHODS)

        payload = self.payload(); payload["prototype_coverage_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "prototype_coverage_sha256 mismatch"):
            validate_alpha_selection(
                payload, prototype_sha256="a" * 64,
                prototype_file_sha256="b" * 64,
                prototype_coverage_sha256="c" * 64,
                prototype_provenance_sha256="d" * 64)

    def test_current_frozen_prototype_has_full_unified_provenance(self):
        root = Path(__file__).resolve().parents[1]
        binding = validate_prototype_provenance(
            root / "outputs/priorrl_cc4/prototypes/frozen_prototypes.json")
        self.assertEqual(binding["prototype_file_sha256"],
                         "127edfd36d4eb2b97b3a545c357b27c1b986cf67d23caad0ab9cbd07752fc119")
        self.assertEqual(binding["prototype_provenance_sha256"],
                         "ddbc74065fd2340ab9826dddee43a7a1dafb06c7ba8ecfc64d646862461d6358")

    def test_alpha_selection_rejects_test_leakage(self):
        payload = self.payload(); payload["test_seeds_used"] = True
        with self.assertRaisesRegex(ValueError, "isolation"):
            validate_alpha_selection(
                payload, prototype_sha256="a" * 64,
                prototype_file_sha256="b" * 64,
                prototype_coverage_sha256="c" * 64,
                prototype_provenance_sha256="d" * 64)

    def test_alpha_selection_tie_break_is_smallest_alpha(self):
        payload = self.payload()
        for row in payload["candidates"]: row["validation_response_reward_mean"] = 1.0
        payload["selected_alpha_kl"] = .05
        with self.assertRaisesRegex(ValueError, "tie-break"):
            validate_alpha_selection(
                payload, prototype_sha256="a" * 64,
                prototype_file_sha256="b" * 64,
                prototype_coverage_sha256="c" * 64,
                prototype_provenance_sha256="d" * 64)

    def test_training_manifest_and_checkpoint_require_complete_frozen_train_history(self):
        rows = [{"episode_seed": seed, "ticks": 500} for seed in TRAIN_SEEDS]
        value = {"schema": "priorrl_formal_training_manifest_v1",
                 "method": "PriorRL-PPO-CC4", "repeat_index": 1,
                 "policy_seed": POLICY_SEEDS[0], "alpha_kl": .05,
                 "train_seeds": list(TRAIN_SEEDS), "ticks_per_episode": 500,
                 "prototype_sha256": "a" * 64, "prototype_file_sha256": "b" * 64,
                 "prototype_coverage_sha256": "c" * 64,
                 "prototype_provenance_sha256": "d" * 64,
                 "checkpoint_file_sha256": "1" * 64,
                 "validation_selection_sha256": "2" * 64,
                 "online_llm_calls": 0, "world_model_used": False,
                 "test_seeds_used": False, "train": rows,
                 "code_commit": "e" * 40, "git_dirty": False,
                 "git_diff_sha256": "f" * 64}
        self.assertEqual(validate_training_manifest(
            value, repeat_index=1, prototype_sha256="a" * 64,
            prototype_file_sha256="b" * 64, prototype_coverage_sha256="c" * 64,
            prototype_provenance_sha256="d" * 64,
            checkpoint_file_sha256="1" * 64,
            selection_file_sha256="2" * 64), .05)
        metadata = {"schema": "priorrl_training_checkpoint_v1",
                    "protocol": "formal_train", "policy_seed": POLICY_SEEDS[0],
                    "alpha_kl": .05, "ticks": 500, "code_commit": "e" * 40,
                    "train": rows}
        validate_checkpoint_metadata(metadata, policy_seed=POLICY_SEEDS[0],
                                     alpha_kl=.05, code_commit="e" * 40)
        metadata["train"] = rows[:-1]
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_checkpoint_metadata(metadata, policy_seed=POLICY_SEEDS[0],
                                         alpha_kl=.05, code_commit="e" * 40)


if __name__ == "__main__":
    unittest.main()
