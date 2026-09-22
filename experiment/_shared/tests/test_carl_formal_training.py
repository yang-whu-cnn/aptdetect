import json
from pathlib import Path
import tempfile
import unittest

import torch

from baselines.carl_cc4.augmentation import gate_imagination_horizon
from baselines.carl_cc4.buffers import CARLTransition
from baselines.carl_cc4.formal_training import (
    GitState, METHOD, POLICY_SEEDS, TICKS, TRAIN_SEEDS, VALIDATION_SEEDS,
    select_validation_candidate, train_repeat,
)


def transition(seed, source, action=0):
    return CARLTransition(seed, source, (0.0,) * 27, action, -0.1, (0.0,) * 27, 1)


class TestCARLFormalTraining(unittest.TestCase):
    clean_git = staticmethod(lambda: GitState("1" * 40, False, "e" * 64))
    def candidate(self, name, score):
        return {"candidate_id": name, "validation_score": score,
                "training_episode_seeds": list(TRAIN_SEEDS),
                "validation_episode_seeds": list(VALIDATION_SEEDS),
                "test_seeds_used": False, "world_model_sha256": "a" * 64,
                "code_commit": "1" * 40, "git_dirty": False,
                "git_diff_sha256": "e" * 64}

    def test_validation_selection_uses_frozen_split_and_rejects_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "selection.json"
            result = select_validation_candidate([self.candidate("b", 2), self.candidate("a", 2)], output=path)
            self.assertEqual(result["selected_candidate_id"], "a")
            bad = self.candidate("bad", 3); bad["validation_episode_seeds"] = [4000]
            with self.assertRaisesRegex(ValueError, "isolation"):
                select_validation_candidate([bad], output=path)
            missing_git = self.candidate("missing-git", 3); del missing_git["code_commit"]
            with self.assertRaisesRegex(ValueError, "Git provenance"):
                select_validation_candidate([missing_git], output=path)

    def test_gate_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            selection = Path(tmp) / "selection.json"
            select_validation_candidate([self.candidate("default", 1)], output=selection)
            with self.assertRaisesRegex(RuntimeError, "H=4"):
                train_repeat(repeat_index=1, output=Path(tmp) / "policy.pt",
                    gate=gate_imagination_horizon(), world_model_sha256="a" * 64,
                    selection_path=selection, collect_real=lambda *args: [],
                    generate_synthetic=lambda *args: [], fit_seed_scm=lambda *args: None)

    def test_dirty_git_is_fail_closed_before_collection(self):
        with tempfile.TemporaryDirectory() as tmp:
            selection = Path(tmp) / "selection.json"
            select_validation_candidate([self.candidate("default", 1)], output=selection)
            with self.assertRaisesRegex(RuntimeError, "clean Git"):
                train_repeat(repeat_index=1, output=Path(tmp) / "policy.pt",
                    gate=gate_imagination_horizon(allow_disclosed_truncation=True),
                    world_model_sha256="a" * 64, selection_path=selection,
                    collect_real=lambda *args: self.fail("collection must not start"),
                    generate_synthetic=lambda *args: [], fit_seed_scm=lambda *args: None,
                    git_state=lambda: GitState("1" * 40, True, "d" * 64))

    def test_resume_rejects_a_different_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            selection = Path(tmp) / "selection.json"
            select_validation_candidate([self.candidate("default", 1)], output=selection)
            output = Path(tmp) / "policy.pt"
            def collect(seed, policy, ticks):
                if seed == TRAIN_SEEDS[1]: raise RuntimeError("intentional interruption")
                return [transition(seed, "real")]
            kwargs = dict(repeat_index=1, output=output,
                gate=gate_imagination_horizon(allow_disclosed_truncation=True),
                world_model_sha256="a" * 64, selection_path=selection, collect_real=collect,
                generate_synthetic=lambda seed, real, policy, horizon, scm:
                    [[transition(seed, "synthetic")] for _ in range(8)],
                fit_seed_scm=lambda seed, rows: object())
            with self.assertRaisesRegex(RuntimeError, "intentional"):
                train_repeat(**kwargs, git_state=self.clean_git)
            with self.assertRaisesRegex(ValueError, "validation-selection"):
                train_repeat(**kwargs, resume=True,
                    git_state=lambda: GitState("2" * 40, False, "e" * 64))

    def test_runner_writes_evaluator_metadata_and_keeps_seed_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            selection = Path(tmp) / "selection.json"
            select_validation_candidate([self.candidate("default", 1)], output=selection)
            seen_scm = []
            def collect(seed, policy, ticks):
                self.assertEqual(ticks, TICKS); return [transition(seed, "real")]
            def synthetic(seed, real, policy, horizon, scm):
                self.assertEqual(horizon, 4)
                self.assertEqual(scm, seed)
                return [[transition(seed, "synthetic")] for _ in range(8)]
            output = Path(tmp) / f"policy_{POLICY_SEEDS[0]}.pt"
            meta = train_repeat(repeat_index=1, output=output,
                gate=gate_imagination_horizon(allow_disclosed_truncation=True),
                world_model_sha256="a" * 64, selection_path=selection,
                collect_real=collect, generate_synthetic=synthetic,
                fit_seed_scm=lambda seed, rows: (seen_scm.append(
                    (seed, {x.episode_seed for x in rows})), seed)[1],
                git_state=self.clean_git)
            saved = torch.load(output, map_location="cpu", weights_only=True)
            self.assertEqual(saved["method"], METHOD)
            self.assertEqual(saved["policy_seed"], 51001)
            self.assertTrue(saved["formal_training_complete"])
            self.assertEqual(saved["training_split"], "train")
            self.assertEqual(tuple(saved["training_episode_seeds"]), TRAIN_SEEDS)
            self.assertEqual(saved["code_commit"], "1" * 40)
            self.assertIs(saved["git_dirty"], False)
            self.assertEqual(saved["git_diff_sha256"], "e" * 64)
            copied_selection = output.with_suffix(".selection.json")
            self.assertTrue(copied_selection.is_file())
            self.assertEqual(saved["validation_selection_file"], copied_selection.name)
            self.assertEqual([x[0] for x in seen_scm], list(TRAIN_SEEDS))
            self.assertTrue(all(seeds == {seed} for seed, seeds in seen_scm))
            self.assertEqual(meta["imagination_gate_status"], "ADAPTED_TRUNCATED")


if __name__ == "__main__":
    unittest.main()
