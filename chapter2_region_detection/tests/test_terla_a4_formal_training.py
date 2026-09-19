from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from baselines.terla_a4.formal_training import (
    POLICY_SEEDS, TEST_SEEDS, TICKS, TRAIN_SEEDS, VALIDATION_SEEDS,
    select_validation_checkpoint, train_policy,
)

CLEAN_GIT = {"code_commit": "a" * 40, "git_dirty": False,
             "git_diff_sha256": "0" * 64}


def clean_git():
    return dict(CLEAN_GIT)


class FakeEpisodes:
    def __init__(self, fail_once_at: int | None = None):
        self.calls = []
        self.fail_once_at = fail_once_at

    def __call__(self, *, policy, episode_seed, ticks, training, **_kwargs):
        self.calls.append((episode_seed, ticks, training))
        if training and episode_seed == self.fail_once_at:
            self.fail_once_at = None
            raise RuntimeError("simulated interruption")
        if training:
            with torch.no_grad():
                next(policy.parameters()).add_(0.000001)
        # Makes the 16-episode candidate win without reading test truth.
        distance = abs(float(next(policy.parameters()).flatten()[0].detach()) - self.target)
        return {"controller_tick_end": ticks, "reward_sum": -distance}

    @property
    def target(self):
        # Seeded initial value plus 16 deterministic mock training increments.
        torch.manual_seed(POLICY_SEEDS[0])
        from baselines.terla_a4.model import TERLAPolicy
        return float(next(TERLAPolicy().parameters()).flatten()[0].detach()) + 16e-6


class TestTERLAFormalTraining(unittest.TestCase):
    def test_frozen_protocol_and_validation_tie_break(self):
        self.assertEqual(TRAIN_SEEDS, tuple(range(1000, 1032)))
        self.assertEqual(VALIDATION_SEEDS, tuple(range(2000, 2008)))
        self.assertEqual(TEST_SEEDS, tuple(range(4000, 4100)))
        self.assertEqual(TICKS, 500)
        selected = select_validation_checkpoint([
            {"training_episodes": 16, "validation_seeds": list(VALIDATION_SEEDS),
             "validation_reward_mean": 1.0},
            {"training_episodes": 8, "validation_seeds": list(VALIDATION_SEEDS),
             "validation_reward_mean": 1.0},
        ])
        self.assertEqual(selected["training_episodes"], 8)
        with self.assertRaisesRegex(ValueError, "frozen validation"):
            select_validation_checkpoint([{"training_episodes": 8,
                "validation_seeds": [4000], "validation_reward_mean": 9.0}])

    def test_resume_selection_and_evaluator_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            interrupted = FakeEpisodes(fail_once_at=1005)
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                train_policy(policy_seed=51001, output_dir=out, episode_runner=interrupted,
                             git_state=clean_git)
            saved = torch.load(out / "policy_51001.resume.pt", map_location="cpu", weights_only=False)
            self.assertEqual(saved["completed_training_seeds"], list(range(1000, 1005)))

            resumed = FakeEpisodes()
            manifest = train_policy(policy_seed=51001, output_dir=out, resume=True,
                                    episode_runner=resumed, git_state=clean_git)
            train_calls = [seed for seed, ticks, training in resumed.calls if training]
            validation_calls = [seed for seed, ticks, training in resumed.calls if not training]
            self.assertEqual(train_calls, list(range(1005, 1032)))
            self.assertEqual(validation_calls, list(VALIDATION_SEEDS) * 4)
            self.assertFalse(set(seed for seed, *_ in resumed.calls) & set(TEST_SEEDS))
            self.assertEqual(manifest["selected_training_episodes"], 16)

            final = torch.load(out / "policy_51001.pt", map_location="cpu", weights_only=True)
            self.assertEqual(final["method"], "terla_a4")
            self.assertEqual(final["policy_seed"], 51001)
            self.assertIs(final["formal_training_complete"], True)
            self.assertEqual(final["training_split"], "train")
            self.assertEqual(final["training_episode_seeds"], list(TRAIN_SEEDS))
            self.assertIs(final["test_seeds_used"], False)
            self.assertIs(final["shared_policy_across_agents"], True)
            self.assertIs(final["per_agent_history_isolated"], True)
            self.assertIs(final["hidden_truth_policy_input"], False)
            self.assertEqual(final["code_commit"], CLEAN_GIT["code_commit"])
            self.assertIs(final["git_dirty"], False)
            self.assertEqual(final["git_diff_sha256"], CLEAN_GIT["git_diff_sha256"])
            candidate = torch.load(out / "policy_51001.step_16.pt", map_location="cpu",
                                   weights_only=True)
            for key, value in CLEAN_GIT.items():
                self.assertEqual(candidate[key], value)

    def test_dirty_git_and_cross_commit_resume_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            dirty = lambda: {**CLEAN_GIT, "git_dirty": True, "git_diff_sha256": "1" * 64}
            with self.assertRaisesRegex(RuntimeError, "clean git worktree"):
                train_policy(policy_seed=51001, output_dir=out, episode_runner=FakeEpisodes(),
                             git_state=dirty)
            interrupted = FakeEpisodes(fail_once_at=1001)
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                train_policy(policy_seed=51001, output_dir=out, episode_runner=interrupted,
                             git_state=clean_git)
            other_commit = lambda: {**CLEAN_GIT, "code_commit": "b" * 40}
            with self.assertRaisesRegex(RuntimeError, "git provenance mismatch"):
                train_policy(policy_seed=51001, output_dir=out, resume=True,
                             episode_runner=FakeEpisodes(), git_state=other_commit)

    def test_wrong_seed_and_nonformal_ticks_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "policy_seed"):
                train_policy(policy_seed=999, output_dir=Path(directory), episode_runner=FakeEpisodes(),
                             git_state=clean_git)
            with self.assertRaisesRegex(ValueError, "500 ticks"):
                train_policy(policy_seed=51001, output_dir=Path(directory), ticks=20,
                             episode_runner=FakeEpisodes(), git_state=clean_git)


if __name__ == "__main__":
    unittest.main()
