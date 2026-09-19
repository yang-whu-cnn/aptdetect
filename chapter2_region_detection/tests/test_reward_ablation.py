import unittest
import inspect
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from formal_experiments.ours.reward_ablation import (
    RewardMode, tick_reward, train_hurdle_failure, transition_reward,
)
from formal_experiments.training.response_reward_predictor import ResponseRewardDataset


class TestRewardAblation(unittest.TestCase):
    def accounting(self, delay=3.0, failure=-7.0, full=-10.0):
        return SimpleNamespace(incident_delay_penalty=delay,
                               incident_host_lwf_raw_penalty=failure,
                               response_reward=full)

    def test_modes_partition_full_reward_exactly(self):
        item = self.accounting()
        delay = tick_reward(item, RewardMode.DELAY_ONLY)
        failure = tick_reward(item, RewardMode.FAIL_ONLY)
        full = tick_reward(item, RewardMode.FULL_REWARD)
        self.assertEqual((delay, failure, full), (-3.0, -7.0, -10.0))
        self.assertEqual(delay + failure, full)

    def test_full_reward_mismatch_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "mismatch"):
            tick_reward(self.accounting(full=-9), RewardMode.FULL_REWARD)

    def test_invalid_component_labels_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "invalid"):
            transition_reward(self.accounting(delay=-1), RewardMode.DELAY_ONLY)
        with self.assertRaisesRegex(ValueError, "invalid"):
            transition_reward(self.accounting(failure=1), RewardMode.FAIL_ONLY)

    def test_no_test_seed_or_online_llm_dependency(self):
        from formal_experiments.ours import reward_ablation
        source = Path(reward_ablation.__file__).read_text(encoding="utf-8")
        self.assertIn('"test_seeds_used": False', source)
        self.assertNotIn("ofox", source.lower())
        self.assertNotIn("online_llm", source.lower())

    def test_hurdle_predictor_is_nonpositive_and_reports_sparsity(self):
        rng = np.random.default_rng(3); n = 12
        dataset = ResponseRewardDataset(states=rng.normal(size=(n, 27)).astype(np.float32),
                                        actions=np.arange(n) % 4,
                                        next_states=rng.normal(size=(n, 27)).astype(np.float32),
                                        rewards=np.asarray([0] * 9 + [-1, -2, -3], dtype=np.float32))
        predictor, report = train_hurdle_failure(dataset, device="cpu")
        predicted = predictor.predict(dataset.states, dataset.actions, dataset.next_states)
        self.assertTrue(np.isfinite(predicted).all()); self.assertTrue((predicted <= 0).all())
        self.assertEqual(report["nonzero_count"], 3)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "hurdle.pt"; predictor.save_checkpoint(path)
            loaded = type(predictor).load_checkpoint(path, device="cpu")
            np.testing.assert_allclose(predicted, loaded.predict(dataset.states, dataset.actions,
                                                                 dataset.next_states), rtol=0, atol=1e-6)

    def test_formal_training_path_never_uses_diagnostic_hurdle(self):
        from formal_experiments.ours.reward_ablation import train_mode
        source = inspect.getsource(train_mode)
        self.assertNotIn("train_hurdle_failure(", source)
        self.assertIn("ResponseRewardPredictor(ResponseRewardPredictorConfig()", source)


if __name__ == "__main__": unittest.main()
