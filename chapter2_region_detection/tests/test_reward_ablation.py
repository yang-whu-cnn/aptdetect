import unittest
import inspect
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch

from formal_experiments.ours.reward_ablation import (
    FAIL_ONLY_FORMAL_BETA, FAIL_ONLY_FORMAL_LOSS, RewardMode,
    formal_predictor_config, tick_reward, train_hurdle_failure,
    transition_reward,
)
from formal_experiments.training.response_reward_predictor import (
    ResponseRewardDataset, ResponseRewardPredictor, ResponseRewardPredictorConfig,
)


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
        self.assertIn("ResponseRewardPredictor(predictor_config", source)

    def test_fail_only_loss_is_the_only_formal_optimizer_change(self):
        baseline = formal_predictor_config(RewardMode.DELAY_ONLY)
        fail_only = formal_predictor_config(RewardMode.FAIL_ONLY)
        self.assertEqual(baseline.loss, "mse")
        self.assertEqual(fail_only.loss, FAIL_ONLY_FORMAL_LOSS)
        self.assertEqual(fail_only.smooth_l1_beta, FAIL_ONLY_FORMAL_BETA)
        baseline_fields = vars(baseline).copy(); fail_fields = vars(fail_only).copy()
        baseline_fields.pop("loss"); fail_fields.pop("loss")
        self.assertEqual(baseline_fields, fail_fields)

    def test_smooth_l1_config_round_trips_checkpoint(self):
        rng = np.random.default_rng(7); n = 16
        dataset = ResponseRewardDataset(
            states=rng.normal(size=(n, 27)).astype(np.float32),
            actions=np.arange(n) % 4,
            next_states=rng.normal(size=(n, 27)).astype(np.float32),
            rewards=np.asarray([0] * 12 + [-1, -2, -3, -4], dtype=np.float32),
        )
        config = ResponseRewardPredictorConfig(
            hidden_dim=16, batch_size=8, epochs=1,
            loss="smooth_l1", smooth_l1_beta=1.0,
        )
        predictor = ResponseRewardPredictor(config); predictor.fit(dataset)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "reward.pt"; predictor.save_checkpoint(path)
            loaded = ResponseRewardPredictor.load_checkpoint(path)
        self.assertEqual(loaded.config.loss, "smooth_l1")
        self.assertEqual(loaded.config.smooth_l1_beta, 1.0)

    def test_formal_fail_only_loss_sees_each_standardized_label_once(self):
        rng = np.random.default_rng(9); n = 17
        dataset = ResponseRewardDataset(
            states=rng.normal(size=(n, 27)).astype(np.float32),
            actions=np.arange(n) % 4,
            next_states=rng.normal(size=(n, 27)).astype(np.float32),
            rewards=np.asarray([0] * 13 + [-1, -2, -3, -4], dtype=np.float32),
        )
        config = ResponseRewardPredictorConfig(
            hidden_dim=16, batch_size=8, epochs=1,
            loss="smooth_l1", smooth_l1_beta=1.0,
        )
        predictor = ResponseRewardPredictor(config)
        target = "formal_experiments.training.response_reward_predictor.F.smooth_l1_loss"
        with patch(target, wraps=torch.nn.functional.smooth_l1_loss) as loss_spy:
            predictor.fit(dataset)
        observed = np.concatenate([
            call.args[1].detach().cpu().numpy() for call in loss_spy.call_args_list
        ])
        expected = predictor.reward_normalizer.normalize_np(dataset.rewards)
        np.testing.assert_allclose(np.sort(observed), np.sort(expected), rtol=0, atol=0)
        self.assertEqual(observed.size, n)
        for call in loss_spy.call_args_list:
            self.assertEqual(call.kwargs, {"beta": 1.0})

    def test_legacy_v1_checkpoint_without_loss_fields_defaults_to_mse(self):
        rng = np.random.default_rng(11); n = 12
        dataset = ResponseRewardDataset(
            states=rng.normal(size=(n, 27)).astype(np.float32),
            actions=np.arange(n) % 4,
            next_states=rng.normal(size=(n, 27)).astype(np.float32),
            rewards=rng.normal(size=n).astype(np.float32),
        )
        predictor = ResponseRewardPredictor(ResponseRewardPredictorConfig(
            hidden_dim=16, batch_size=6, epochs=1,
        )); predictor.fit(dataset)
        with TemporaryDirectory() as directory:
            current = Path(directory) / "current.pt"; legacy = Path(directory) / "legacy.pt"
            predictor.save_checkpoint(current)
            payload = torch.load(current, map_location="cpu", weights_only=False)
            payload["config"].pop("loss"); payload["config"].pop("smooth_l1_beta")
            torch.save(payload, legacy)
            loaded = ResponseRewardPredictor.load_checkpoint(legacy)
        self.assertEqual(loaded.config.loss, "mse")
        self.assertEqual(loaded.config.smooth_l1_beta, 1.0)


if __name__ == "__main__": unittest.main()
