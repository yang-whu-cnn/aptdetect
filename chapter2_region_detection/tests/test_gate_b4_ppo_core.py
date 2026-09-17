import inspect
import math
import unittest

import numpy as np
import torch

from formal_experiments.ours.ppo_core import (
    CandidateActorCritic,
    PPOCoreConfig,
    compute_duration_aware_gae,
    compute_ppo_losses,
    duration_discounts,
    normalize_advantages,
)
from formal_experiments.ours.posterior_features import POSTERIOR_CANDIDATE_DIM


class TestGateB4PPOCore(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260917)
        self.model = CandidateActorCritic(PPOCoreConfig())

    def features(self, *shape):
        return torch.randn(*shape, POSTERIOR_CANDIDATE_DIM, dtype=torch.float32)

    def test_network_shapes_for_unbatched_and_batched_k6(self):
        logits, value = self.model(self.features(6))
        self.assertEqual(tuple(logits.shape), (6,))
        self.assertEqual(tuple(value.shape), ())

        logits_b, value_b = self.model(self.features(7, 6))
        self.assertEqual(tuple(logits_b.shape), (7, 6))
        self.assertEqual(tuple(value_b.shape), (7,))
        self.assertTrue(torch.isfinite(logits_b).all())
        self.assertTrue(torch.isfinite(value_b).all())

    def test_network_supports_variable_candidate_count(self):
        for k in (1, 3, 6, 10):
            logits, value = self.model(self.features(4, k))
            self.assertEqual(tuple(logits.shape), (4, k))
            self.assertEqual(tuple(value.shape), (4,))

    def test_actor_equivariant_and_critic_invariant_to_candidate_permutation(self):
        features = self.features(3, 6)
        permutation = torch.tensor([5, 2, 0, 4, 1, 3], dtype=torch.long)
        base_logits, base_value = self.model(features)
        perm_logits, perm_value = self.model(features[:, permutation])
        torch.testing.assert_close(perm_logits, base_logits[:, permutation])
        torch.testing.assert_close(perm_value, base_value)

    def test_deterministic_action_tracks_same_candidate_under_permutation(self):
        features = self.features(6)
        with torch.no_grad():
            logits, _ = self.model(features)
            original = int(torch.argmax(logits).item())
            permutation = torch.tensor([5, 2, 0, 4, 1, 3], dtype=torch.long)
            permuted_action, _, _, _ = self.model.act(
                features[permutation], deterministic=True
            )
        expected_new_index = int((permutation == original).nonzero(as_tuple=False).item())
        self.assertEqual(int(permuted_action.item()), expected_new_index)

    def test_orthogonal_output_head_gains_and_zero_biases(self):
        self.assertAlmostEqual(float(self.model.actor_head.weight.norm().item()), 0.01, places=5)
        self.assertAlmostEqual(float(self.model.critic_head.weight.norm().item()), 1.0, places=5)
        for layer in (
            self.model.encoder_fc1,
            self.model.encoder_fc2,
            self.model.actor_head,
            self.model.critic_fc,
            self.model.critic_head,
        ):
            torch.testing.assert_close(layer.bias, torch.zeros_like(layer.bias))

    def test_network_and_action_guards(self):
        with self.assertRaises(ValueError):
            self.model(torch.zeros(6, 45))
        bad = torch.zeros(6, 46)
        bad[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            self.model(bad)
        with self.assertRaises(ValueError):
            self.model.evaluate_actions(self.features(2, 6), torch.tensor([0]))
        with self.assertRaises(ValueError):
            self.model.evaluate_actions(self.features(2, 6), torch.tensor([0, 6]))

    def test_duration_discount_uses_gamma_tick_power_dt(self):
        discounts = duration_discounts([1, 2, 3, 5], gamma_tick=0.99)
        expected = torch.tensor([0.99, 0.99**2, 0.99**3, 0.99**5], dtype=torch.float32)
        torch.testing.assert_close(discounts, expected)
        with self.assertRaises(ValueError):
            duration_discounts([0, 1])
        with self.assertRaises(ValueError):
            duration_discounts([1.5, 2])

    def test_duration_aware_gae_matches_manual_reference(self):
        rewards = torch.tensor([-1.0, -2.0, -3.0], dtype=torch.float32)
        values = torch.tensor([0.5, 0.4, 0.2], dtype=torch.float32)
        next_values = torch.tensor([0.4, 0.2, 0.0], dtype=torch.float32)
        dones = torch.tensor([False, False, True])
        dt = torch.tensor([1, 2, 3])
        gamma = 0.99
        lam = 0.95

        out = compute_duration_aware_gae(
            rewards,
            values,
            next_values,
            dones,
            dt,
            gamma_tick=gamma,
            gae_lambda=lam,
        )

        g0, g1, g2 = gamma, gamma**2, gamma**3
        d2 = -3.0 - 0.2
        a2 = d2
        d1 = -2.0 + g1 * 0.2 - 0.4
        a1 = d1 + g1 * lam * a2
        d0 = -1.0 + g0 * 0.4 - 0.5
        a0 = d0 + g0 * lam * a1
        expected_adv = torch.tensor([a0, a1, a2], dtype=torch.float32)
        expected_delta = torch.tensor([d0, d1, d2], dtype=torch.float32)
        torch.testing.assert_close(out.advantages, expected_adv)
        torch.testing.assert_close(out.deltas, expected_delta)
        torch.testing.assert_close(out.returns, expected_adv + values)
        torch.testing.assert_close(out.discounts, torch.tensor([g0, g1, g2]))

    def test_gae_lambda_is_per_decision_not_lambda_power_dt(self):
        out = compute_duration_aware_gae(
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [False, True],
            [5, 1],
            gamma_tick=0.99,
            gae_lambda=0.95,
        )
        expected_first = (0.99**5) * 0.95
        self.assertAlmostEqual(float(out.advantages[0]), expected_first, places=6)
        self.assertNotAlmostEqual(
            float(out.advantages[0]),
            (0.99**5) * (0.95**5),
            places=5,
        )

    def test_terminal_transition_cuts_bootstrap_and_gae_chain(self):
        out = compute_duration_aware_gae(
            [2.0, 100.0],
            [1.0, 0.0],
            [999.0, 0.0],
            [True, True],
            [5, 1],
        )
        self.assertAlmostEqual(float(out.deltas[0]), 1.0, places=6)
        self.assertAlmostEqual(float(out.advantages[0]), 1.0, places=6)

    def test_gae_api_excludes_world_model_predicted_reward_target(self):
        parameters = inspect.signature(compute_duration_aware_gae).parameters
        self.assertIn("real_response_rewards", parameters)
        for forbidden in ("predicted_reward", "predicted_value", "wm_reward", "llm_prior"):
            self.assertNotIn(forbidden, parameters)

    def test_advantage_normalization_has_zero_mean_and_unit_population_std(self):
        normalized = normalize_advantages(torch.tensor([1.0, 2.0, 4.0, 8.0]))
        self.assertAlmostEqual(float(normalized.mean()), 0.0, places=6)
        self.assertAlmostEqual(float(normalized.std(unbiased=False)), 1.0, places=6)

    def test_ppo_clipping_matches_reference_and_reports_clip_fraction(self):
        old = torch.zeros(4)
        ratios = torch.tensor([1.0, 1.1, 1.3, 0.7], dtype=torch.float32)
        new = torch.log(ratios)
        adv = torch.tensor([1.0, 1.0, 1.0, -1.0], dtype=torch.float32)
        values = torch.tensor([0.0, 1.0, 2.0, 3.0])
        returns = torch.tensor([0.0, 1.0, 1.0, 4.0])
        entropy = torch.tensor([0.5, 0.5, 0.5, 0.5])
        out = compute_ppo_losses(
            new,
            old,
            adv,
            values,
            returns,
            entropy,
            clip_epsilon=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
        )
        surrogate = torch.minimum(
            ratios * adv,
            torch.clamp(ratios, 0.8, 1.2) * adv,
        )
        expected_policy = -surrogate.mean()
        expected_value = 0.5 * torch.mean((values - returns) ** 2)
        expected_total = expected_policy + 0.5 * expected_value - 0.01 * 0.5
        torch.testing.assert_close(out.policy_loss, expected_policy)
        torch.testing.assert_close(out.value_loss, expected_value)
        torch.testing.assert_close(out.total_loss, expected_total)
        self.assertAlmostEqual(float(out.clip_fraction), 0.5, places=6)
        self.assertGreaterEqual(float(out.approx_kl), -1e-7)

    def test_ppo_loss_guards_nonfinite_and_length_mismatch(self):
        with self.assertRaises(ValueError):
            compute_ppo_losses(
                [0.0, 0.0],
                [0.0],
                [1.0, 1.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.1, 0.1],
            )
        with self.assertRaises(ValueError):
            compute_ppo_losses(
                [0.0, float("nan")],
                [0.0, 0.0],
                [1.0, 1.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.1, 0.1],
            )


if __name__ == "__main__":
    unittest.main()
