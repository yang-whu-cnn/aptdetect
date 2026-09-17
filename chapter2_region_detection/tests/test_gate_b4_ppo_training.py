import inspect
import unittest
from pathlib import Path

import numpy as np
import torch

from formal_experiments.ours.ppo_core import CandidateActorCritic, PPOCoreConfig
from formal_experiments.ours.ppo_training import (
    AsyncPPORolloutBuffer,
    FORMAL_TRAIN_SEEDS,
    PPOTrainer,
    PPOTrainingConfig,
    RolloutStep,
    assert_formal_train_seed,
    build_training_batch,
)
from formal_experiments.ours.posterior_features import POSTERIOR_CANDIDATE_DIM
from shared.formal_state import BLUE_AGENTS


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "formal_experiments" / "ours" / "ppo_training.py"


class TestGateB4PPOTraining(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260917)
        np.random.seed(20260917)
        self.policy = CandidateActorCritic(PPOCoreConfig())

    def features(self):
        return torch.randn(6, POSTERIOR_CANDIDATE_DIM, dtype=torch.float32)

    def policy_snapshot(self, features, action=0):
        with torch.no_grad():
            evaluation = self.policy.evaluate_actions(features.unsqueeze(0), torch.tensor([action]))
        return float(evaluation.log_prob[0]), float(evaluation.value[0])

    def make_step(
        self,
        *,
        seed=1000,
        agent="blue_agent_0",
        index=0,
        reward=-1.0,
        dt=1,
        done=False,
        action=0,
        next_value=0.0,
    ):
        features = self.features()
        log_prob, value = self.policy_snapshot(features, action=action)
        return RolloutStep(
            episode_seed=seed,
            agent_name=agent,
            decision_index=index,
            candidate_features=features,
            candidate_index=action,
            behavior_log_prob=log_prob,
            critic_value=value,
            critic_next_value=float(next_value),
            real_response_reward=float(reward),
            decision_dt=int(dt),
            done=bool(done),
        )

    def test_train_seed_contract_accepts_only_frozen_train_split(self):
        self.assertEqual(FORMAL_TRAIN_SEEDS, tuple(range(1000, 1032)))
        self.assertEqual(assert_formal_train_seed(1000), 1000)
        self.assertEqual(assert_formal_train_seed(1031), 1031)
        for bad in (999, 1032, 2000, 3000, 4000):
            with self.assertRaises(ValueError):
                assert_formal_train_seed(bad)

    def test_async_buffer_begin_complete_contract(self):
        buffer = AsyncPPORolloutBuffer(train_only=True)
        features = self.features()
        log_prob, value = self.policy_snapshot(features, action=2)
        buffer.begin(
            episode_seed=1000,
            agent_name="blue_agent_0",
            decision_index=0,
            candidate_features=features,
            candidate_index=2,
            behavior_log_prob=log_prob,
            critic_value=value,
        )
        self.assertEqual(buffer.open_agents, ("blue_agent_0",))
        step = buffer.complete(
            agent_name="blue_agent_0",
            real_response_reward=-4.5,
            decision_dt=3,
            done=False,
            critic_next_value=0.25,
        )
        self.assertEqual(len(buffer), 1)
        self.assertEqual(buffer.open_agents, ())
        self.assertEqual(step.decision_dt, 3)
        self.assertAlmostEqual(step.real_response_reward, -4.5)
        self.assertAlmostEqual(step.critic_next_value, 0.25)
        self.assertEqual(tuple(step.candidate_features.shape), (6, 46))

    def test_async_buffer_rejects_busy_duplicate_and_index_gaps(self):
        buffer = AsyncPPORolloutBuffer()
        features = self.features()
        lp, value = self.policy_snapshot(features)
        buffer.begin(
            episode_seed=1000,
            agent_name="blue_agent_0",
            decision_index=0,
            candidate_features=features,
            candidate_index=0,
            behavior_log_prob=lp,
            critic_value=value,
        )
        with self.assertRaises(RuntimeError):
            buffer.begin(
                episode_seed=1000,
                agent_name="blue_agent_0",
                decision_index=0,
                candidate_features=features,
                candidate_index=0,
                behavior_log_prob=lp,
                critic_value=value,
            )
        buffer.complete(
            agent_name="blue_agent_0",
            real_response_reward=0.0,
            decision_dt=1,
            done=False,
            critic_next_value=0.0,
        )
        with self.assertRaises(ValueError):
            buffer.begin(
                episode_seed=1000,
                agent_name="blue_agent_0",
                decision_index=2,
                candidate_features=features,
                candidate_index=0,
                behavior_log_prob=lp,
                critic_value=value,
            )

    def test_terminal_transition_requires_zero_critic_bootstrap(self):
        buffer = AsyncPPORolloutBuffer()
        features = self.features()
        lp, value = self.policy_snapshot(features)
        buffer.begin(
            episode_seed=1000,
            agent_name="blue_agent_0",
            decision_index=0,
            candidate_features=features,
            candidate_index=0,
            behavior_log_prob=lp,
            critic_value=value,
        )
        with self.assertRaises(ValueError):
            buffer.complete(
                agent_name="blue_agent_0",
                real_response_reward=-1.0,
                decision_dt=1,
                done=True,
                critic_next_value=1.0,
            )

    def test_buffer_rejects_validation_test_seeds_and_bad_feature_action(self):
        for seed in (2000, 3000, 4000):
            buffer = AsyncPPORolloutBuffer(train_only=True)
            with self.assertRaises(ValueError):
                buffer.begin(
                    episode_seed=seed,
                    agent_name="blue_agent_0",
                    decision_index=0,
                    candidate_features=self.features(),
                    candidate_index=0,
                    behavior_log_prob=0.0,
                    critic_value=0.0,
                )
        buffer = AsyncPPORolloutBuffer()
        with self.assertRaises(ValueError):
            buffer.begin(
                episode_seed=1000,
                agent_name="blue_agent_0",
                decision_index=0,
                candidate_features=torch.zeros(5, 46),
                candidate_index=0,
                behavior_log_prob=0.0,
                critic_value=0.0,
            )
        with self.assertRaises(ValueError):
            buffer.begin(
                episode_seed=1000,
                agent_name="blue_agent_0",
                decision_index=0,
                candidate_features=self.features(),
                candidate_index=6,
                behavior_log_prob=0.0,
                critic_value=0.0,
            )

    def test_build_batch_isolates_agent_trajectories(self):
        # Agent 0 has a two-step chain. Agent 1 is terminal and must not receive any
        # continuation from agent 0 despite the merged training batch.
        a0_0 = self.make_step(agent="blue_agent_0", index=0, reward=0.0, dt=1, done=False, next_value=0.0)
        a0_1 = self.make_step(agent="blue_agent_0", index=1, reward=10.0, dt=1, done=True, next_value=0.0)
        a1_0 = self.make_step(agent="blue_agent_1", index=0, reward=-3.0, dt=5, done=True, next_value=0.0)
        batch = build_training_batch([a1_0, a0_1, a0_0], normalize_advantage=False)
        self.assertEqual(batch.size, 3)
        self.assertEqual(
            batch.provenance,
            ((1000, "blue_agent_0", 0), (1000, "blue_agent_0", 1), (1000, "blue_agent_1", 0)),
        )
        # Terminal one-step return is exactly real reward because A = r - V; return=A+V.
        self.assertAlmostEqual(float(batch.returns[2]), -3.0, places=5)
        self.assertGreater(float(batch.returns[0]), 0.0)

    def test_build_batch_rejects_noncontiguous_or_post_terminal_trajectory(self):
        with self.assertRaises(ValueError):
            build_training_batch(
                [
                    self.make_step(index=0),
                    self.make_step(index=2),
                ],
                normalize_advantage=False,
            )
        with self.assertRaises(ValueError):
            build_training_batch(
                [
                    self.make_step(index=0, done=True),
                    self.make_step(index=1, done=False),
                ],
                normalize_advantage=False,
            )

    def test_batch_advantage_normalization_is_global_after_per_agent_gae(self):
        steps = []
        for agent_index, agent in enumerate(BLUE_AGENTS[:2]):
            steps.append(
                self.make_step(
                    agent=agent,
                    index=0,
                    reward=float(agent_index + 1),
                    done=True,
                    next_value=0.0,
                )
            )
        batch = build_training_batch(steps, normalize_advantage=True)
        self.assertAlmostEqual(float(batch.advantages.mean()), 0.0, places=6)
        self.assertAlmostEqual(float(batch.advantages.std(unbiased=False)), 1.0, places=6)

    def test_training_config_matches_frozen_defaults(self):
        cfg = PPOTrainingConfig()
        self.assertEqual(cfg.learning_rate, 3e-4)
        self.assertEqual(cfg.rollout_length, 128)
        self.assertEqual(cfg.update_epochs, 5)
        self.assertEqual(cfg.minibatch_size, 64)
        self.assertEqual(cfg.max_grad_norm, 0.5)
        self.assertTrue(cfg.normalize_advantage)

    def test_trainer_update_is_finite_and_changes_policy_parameters(self):
        steps = []
        for index in range(8):
            steps.append(
                self.make_step(
                    index=index,
                    reward=float((index % 3) - 2),
                    dt=(1, 2, 3, 5)[index % 4],
                    done=(index == 7),
                    action=index % 6,
                    next_value=0.0,
                )
            )
        batch = build_training_batch(steps, normalize_advantage=True)
        cfg = PPOTrainingConfig(update_epochs=2, minibatch_size=4, seed=123)
        trainer = PPOTrainer(self.policy, training_config=cfg)
        before = [parameter.detach().clone() for parameter in self.policy.parameters()]
        metrics = trainer.update(batch)
        after = list(self.policy.parameters())
        self.assertEqual(metrics.samples, 8)
        self.assertEqual(metrics.optimizer_steps, 4)
        self.assertTrue(all(np.isfinite([
            metrics.policy_loss_mean,
            metrics.value_loss_mean,
            metrics.entropy_mean,
            metrics.approx_kl_mean,
            metrics.clip_fraction_mean,
            metrics.grad_norm_mean,
            metrics.grad_norm_max,
        ])))
        self.assertTrue(any(not torch.equal(a, b.detach()) for a, b in zip(before, after)))

    def test_optimizer_epochs_have_no_provider_or_environment_dependency(self):
        parameters = inspect.signature(PPOTrainer.update).parameters
        self.assertEqual(list(parameters), ["self", "batch"])
        source = SOURCE.read_text(encoding="utf-8").lower()
        for forbidden in (
            "openai",
            "ofox",
            "api_key",
            "chat.completions",
            "priorcache",
            "make_env",
            "cyborg",
            "hidden red",
        ):
            self.assertNotIn(forbidden, source)

    def test_real_reward_field_is_explicit_and_no_predicted_reward_field_exists(self):
        fields = RolloutStep.__dataclass_fields__
        self.assertIn("real_response_reward", fields)
        self.assertIn("critic_next_value", fields)
        for forbidden in ("predicted_reward", "wm_reward", "llm_reward", "prior_reward"):
            self.assertNotIn(forbidden, fields)


if __name__ == "__main__":
    unittest.main()
