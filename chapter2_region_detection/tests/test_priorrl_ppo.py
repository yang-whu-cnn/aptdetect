import unittest

import numpy as np
import torch

from baselines.priorrl_cc4.development_pilot import (
    DevelopmentCollector, DevelopmentPilotConfig, FormalRewardChannel, development_manifest,
    run_development_pilot,
)
from baselines.priorrl_cc4.policy import PriorRLActorCritic, forward_kl, prior_regularized_ppo_loss
from baselines.priorrl_cc4.training import A4PPOConfig, A4PPOTrainer, A4RolloutBuffer, checkpoint_sha256
from formal_experiments.data_collection.incident_response import IncidentResponseBookkeeper


class FakeRetriever:
    def __init__(self): self.calls = 0; self.blocked = False; self.seen = []
    def lookup(self, state, *, agent_name):
        if self.blocked: raise AssertionError("optimizer accessed retriever/cache")
        self.calls += 1; self.seen.append((np.asarray(state).copy(), agent_name))
        return torch.tensor([.1, .2, .3, .4])


def add_rows(buffer, n=4):
    for i in range(n):
        done = i == n-1
        buffer.add(state=np.full(27, i / 10, np.float32), action=i % 4,
            old_log_prob=-1.4, value=.1, reward=float(done), next_value=0.0 if done else .2,
            done=done, decision_dt=(1, 2, 3, 5)[i % 4],
            frozen_prior=[.1, .2, .3, .4], trajectory_id="agent0:episode0")


class TestPriorRLPPO(unittest.TestCase):
    def test_numpy_and_cpu_input_follow_parameter_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PriorRLActorCritic().to(device)
        expected = next(model.parameters()).device
        for state in (np.zeros(27, np.float32), torch.zeros(27, device="cpu")):
            logits, value = model(state)
            self.assertEqual(logits.device, expected); self.assertEqual(value.device, expected)

    def test_forward_kl_direction_and_alpha_zero(self):
        logits = torch.tensor([[1.2, -.3, .2, .7]])
        prior = torch.tensor([[.55, .1, .2, .15]])
        pi = logits.softmax(-1)
        expected = (pi * (pi.log() - prior.log())).sum()
        torch.testing.assert_close(forward_kl(logits, prior), expected)
        base = torch.tensor(2.5)
        torch.testing.assert_close(prior_regularized_ppo_loss(base, logits, prior, alpha_kl=0), base)

    def test_masked_behavior_and_forward_kl_are_stable(self):
        model = PriorRLActorCritic()
        with torch.no_grad():
            model.actor.bias.copy_(torch.tensor([-10., 100., 90., 80.]))
        action, _, _ = model.act(np.zeros(27, np.float32), deterministic=True,
                                 action_mask=[True, False, False, False])
        self.assertEqual(int(action), 0)
        logits = torch.tensor([[2., 50., -1., 30.]], requires_grad=True)
        prior = torch.tensor([[.1, .7, .2, 0.]])
        kl = forward_kl(logits, prior, torch.tensor([[True, False, True, False]]))
        self.assertTrue(torch.isfinite(kl)); kl.backward()
        self.assertEqual(float(logits.grad[0, 1]), 0.0)
        self.assertEqual(float(logits.grad[0, 3]), 0.0)

    def test_duration_aware_buffer_and_optimizer_never_calls_retriever(self):
        retriever = FakeRetriever(); model = PriorRLActorCritic(); buffer = A4RolloutBuffer()
        collector = DevelopmentCollector(model, retriever, buffer)
        decision = collector.decide(np.zeros(27, np.float32), agent_name="blue_agent_0")
        bookkeeper = IncidentResponseBookkeeper(episode_seed=1000, agent_name="blue_agent_0")
        bookkeeper.reset(initial_red_presence_by_host={"host0": False})
        reward = FormalRewardChannel(bookkeeper)
        reward.record_hidden_tick(global_tick_end=1, red_presence_after={"host0": False})
        collector.complete(decision, next_observable_state=np.ones(27, np.float32),
                           reward_channel=reward, done=True, decision_dt=5,
                           trajectory_id="blue_agent_0:1000")
        tensors = buffer.training_tensors(A4PPOConfig())
        self.assertEqual(int(tensors["decision_dt"][0]), 5)
        retriever.blocked = True
        metrics = A4PPOTrainer(model, A4PPOConfig(epochs=1, minibatch_size=1)).optimize(buffer)
        self.assertEqual(retriever.calls, 1); self.assertTrue(metrics)

    def test_seed_split_hidden_boundary_and_checkpoint_hash(self):
        config = DevelopmentPilotConfig(training_episode_budget=3)
        self.assertEqual(config.train_seeds(), (1000, 1001, 1002))
        self.assertEqual(config.eval_seed_map(), {61001: (3200,3201,3202,3203,3204),
                                                  61002: (3205,3206,3207,3208,3209)})
        retriever = FakeRetriever(); model = PriorRLActorCritic()
        DevelopmentCollector(model, retriever, A4RolloutBuffer()).decide(
            np.zeros(27, np.float32), agent_name="blue_agent_0")
        self.assertEqual(retriever.seen[0][0].shape, (27,))
        policies = {61001: PriorRLActorCritic(), 61002: PriorRLActorCritic()}
        manifest = development_manifest(config, policies)
        self.assertFalse(manifest["formal_result_eligible"])
        before = checkpoint_sha256(policies[61001]); self.assertEqual(before, checkpoint_sha256(policies[61001]))
        with torch.no_grad(): next(policies[61001].parameters()).add_(1)
        self.assertNotEqual(before, checkpoint_sha256(policies[61001]))

    def test_development_runner_executes_exact_split_and_is_ineligible(self):
        config = DevelopmentPilotConfig(training_episode_budget=2)
        train_calls = []; eval_calls = []
        def train_episode(**kwargs):
            train_calls.append((kwargs["policy_seed"], kwargs["episode_seed"], kwargs["episode_ticks"]))
            return {"seed": kwargs["episode_seed"]}
        def evaluate_episode(**kwargs):
            eval_calls.append((kwargs["policy_seed"], kwargs["episode_seed"], kwargs["episode_ticks"]))
            return {"seed": kwargs["episode_seed"]}
        result = run_development_pilot(
            config, policy_factory=lambda _: PriorRLActorCritic(),
            retriever_factory=lambda _: FakeRetriever(), train_episode=train_episode,
            evaluate_episode=evaluate_episode, device="cpu")
        self.assertEqual(train_calls, [(61001,1000,100),(61001,1001,100),
                                       (61002,1000,100),(61002,1001,100)])
        self.assertEqual(eval_calls, [(61001,s,100) for s in range(3200,3205)] +
                                     [(61002,s,100) for s in range(3205,3210)])
        self.assertFalse(result["formal_result_eligible"])

    def test_transition_finiteness_terminal_and_trajectory_guards(self):
        base = dict(state=np.zeros(27, np.float32), action=0, old_log_prob=-1.0,
                    value=0.0, reward=0.0, next_value=0.0, done=True, decision_dt=1,
                    frozen_prior=[.25]*4, trajectory_id="episode")
        for field in ("old_log_prob", "value", "reward", "next_value"):
            with self.subTest(field=field):
                buffer = A4RolloutBuffer(); bad = dict(base); bad[field] = float("nan")
                with self.assertRaisesRegex(ValueError, "finite"):
                    buffer.add(**bad)
        with self.assertRaisesRegex(ValueError, "next_value"):
            A4RolloutBuffer().add(**(base | {"next_value": .01}))
        with self.assertRaisesRegex(ValueError, "non-empty"):
            A4RolloutBuffer().add(**(base | {"trajectory_id": " "}))
        buffer = A4RolloutBuffer(); buffer.add(**base)
        buffer.add(**(base | {"done": False, "next_value": .1}))
        with self.assertRaisesRegex(ValueError, "after terminal"):
            buffer.training_tensors(A4PPOConfig())

    def test_rollout_buffer_clear_preserves_object_identity(self):
        buffer = A4RolloutBuffer(); add_rows(buffer, 2)
        identity = id(buffer); buffer.clear()
        self.assertEqual(id(buffer), identity); self.assertEqual(len(buffer), 0)


if __name__ == "__main__": unittest.main()
