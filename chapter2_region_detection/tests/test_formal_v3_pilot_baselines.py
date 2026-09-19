import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from baselines.dca_cc4 import (DCAConfig, DCAResponseAdapter, TabularAttackPathModel,
                               cluster_alerts, cluster_state, tokenize_alerts)
from baselines.dca_cc4.runtime import CC4ObservableAlertStream, DCAAgentRuntime
from baselines.dca_cc4.run_pilot import _episode_passed
from baselines.priorrl_cc4 import (OfflineActionPrior, PriorRLPPOConfig, PriorRLActorCritic,
                                  action_prior_from_plans, forward_kl,
                                  prior_regularized_ppo_loss)
from baselines.rsmbrl_cc4 import RSMBRLConfig, risk_adjusted_score
from baselines.rsmbrl_cc4 import RSMBRLPlanner
from baselines.ug_cem_apt.categorical_cem import CategoricalCEMConfig, CategoricalCEMOptimizer
from baselines.ug_cem_apt.uncertainty import UGUncertainty, UGUncertaintyConfig
from shared.rollout_evaluator import SharedRolloutResult


class TestDCAAdapter(unittest.TestCase):
    def test_episode_gate_rejects_off_by_one_transition_count_or_tick(self):
        good = {"requested_ticks": 100, "environment_steps": 100,
                "controller_tick_end": 100, "all_agents_done": True, "errors": []}
        self.assertTrue(_episode_passed(good))
        for field in ("environment_steps", "controller_tick_end"):
            bad = dict(good); bad[field] = 99
            self.assertFalse(_episode_passed(bad))
        incomplete = dict(good); incomplete["all_agents_done"] = False
        self.assertFalse(_episode_passed(incomplete))
    def test_cc4_stream_reads_visible_rows_and_never_controller_truth(self):
        class Tracker:
            def canonical_hostname(self, raw, _data): return str(raw)
        stream = CC4ObservableAlertStream(Tracker())
        observation = {"success": "UNKNOWN", "h1": {
            "Processes": [{"PID": 7}], "Connections": [{"remote": "10.0.0.2"}],
            "Files": [{"path": "/tmp/x"}], "System info": {"Hostname": "h1"},
        }}
        tokens = stream.extract(observation, tick=4)
        self.assertEqual([x.evidence_type for x in tokens], ["connection", "file", "process"])
        self.assertTrue(all(x.host == "h1" and x.tick == 4 for x in tokens))

    def test_runtime_discards_visible_evidence_outside_temporal_window(self):
        class Tracker:
            def canonical_hostname(self, raw, _data): return str(raw)
        runtime = DCAAgentRuntime(tracker=Tracker(), config=DCAConfig(time_window=10))
        visible = {"h1": {"Processes": [{"PID": 7}]}}
        runtime.observe(visible, tick=0)
        runtime.observe(visible, tick=10)
        self.assertEqual([token.tick for token in runtime.tokens], [0, 10])
        runtime.observe(visible, tick=11)
        self.assertEqual([token.tick for token in runtime.tokens], [10, 11])

    def test_dca_runner_has_no_hidden_state_or_red_session_policy_read(self):
        runner = Path(importlib.util.find_spec("baselines.dca_cc4.run_pilot").origin).read_text()
        runtime = Path(importlib.util.find_spec("baselines.dca_cc4.runtime").origin).read_text()
        policy_sources = runner + runtime
        self.assertNotIn("controller.state", policy_sources)
        self.assertNotIn("ground_truth_red_presence", policy_sources)
        self.assertNotIn("red_agent", policy_sources.lower())
    def test_four_fixed_mapping_branches_and_observed_target(self):
        adapter = DCAResponseAdapter()
        self.assertEqual(adapter.act([]).requested_action_id, 0)
        single = [{"tick": 1, "host": "h1", "type": "process", "confidence": .9}]
        single_decision = adapter.act(single)
        self.assertEqual(single_decision.requested_action_id, 1)
        self.assertEqual(single_decision.target, "h1")
        self.assertEqual(single_decision.reason, "unclustered_or_single_observable_evidence")
        low = [{"tick": i, "host": "h1", "type": "process", "confidence": .9} for i in range(3)]
        high = low + [{"tick": 2, "host": "h1", "type": "network", "confidence": .8}]
        decision = adapter.act(high)
        self.assertEqual((decision.requested_action_id, decision.target), (2, "h1"))
        high[1]["persisted_after_remove"] = True
        self.assertEqual(adapter.act(high).requested_action_id, 3)

    def test_hidden_truth_is_rejected_and_deterministic(self):
        with self.assertRaises(ValueError):
            tokenize_alerts([{"host": "h", "type": "x", "compromised": True}])
        events = [{"tick": i, "host": "h", "type": "x", "confidence": .2} for i in range(3)]
        self.assertEqual(DCAResponseAdapter().act(events), DCAResponseAdapter().act(events))

    def test_known_sequence_dbscan_and_temporal_separation(self):
        raw = ([{"tick": i, "host": "h1", "type": "process", "confidence": .8} for i in (1, 2, 3)] +
               [{"tick": i, "host": "h2", "type": "network", "confidence": .8} for i in (30, 31, 32)] +
               [{"tick": 90, "host": "noise", "type": "file", "confidence": .9}])
        clusters = cluster_alerts(tokenize_alerts(raw), DCAConfig())
        self.assertEqual([[x.host for x in c] for c in clusters], [["h1"] * 3, ["h2"] * 3])
        self.assertEqual([[x.tick for x in c] for c in clusters], [[1, 2, 3], [30, 31, 32]])
        noise_decision = DCAResponseAdapter().act([raw[-1]])
        self.assertEqual((noise_decision.requested_action_id, noise_decision.target), (1, "noise"))
        self.assertEqual(noise_decision.reason, "unclustered_or_single_observable_evidence")

    def test_path_dyna_update_value_iteration_and_adapter_consumption(self):
        model = TabularAttackPathModel(gamma=.5, dyna_steps=4, seed=7)
        s0 = ("h1", ("network", "process")); s1 = ("h1", ("file", "network", "process"))
        model.update(s0, "credential_access", s1, service_impact=.2, n_steps=2,
                     objective_reached=False)
        model.update(s1, "impact", "terminal", service_impact=1.0, n_steps=3,
                     objective_reached=True)
        iterations = model.value_iteration()
        self.assertLess(iterations, 10000)
        self.assertEqual(model.infer(s0).attacker_action, "credential_access")
        events = ([{"tick": i, "host": "h1", "type": "process", "confidence": .9} for i in range(3)] +
                  [{"tick": 1, "host": "h1", "type": "network", "confidence": .9}])
        decision = DCAResponseAdapter(path_model=model).act(events)
        self.assertEqual(decision.inferred_attacker_action, "credential_access")


class TestPriorRL(unittest.TestCase):
    def test_first_action_frequency_prior(self):
        plans = np.array([[0,0,0,0], [1,0,0,0], [1,2,3,0], [2,0,0,0], [3,0,0,0], [3,1,0,0]])
        prior = action_prior_from_plans(plans, np.ones(6))
        self.assertAlmostEqual(float(prior.sum()), 1.0, places=6)
        self.assertGreater(float(prior[1]), float(prior[0]))

    def test_forward_kl_direction_and_alpha_zero_identity(self):
        logits = torch.tensor([[2.0, 0.0, 0.0, 0.0]], requires_grad=True)
        prior = torch.tensor([[.1, .2, .3, .4]])
        got = forward_kl(logits, prior)
        pi = torch.softmax(logits, -1)
        expected = (pi * (torch.log(pi) - torch.log(prior))).sum(-1).mean()
        reverse = (prior * (torch.log(prior) - torch.log(pi))).sum(-1).mean()
        self.assertTrue(torch.allclose(got, expected))
        self.assertFalse(torch.allclose(got, reverse))
        base = torch.tensor(1.25)
        self.assertEqual(float(prior_regularized_ppo_loss(base, logits, prior, alpha_kl=0).detach()), 1.25)

    def test_seeded_initialization_is_repeatable_and_no_world_model_import(self):
        torch.manual_seed(51001); a = PriorRLActorCritic()
        torch.manual_seed(51001); b = PriorRLActorCritic()
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters())))
        source = Path(importlib.util.find_spec("baselines.priorrl_cc4.policy").origin).read_text()
        imports = [line for line in source.splitlines() if line.startswith(("import ", "from "))]
        self.assertFalse(any("world_model" in line for line in imports))
        with self.assertRaises(ValueError):
            PriorRLPPOConfig(world_model_allowed=True)

    def test_offline_cache_miss_fails_closed(self):
        adapter = OfflineActionPrior(lambda _state, **_kwargs: None)
        with self.assertRaisesRegex(RuntimeError, "fail closed"):
            adapter.load(np.zeros(27, dtype=np.float32), agent_name="blue_agent_0")

    def test_sharp_prior_gradient_moves_policy_toward_prior(self):
        logits = torch.zeros(1, 4, requires_grad=True)
        sharp = torch.tensor([[1e-4, 0.9997, 1e-4, 1e-4]])
        forward_kl(logits, sharp).backward()
        self.assertLess(float(logits.grad[0, 1]), 0.0)
        self.assertTrue(torch.all(logits.grad[0, [0, 2, 3]] > 0))

    def test_independent_policy_seeds_produce_different_initializations(self):
        torch.manual_seed(51001); first = PriorRLActorCritic()
        torch.manual_seed(51002); second = PriorRLActorCritic()
        self.assertTrue(any(not torch.equal(x, y) for x, y in zip(first.parameters(), second.parameters())))


class TestRSMBRL(unittest.TestCase):
    def test_root_mask_survives_floor_and_adversarial_invalid_scores(self):
        cem = CategoricalCEMOptimizer(CategoricalCEMConfig(
            horizon=4, n_actions=4, population_size=200, num_iterations=5,
            elite_ratio=.3, alpha=.1, prob_floor=.01, seed=7,
        ))
        seen = []
        def objective(plans, _iteration):
            seen.extend(plans[:, 0].tolist())
            # If sampled, forbidden actions would dominate every valid plan.
            return (plans[:, 0] * 1_000_000 + plans[:, 1:].sum(1)).float()
        result = cem.optimize(objective, root_action_mask=torch.tensor([True, False, False, False]))
        self.assertEqual(set(seen), {0})
        self.assertEqual(int(result.best_plan[0]), 0)
        torch.testing.assert_close(result.final_probs[0], torch.tensor([1.0, 0.0, 0.0, 0.0]))

    def test_beta_monotonic_and_exact_paper_objective(self):
        reward = torch.tensor([5.0, 5.0]); risk = torch.tensor([1.0, 3.0])
        low = risk_adjusted_score(reward, risk, beta=.1)
        high = risk_adjusted_score(reward, risk, beta=.5)
        self.assertTrue(torch.all(high <= low))
        torch.testing.assert_close(high, torch.tensor([4.5, 3.5]))

    def test_eval_normalizer_updates_forbidden(self):
        with self.assertRaises(ValueError):
            RSMBRLConfig(update_normalizer_during_evaluation=True)

    def test_seeded_planner_smoke_is_deterministic_and_first_action_only(self):
        class Evaluator:
            config = SimpleNamespace(horizon=4)
            ensemble_size = 5
            def evaluate(self, _state, plans, *, projection_context=None):
                plans = torch.as_tensor(plans)
                n = len(plans)
                rewards = (plans == 3).sum(1).float()
                states = torch.zeros(4, n, 5, 27)
                return SharedRolloutResult(states, rewards[:, None].expand(n, 5), rewards)

        def build(seed):
            cem = CategoricalCEMOptimizer(CategoricalCEMConfig(
                horizon=4, n_actions=4, population_size=200, num_iterations=5,
                elite_ratio=.3, alpha=.1, seed=seed))
            return RSMBRLPlanner(cem_optimizer=cem, rollout_evaluator=Evaluator(),
                                 uncertainty=UGUncertainty(UGUncertaintyConfig(device="cpu")))
        a = build(11).plan(np.zeros(27, dtype=np.float32))
        b = build(11).plan(np.zeros(27, dtype=np.float32))
        self.assertTrue(torch.equal(a.best_plan, b.best_plan))
        self.assertEqual(a.requested_action_id, int(a.best_plan[0]))
        self.assertEqual(a.method, "RSMBRL-CC4")


if __name__ == "__main__":
    unittest.main()
