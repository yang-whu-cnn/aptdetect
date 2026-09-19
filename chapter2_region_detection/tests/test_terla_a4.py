import importlib.util
from pathlib import Path
import unittest

import torch

from baselines.terla_a4 import (CC4HiddenRewardAdapter, HiddenRewardRecord, TERLAGraphBuilder, TERLAPolicy,
                                TERLAProductionGate, TERLARewardChannel,
                                TERLASharedController, resolve_terla_action)
from baselines.terla_a4.run_development import select_observable_action
from baselines.terla_a4.runtime import public_mission_phase
from baselines.terla_a4.training import policy_hash
from baselines.terla_a4.training import PPOConfig, TERLAPPOTrainer, Transition


def observation(order=("h_both", "h_process", "h_network", "h_clean")):
    rows = {
        "h_both": {"subnet": "s1", "malicious_process_event": 1,
                   "malicious_network_event": 1},
        "h_process": {"subnet": "s1", "malicious_process_event": 1,
                      "malicious_network_event": 0},
        "h_network": {"subnet": "s2", "malicious_process_event": 0,
                      "malicious_network_event": 1},
        "h_clean": {"subnet": "s2", "malicious_process_event": 0,
                    "malicious_network_event": 0},
    }
    return {key: rows[key] for key in order}


class TestTERLAA4(unittest.TestCase):
    def setUp(self): self.builder = TERLAGraphBuilder()

    def test_graph_schema_shapes_and_permutation_invariance(self):
        a = self.builder.build(observation(), mission_phase=1)
        b = self.builder.build(observation(tuple(reversed(tuple(observation())))), mission_phase=1)
        self.assertEqual(a.host_names, b.host_names)
        self.assertTrue(torch.equal(a.node_features["host"], b.node_features["host"]))
        self.assertEqual(a.node_features["mission"].shape, (1, 3))
        self.assertEqual(a.node_features["subnet"].shape, (2, 1))
        self.assertEqual(a.node_features["host"].shape, (4, 2))
        self.assertEqual(set(a.edges), {
            ("mission", "contains", "subnet"), ("subnet", "in_mission", "mission"),
            ("subnet", "contains", "host"), ("host", "in_subnet", "subnet")})

    def test_two_layer_policy_shape_and_graph_permutation(self):
        torch.manual_seed(7); policy = TERLAPolicy()
        a = self.builder.build(observation(), mission_phase=2)
        b = self.builder.build(observation(("h_clean", "h_network", "h_both", "h_process")), mission_phase=2)
        logits_a, value_a = policy(a); logits_b, value_b = policy(b)
        self.assertEqual(logits_a.shape, (4,)); self.assertEqual(value_a.shape, ())
        self.assertTrue(torch.allclose(logits_a, logits_b))
        self.assertTrue(torch.allclose(value_a, value_b))
        self.assertEqual(policy.hidden, 60)
        self.assertEqual(policy.trunk[0].out_features, 120)
        self.assertEqual(policy.trunk[2].out_features, 120)

    def test_shared_parameters_but_isolated_agent_context(self):
        controller = TERLASharedController()
        policy_id = id(controller.policy)
        controller.record_action("blue_agent_0", 1)
        self.assertEqual(id(controller.policy), policy_id)
        self.assertEqual(controller.context("blue_agent_0").last_action, 1)
        self.assertIsNone(controller.context("blue_agent_1").last_action)
        self.assertIsNot(controller.context("blue_agent_0"), controller.context("blue_agent_1"))

    def test_fixed_seed_initialization_is_repeatable(self):
        torch.manual_seed(51004); first = TERLAPolicy()
        torch.manual_seed(51004); second = TERLAPolicy()
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters())))

    def test_target_order_duration_and_sleep_fallback(self):
        graph = self.builder.build(observation(), mission_phase=0)
        analyse = resolve_terla_action(1, graph)
        remove = resolve_terla_action(2, graph)
        restore = resolve_terla_action(3, graph)
        self.assertEqual((analyse.target, analyse.duration), ("h_network", 2))
        self.assertEqual((remove.target, remove.duration), ("h_both", 3))
        self.assertEqual((restore.target, restore.duration), ("h_both", 5))
        clean = self.builder.build({"h": {}}, mission_phase=0)
        fallback = resolve_terla_action(3, clean)
        self.assertEqual((fallback.requested_action, fallback.executed_action,
                          fallback.target, fallback.duration), (3, 0, None, 1))

    def test_exact_hidden_reward_formula_and_change(self):
        previous = HiddenRewardRecord((1, 0), ((.5,), (.25, .25)), (False, True))
        current = HiddenRewardRecord((0, 0), ((0.0,), (.25,)), (False, True))
        self.assertEqual(TERLARewardChannel.health(previous), -2.5)
        self.assertEqual(TERLARewardChannel.health(current), -.5)
        self.assertEqual(TERLARewardChannel.transition_reward(previous, current), 2.0)
        with self.assertRaises(RuntimeError): TERLAProductionGate().require_ready()

    def test_exact_controller_adapter_with_fake_hidden_state(self):
        class Service:
            def __init__(self, reliability, active=True):
                self.active = active; self.reliability = reliability
            def get_service_reliability(self): return self.reliability
        class Session:
            def __init__(self, active): self.active = active
        from types import SimpleNamespace
        host = SimpleNamespace(
            sessions={"red_agent_0": [1, 2], "green_agent_0": [3]},
            services={"SSHD": Service(80), "OTSERVICE": Service(60)})
        state = SimpleNamespace(hosts={"h": host}, sessions={
            "red_agent_0": {1: Session(True), 2: Session(False)},
            "green_agent_0": {3: Session(True)}})
        record = CC4HiddenRewardAdapter.record(SimpleNamespace(state=state), ["h"])
        self.assertEqual(record.red_sessions_by_host, (1,))
        self.assertEqual(record.service_unreliability_by_host, ((.3,),))
        self.assertEqual(record.ot_host, (True,))
        self.assertAlmostEqual(TERLARewardChannel.health(record), -1.6)

    def test_hidden_truth_rejected_and_policy_has_no_reward_import(self):
        with self.assertRaises(ValueError):
            self.builder.build({"h": {"compromised": True}}, mission_phase=0)
        model_path = Path(importlib.util.find_spec("baselines.terla_a4.model").origin)
        graph_path = Path(importlib.util.find_spec("baselines.terla_a4.graph").origin)
        policy_source = model_path.read_text(encoding="utf-8") + graph_path.read_text(encoding="utf-8")
        self.assertNotIn("from .reward", policy_source)
        self.assertNotIn("red_session", model_path.read_text(encoding="utf-8").lower())

    def test_hidden_reward_perturbation_cannot_change_policy_action(self):
        torch.manual_seed(4); policy = TERLAPolicy()
        graph = self.builder.build(observation(), mission_phase=0)
        mask = torch.tensor([True, True, False, True])
        generator = torch.Generator().manual_seed(8)
        before = select_observable_action(policy, graph, mask, deterministic=True, generator=generator)[0]
        hidden_a = HiddenRewardRecord((0,), ((0.0,),), (False,))
        hidden_b = HiddenRewardRecord((999,), ((1.0,),), (True,))
        self.assertNotEqual(TERLARewardChannel.health(hidden_a), TERLARewardChannel.health(hidden_b))
        after = select_observable_action(policy, graph, mask, deterministic=True, generator=generator)[0]
        self.assertEqual(before, after)
        self.assertNotEqual(after, 2)

    def test_eval_forward_keeps_parameters_and_public_phase_clock(self):
        torch.manual_seed(9); policy = TERLAPolicy()
        graph = self.builder.build(observation(), mission_phase=0)
        before = policy_hash(policy)
        with torch.inference_mode(): policy(graph)
        self.assertEqual(before, policy_hash(policy))
        self.assertEqual([public_mission_phase(t, 100) for t in (0, 33, 34, 66, 67, 100)],
                         [0, 0, 1, 1, 2, 2])

    def test_graph_device_transfer_preserves_names_and_values(self):
        graph = self.builder.build(observation(), mission_phase=1)
        moved = graph.to("cpu")
        self.assertEqual(moved.host_names, graph.host_names)
        self.assertTrue(torch.equal(moved.host_scores, graph.host_scores))
        self.assertTrue(all(value.device.type == "cpu" for value in moved.node_features.values()))

    def test_ppo_reuses_stored_availability_mask(self):
        torch.manual_seed(31); policy = TERLAPolicy()
        graph = self.builder.build(observation(), mission_phase=1)
        logits, value = policy(graph)
        mask = torch.tensor([True, True, False, False])
        dist = torch.distributions.Categorical(
            logits=logits.masked_fill(~mask, torch.finfo(logits.dtype).min))
        row = Transition(graph, 1, float(dist.log_prob(torch.tensor(1)).detach()),
                         float(value.detach()), 1.0, 0.0, True, 2, "episode:agent",
                         (True, True, False, False))
        before = policy.actor.weight.detach().clone()
        trainer = TERLAPPOTrainer(policy, PPOConfig(epochs=1))
        trainer.optimize([row])
        # Masked actor rows cannot affect the sampled policy ratio or entropy.
        self.assertTrue(torch.equal(before[2:], policy.actor.weight.detach()[2:]))

    def test_episode_boundary_flush_and_persistent_buffer(self):
        policy = TERLAPolicy(); trainer = TERLAPPOTrainer(policy, PPOConfig(rollout=2))
        graph = self.builder.build(observation(), mission_phase=0)
        rows = [Transition(graph, 0, 0.0, 0.0, 0.0, 0.0, True, 1, f"t{i}")
                for i in range(3)]
        calls = []
        trainer.optimize = lambda batch: calls.append(len(batch)) or {
            "transitions": len(batch), "updates": len(batch), "mean_loss": 0.0}
        buffer_id = id(trainer.buffer)
        trainer.add(rows); reports = trainer.flush_episode()
        self.assertEqual([r["transitions"] for r in reports], [2, 1])
        self.assertEqual(calls, [2, 1]); self.assertEqual(trainer.buffer, [])
        self.assertEqual(id(trainer.buffer), buffer_id)


if __name__ == "__main__": unittest.main()
