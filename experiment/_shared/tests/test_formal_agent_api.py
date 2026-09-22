import unittest

from formal_experiments.common.agent_api import ActionDecision, FixedActionAgent, RandomActionAgent
from formal_experiments.evaluation.run_formal_suite import failure_owner_from_inventories


class TestFormalAgentApi(unittest.TestCase):
    def test_fixed_agent_falls_back_to_sleep_on_masked_action(self):
        agent = FixedActionAgent(2)
        agent.reset(episode_seed=1, agent_id="blue_agent_0")
        decision = agent.act(blue_observation={}, tick=0, action_mask=(True, True, False, True))
        self.assertEqual(decision.requested_action, 2)
        self.assertEqual(decision.executed_action, 0)
        self.assertTrue(decision.fallback)

    def test_random_agent_is_seeded_and_mask_respecting(self):
        one, two = RandomActionAgent(), RandomActionAgent()
        for agent in (one, two):
            agent.reset(episode_seed=42, agent_id="blue_agent_2")
        seq_one = [one.act(blue_observation={}, tick=i, action_mask=(True, False, True, False)).executed_action for i in range(20)]
        seq_two = [two.act(blue_observation={}, tick=i, action_mask=(True, False, True, False)).executed_action for i in range(20)]
        self.assertEqual(seq_one, seq_two)
        self.assertTrue(set(seq_one) <= {0, 2})

    def test_action_decision_rejects_non_a4(self):
        with self.assertRaises(ValueError):
            ActionDecision("bad", 4, 0, None, False, None, 0.0)

    def test_failure_owner_uses_blue_inventory_membership(self):
        inventories = {"blue_agent_0": ("host-a",), "blue_agent_1": ("host-b",)}
        self.assertEqual(failure_owner_from_inventories("host-a", inventories), "blue_agent_0")
        self.assertIsNone(failure_owner_from_inventories("contractor_network", inventories))

    def test_failure_owner_fails_closed_on_overlapping_inventory(self):
        inventories = {"blue_agent_0": ("host-a",), "blue_agent_1": ("host-a",)}
        with self.assertRaisesRegex(RuntimeError, "multiple Blue inventories"):
            failure_owner_from_inventories("host-a", inventories)


if __name__ == "__main__":
    unittest.main()
