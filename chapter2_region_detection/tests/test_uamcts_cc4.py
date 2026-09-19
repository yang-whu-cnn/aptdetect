import unittest
import importlib.util
from pathlib import Path

import numpy as np

from baselines.uamcts_cc4 import (ActionStatistics, D27ProgressEnsemble,
                                  OfflinePrior, ProductionGate, SearchNode,
                                  UAMCTSConfig, UAMCTSPlanner,
                                  UncertaintyTriple,
                                  duration_aware_shaped_reward, hybrid_score)


class FakeWM:
    def predict_ensemble(self, state, action):
        a = np.zeros(27); a[action] = .1
        return [state + a, state + 2 * a]


class FakeReward:
    def predict(self, _state, action, _next): return float(action)


def progress():
    return D27ProgressEnsemble([lambda x: .4 * x[:4].sum(),
                                lambda x: .4 * x[:4].sum() + .02])


def prior():
    return OfflinePrior(lambda _key: [.1, .2, .3, .4])


class TestUAMCTSCore(unittest.TestCase):
    def test_exact_paper_score_and_uncertainty_direction(self):
        got = hybrid_score(q=2, uncertainty=.5, prior=.25, node_visits=16,
                           action_visits=3, c_puct=2, rho=.7, c_u=.2)
        expected = 2 - .7 * .5 + 2 * .25 * 4 / 4 + .2 * .5
        self.assertAlmostEqual(got, expected)
        riskier = hybrid_score(q=2, uncertainty=.8, prior=.25, node_visits=16,
                               action_visits=3, c_puct=2, rho=.7, c_u=.2)
        self.assertLess(riskier, got)

    def test_node_backup_and_visit_selection(self):
        node = SearchNode((0.0,) * 27, actions={
            0: ActionStatistics(prior=.5), 1: ActionStatistics(prior=.5)})
        node.backup(0, 2); node.backup(0, 4); node.backup(1, 10)
        self.assertEqual((node.n, node.actions[0].n, node.actions[0].q), (3, 2, 3))
        chosen = max(node.actions, key=lambda a: (node.actions[a].n, node.actions[a].q, -a))
        self.assertEqual(chosen, 0)

    def test_beta_zero_is_base_reward_and_duration_is_used(self):
        self.assertEqual(duration_aware_shaped_reward(3, .2, .8, beta=0,
                                                       gamma=.9, duration=3), 3)
        self.assertAlmostEqual(duration_aware_shaped_reward(
            3, .2, .8, beta=.5, gamma=.9, duration=3), 3 + .5 * (.9**3 * .8 - .2))

    def test_planner_a4_availability_first_action_visit_and_same_seed(self):
        kwargs = dict(world_model=FakeWM(), reward_model=FakeReward(),
                      progress_model=progress(), prior_model=prior(),
                      config=UAMCTSConfig(simulations=12, horizon=2), seed=9)
        a = UAMCTSPlanner(**kwargs).plan(np.zeros(27), available_actions=[0, 2])
        b = UAMCTSPlanner(**kwargs).plan(np.zeros(27), available_actions=[0, 2])
        self.assertEqual(a, b)
        self.assertIn(a.action, (0, 2))
        self.assertEqual(a.root_visits[1], 0); self.assertEqual(a.root_visits[3], 0)
        self.assertEqual(a.action, max((0, 2), key=lambda x: (a.root_visits[x], a.root_q[x], -x)))

    def test_prior_miss_and_production_gate_fail_closed(self):
        with self.assertRaises(RuntimeError):
            OfflinePrior(lambda _key: None).probabilities(np.zeros(27), [0, 1])
        with self.assertRaises(RuntimeError):
            ProductionGate().require_ready()

    def test_d27_only_and_three_uncertainties_are_retained(self):
        with self.assertRaises(ValueError): progress().score(np.zeros(28))
        with self.assertRaises(ValueError):
            D27ProgressEnsemble([lambda _x: 1.1]).score(np.zeros(27))
        triple = UncertaintyTriple(.1, .2, .3)
        self.assertAlmostEqual(triple.weighted((1, 2, 3)), 1.4)

    def test_core_has_no_hidden_truth_or_online_llm_imports(self):
        root = Path(importlib.util.find_spec("baselines.uamcts_cc4").origin).parent
        source = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
        imports = [line.lower() for line in source.splitlines()
                   if line.startswith(("import ", "from "))]
        self.assertFalse(any("controller" in line or "red_agent" in line for line in imports))
        self.assertNotIn("ground_truth", "\n".join(imports))
        self.assertNotIn("openai", "\n".join(imports))


if __name__ == "__main__":
    unittest.main()
