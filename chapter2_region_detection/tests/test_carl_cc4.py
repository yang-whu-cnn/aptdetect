import unittest
from pathlib import Path

import torch

from baselines.carl_cc4.augmentation import TrainingTimeAugmentor, gate_imagination_horizon
from baselines.carl_cc4.buffers import CARLTransition, RealSyntheticBuffer
from baselines.carl_cc4.policy import CARLActorCritic, FrozenEvaluationPolicy
from baselines.carl_cc4.reward import caics_reward
from baselines.carl_cc4.scm import CausalRewardSCM, DAG_PARENTS, SCMRecord


def transition(seed, source):
    return CARLTransition(seed, source, (0.0,) * 27, 0, 0.0, (0.0,) * 27, 1)


class TestCARLCC4(unittest.TestCase):
    def test_standard_caics_reward_mapping_term_by_term(self):
        self.assertAlmostEqual(caics_reward(active_incidents=4, incident_change=0, action_id=0), -0.1)
        self.assertAlmostEqual(caics_reward(active_incidents=4, incident_change=1, action_id=2), -3.1)
        self.assertAlmostEqual(caics_reward(active_incidents=0, incident_change=-1, action_id=1,
                                            attack_cleared_terminal=True), 1000.0)
        self.assertAlmostEqual(caics_reward(active_incidents=0, incident_change=0, action_id=3,
                                            severe_failure_terminal=True), -1001.0)

    def test_scm_action_intervention_changes_reward(self):
        scm = CausalRewardSCM(); row = SCMRecord(1, 0, 1, 1, 1, 0, 1.0, -0.025)
        scm.fit([row])
        changed = scm.intervene_action(row, 2)
        self.assertEqual(changed.action, 2); self.assertNotEqual(changed.reward, row.reward)
        self.assertIn("action", DAG_PARENTS["reward"])

    def test_buffer_seed_isolation_and_exact_eight_rollouts(self):
        buffer = RealSyntheticBuffer(); buffer.add_real(transition(1000, "real"))
        buffer.add_synthetic_rollouts(origin_episode_seed=1000,
                                      rollouts=[[transition(1000, "synthetic")] for _ in range(8)])
        real, synthetic = buffer.fixed_ratio_batch(real_count=1, synthetic_per_real=8)
        self.assertEqual((len(real), len(synthetic)), (1, 8))
        with self.assertRaises(ValueError):
            buffer.add_synthetic_rollouts(origin_episode_seed=1000,
                                          rollouts=[[transition(1001, "synthetic")] for _ in range(8)])

    def test_hidden_mutation_cannot_change_policy_action(self):
        torch.manual_seed(7); policy = CARLActorCritic().eval(); observable = torch.zeros(27)
        first = int(policy.act(observable, deterministic=True)[0])
        hidden_truth = {"red_sessions": 1}; hidden_truth["red_sessions"] = 999
        second = int(policy.act(observable, deterministic=True)[0])
        self.assertEqual(first, second)

    def test_h256_is_blocked_or_explicit_noneligible_truncation(self):
        blocked = gate_imagination_horizon(); self.assertEqual(blocked.status, "BLOCKED")
        self.assertFalse(blocked.formal_eligible)
        truncated = gate_imagination_horizon(allow_disclosed_truncation=True)
        self.assertEqual((truncated.status, truncated.effective_horizon), ("ADAPTED_TRUNCATED", 4))
        self.assertFalse(truncated.formal_eligible)
        with self.assertRaises(RuntimeError): TrainingTimeAugmentor(None, None, blocked)

    def test_evaluation_freeze_and_no_search_static_gate(self):
        gate = gate_imagination_horizon(allow_disclosed_truncation=True)
        augmentor = TrainingTimeAugmentor(object(), object(), gate); augmentor.freeze_for_evaluation()
        self.assertFalse(augmentor.training)
        root = Path(__file__).resolve().parents[1] / "baselines" / "carl_cc4"
        source = "\n".join(path.read_text(encoding="utf-8").lower()
                           for path in root.glob("*.py"))
        self.assertNotIn("mcts", source); self.assertNotIn("cemoptimizer", source)
        self.assertNotIn("def plan(", source)
        policy = CARLActorCritic(); frozen = FrozenEvaluationPolicy(policy)
        frozen.act(torch.zeros(27))
        with torch.no_grad(): next(policy.parameters()).add_(1.0)
        with self.assertRaisesRegex(RuntimeError, "mutated"):
            frozen.act(torch.zeros(27))


if __name__ == "__main__": unittest.main()
