import unittest

import numpy as np
import torch

from formal_experiments.ours.run_table2_real_smoke import build_runtime
from formal_experiments.ours.table2_variants import Table2DecisionRuntime, Table2Variant
from formal_experiments.ours.variant_training import VariantPPOBuffer, VariantTransition, ppo_update


class TestTable2RealSmokeContracts(unittest.TestCase):
    def transition(self, runtime, *, split="train", duration=1, mask=None):
        mask = mask or [True, True, False, True]
        decision = runtime.decide(
            np.zeros(27, dtype=np.float32), agent_name="blue_agent_0",
            action_mask=mask, deterministic=True,
        )
        return VariantTransition(
            agent_name="blue_agent_0", policy_input=decision["policy_input"], policy_mask=decision["policy_mask"],
            policy_action=decision["policy_action"], old_log_prob=decision["log_prob"],
            value=decision["value"], next_value=0.0, reward=1.0,
            duration=duration, done=False, split=split,
        )

    def test_duration_aware_masked_ppo_updates_train_policy(self):
        runtime = Table2DecisionRuntime(variant=Table2Variant.RL_ONLY, init_seed=7)
        buffer = VariantPPOBuffer(split="train")
        buffer.append(self.transition(runtime, duration=3))
        before = [p.detach().clone() for p in runtime.policy.parameters()]
        result = ppo_update(runtime, buffer)
        self.assertEqual(result["duration_counts"], {3: 1})
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, runtime.policy.parameters())))

    def test_ppo_rejects_split_contamination_and_eval_update(self):
        runtime = Table2DecisionRuntime(variant=Table2Variant.RL_ONLY, init_seed=8)
        validation = VariantPPOBuffer(split="validation")
        with self.assertRaisesRegex(ValueError, "contamination"):
            validation.append(self.transition(runtime, split="train"))
        validation.append(self.transition(runtime, split="validation"))
        with self.assertRaisesRegex(RuntimeError, "train-split"):
            ppo_update(runtime, validation)

    def test_ppo_rejects_action_excluded_by_stored_mask(self):
        runtime = Table2DecisionRuntime(variant=Table2Variant.RL_ONLY, init_seed=9)
        item = self.transition(runtime)
        bad = VariantTransition(**{**item.__dict__, "policy_mask": torch.zeros(4, dtype=torch.bool)})
        buffer = VariantPPOBuffer(split="train"); buffer.append(bad)
        with self.assertRaisesRegex(ValueError, "action mask"):
            ppo_update(runtime, buffer)

    def test_gae_does_not_propagate_between_agents(self):
        runtime = Table2DecisionRuntime(variant=Table2Variant.RL_ONLY, init_seed=10)
        first = self.transition(runtime); second = self.transition(runtime)
        first = VariantTransition(**{**first.__dict__, "agent_name": "blue_agent_0", "reward": 0.0})
        second = VariantTransition(**{**second.__dict__, "agent_name": "blue_agent_1", "reward": 1000.0})
        buffer = VariantPPOBuffer(split="train"); buffer.append(first); buffer.append(second)
        result = ppo_update(runtime, buffer)
        self.assertAlmostEqual(result["return_abs_max_by_agent"]["blue_agent_0"], 0.0, places=6)

    def test_llm_variants_fail_closed_without_auditable_k6_h4_cache(self):
        for variant in (Table2Variant.LLM_RL, Table2Variant.LWM_RL):
            with self.subTest(variant=variant.value), self.assertRaisesRegex(RuntimeError, "BLOCKED.*offline prior"):
                build_runtime(variant, seed=51001, device="cpu",
                              offline_prior_artifact=__import__("pathlib").Path("missing-k6h4.json"))


if __name__ == "__main__":
    unittest.main()
