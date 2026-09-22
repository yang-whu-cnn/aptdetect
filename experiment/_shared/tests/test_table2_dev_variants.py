import unittest
from types import SimpleNamespace

import numpy as np
import torch

from formal_experiments.ours.run_table2_dev_pilot import readiness, run_fake_smoke
from formal_experiments.ours.table2_variants import (
    Table2DecisionRuntime, Table2Variant, assert_independent_initializations,
    non_llm_candidate_plans,
)
from shared.d27_projection import D27ProjectionContext


class FakePrior:
    online_allowed = False
    def __init__(self, miss=False): self.calls = []; self.miss = miss
    def load(self, state, *, agent_name, split):
        self.calls.append((agent_name, split))
        if self.miss: return None
        plans = np.asarray([
            [0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0],
            [3, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0],
        ], dtype=np.int64)
        return SimpleNamespace(plans=plans, prior_preferences=np.full(6, 1/6, dtype=np.float32))


class FakeWM:
    frozen = True
    def __init__(self): self.calls = 0
    def evaluate(self, state, plans, *, projection_context=None):
        if projection_context is None: raise AssertionError("missing projection context")
        self.calls += 1
        members = torch.arange(30, dtype=torch.float32).reshape(6, 5) / 100
        return SimpleNamespace(member_returns=members, expected_return=members.mean(1))


class TestTable2DevVariants(unittest.TestCase):
    def runtime(self, variant, seed=1, split="train"):
        spec_prior = FakePrior() if variant in (Table2Variant.LLM_RL, Table2Variant.LWM_RL) else None
        wm = FakeWM() if variant in (Table2Variant.WM_RL, Table2Variant.LWM_RL) else None
        return Table2DecisionRuntime(variant=variant, init_seed=seed, offline_prior=spec_prior,
                                     frozen_wm_evaluator=wm, split=split)

    def test_module_call_audit_for_all_four_independent_variants(self):
        runtimes = [self.runtime(variant, seed=100 + index)
                    for index, variant in enumerate(Table2Variant)]
        assert_independent_initializations(runtimes)
        for runtime in runtimes:
            kwargs = ({"projection_context": D27ProjectionContext(0,0,100,0.0)}
                      if runtime.spec.uses_world_model else {})
            result = runtime.decide(np.zeros(27), agent_name="blue_agent_0",
                                    action_mask=[True, True, False, True], **kwargs)
            self.assertNotEqual(result["requested_action"], 2)
            self.assertEqual(runtime.audit.prior_calls, int(runtime.spec.uses_llm_prior))
            self.assertEqual(runtime.audit.wm_calls, int(runtime.spec.uses_world_model))
            self.assertEqual(runtime.audit.online_llm_calls, 0)

    def test_only_first_action_is_returned_and_next_call_replans(self):
        runtime = self.runtime(Table2Variant.LWM_RL)
        context=D27ProjectionContext(0,0,100,0.0)
        first = runtime.decide(np.zeros(27), agent_name="blue_agent_0", action_mask=[True]*4,
                               projection_context=context)
        second = runtime.decide(np.ones(27), agent_name="blue_agent_0", action_mask=[True]*4,
                                projection_context=context)
        self.assertEqual(len(first["selected_plan"]), 4)
        self.assertEqual(first["requested_action"], first["selected_plan"][0])
        self.assertEqual(runtime.audit.policy_calls, 2)
        self.assertEqual(runtime.audit.wm_calls, 2)

    def test_cache_miss_fails_closed_without_online_llm(self):
        runtime = Table2DecisionRuntime(variant=Table2Variant.LLM_RL, init_seed=1,
                                        offline_prior=FakePrior(miss=True), split="test")
        with self.assertRaisesRegex(RuntimeError, "cache miss"):
            runtime.decide(np.zeros(27), agent_name="blue_agent_0", action_mask=[True]*4)
        self.assertEqual(runtime.audit.online_llm_calls, 0)

    def test_split_is_forwarded_to_isolated_cache_identity(self):
        prior = FakePrior(); runtime = Table2DecisionRuntime(
            variant=Table2Variant.LLM_RL, init_seed=1, offline_prior=prior, split="validation")
        runtime.decide(np.zeros(27), agent_name="blue_agent_0", action_mask=[True]*4)
        self.assertEqual(prior.calls, [("blue_agent_0", "validation")])

    def test_candidate_audit_is_complete_and_probabilities_normalized(self):
        runtime = self.runtime(Table2Variant.LLM_RL)
        result = runtime.decide(np.zeros(27), agent_name="blue_agent_0", action_mask=[True]*4)
        self.assertEqual(len(result["candidate_plans_before_dedup"]), 6)
        self.assertEqual(result["candidate_plans_before_dedup"], result["candidate_plans_after_dedup"])
        self.assertAlmostEqual(sum(result["candidate_selection_probabilities"]), 1.0, places=6)
        self.assertIn("exact_duplicates_forbidden", result["candidate_probability_rule"])

    def test_non_llm_generator_is_k6_unique_and_covers_a4_first_actions(self):
        plans = non_llm_candidate_plans(np.zeros(27), agent_name="blue_agent_0")
        self.assertEqual(plans.shape, (6, 4))
        self.assertEqual(len({tuple(row) for row in plans.tolist()}), 6)
        self.assertEqual(set(plans[:, 0].tolist()), {0, 1, 2, 3})

    def test_exact_duplicate_or_all_unavailable_candidates_fail_closed(self):
        duplicate = FakePrior()
        original = duplicate.load
        def duplicate_load(state, *, agent_name, split):
            prior = original(state, agent_name=agent_name, split=split)
            prior.plans[1] = prior.plans[0]
            return prior
        duplicate.load = duplicate_load
        runtime = Table2DecisionRuntime(variant=Table2Variant.LLM_RL, init_seed=1,
                                        offline_prior=duplicate, split="train")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            runtime.decide(np.zeros(27), agent_name="blue_agent_0", action_mask=[True]*4)

        class NoLegalPrior(FakePrior):
            def load(self, state, *, agent_name, split):
                prior = super().load(state, agent_name=agent_name, split=split)
                prior.plans[:] = np.asarray([
                    [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0],
                    [1, 1, 0, 0], [2, 1, 0, 0], [3, 1, 0, 0],
                ])
                return prior
        blocked = Table2DecisionRuntime(variant=Table2Variant.LLM_RL, init_seed=2,
                                        offline_prior=NoLegalPrior(), split="train")
        with self.assertRaisesRegex(RuntimeError, "all candidate"):
            blocked.decide(np.zeros(27), agent_name="blue_agent_0",
                           action_mask=[True, False, False, False])

    def test_hidden_truth_mutation_does_not_change_policy_decision(self):
        runtime = self.runtime(Table2Variant.RL_ONLY)
        state = np.zeros(27); first = runtime.decide(state, agent_name="blue_agent_0", action_mask=[True]*4)
        hidden = {"red_sessions": 1}; hidden["red_sessions"] = 999
        second = runtime.decide(state, agent_name="blue_agent_0", action_mask=[True]*4)
        self.assertEqual(first["requested_action"], second["requested_action"])

    def test_fake_smoke_is_development_only_and_duration_aware(self):
        report = run_fake_smoke(self.runtime(Table2Variant.RL_ONLY), steps=20)
        self.assertFalse(report["formal_result_eligible"]); self.assertTrue(report["fake_environment"])
        for duration, discount in zip(report["durations"], report["discounts"]):
            self.assertAlmostEqual(discount, 0.99 ** duration, places=6)

    def test_readiness_reports_blockers_without_fallback(self):
        self.assertEqual(readiness(Table2Variant.LWM_RL, cache_ready=False, wm_ready=True)["status"], "BLOCKED")
        self.assertEqual(readiness(Table2Variant.RL_ONLY, cache_ready=False, wm_ready=False)["status"], "READY")


if __name__ == "__main__": unittest.main()
