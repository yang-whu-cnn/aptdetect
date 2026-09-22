import math
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from baselines.ug_cem_apt.categorical_cem import (
    CategoricalCEMConfig,
    CategoricalCEMOptimizer,
    CEMResult,
)
from baselines.ug_cem_apt.planner import (
    UGCEMPlanner,
    UGCEMPlannerConfig,
)
from baselines.ug_cem_apt.uncertainty import (
    UGUncertainty,
    UGUncertaintyConfig,
)
from shared.formal_state import FORMAL_STATE_DIM
from shared.rollout_evaluator import SharedRolloutResult


def state():
    return np.zeros(FORMAL_STATE_DIM, dtype=np.float32)


class FakeRolloutEvaluator:
    def __init__(
        self,
        *,
        device="cpu",
        ensemble_size=5,
        horizon=4,
        mode="first_action",
    ):
        self.device = torch.device(device)
        self.ensemble_size = int(ensemble_size)
        self.config = SimpleNamespace(horizon=int(horizon))
        self.mode = mode
        self.calls = 0

    def evaluate(self, current_state, plans):
        self.calls += 1
        plans = torch.as_tensor(plans, dtype=torch.long, device=self.device)
        n = int(plans.shape[0])

        next_states = torch.zeros(
            (4, n, self.ensemble_size, FORMAL_STATE_DIM),
            dtype=torch.float32,
            device=self.device,
        )

        code = plans[:, 0].to(dtype=torch.float32)
        next_states[:, :, :, 0] = code.view(1, n, 1)

        if self.mode == "first_action":
            expected = 10.0 + code
        elif self.mode == "count_restore":
            expected = (plans == 3).sum(dim=1).to(dtype=torch.float32)
        else:
            raise RuntimeError("unknown fake rollout mode")

        member_returns = expected.unsqueeze(1).expand(n, self.ensemble_size).clone()

        return SharedRolloutResult(
            next_states=next_states,
            member_returns=member_returns,
            expected_return=expected,
        )


class ScriptedUncertainty(UGUncertainty):
    def __init__(self, *, device="cpu", scale=1.0):
        super().__init__(UGUncertaintyConfig(device=device))
        self.scale = float(scale)
        self.update_flags = []
        self.reset_calls = 0

    def compute(self, next_states, update_stats=True):
        self.update_flags.append(bool(update_stats))
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        code = next_states[0, :, 0, 0]
        return self.scale * (2.0 + code)

    def reset(self):
        self.reset_calls += 1
        super().reset()


class ScriptedCEM:
    def __init__(
        self,
        populations,
        *,
        final_probs=None,
        device="cpu",
        horizon=4,
        n_actions=4,
    ):
        self.device = torch.device(device)
        self.cfg = SimpleNamespace(
            horizon=int(horizon),
            n_actions=int(n_actions),
            num_iterations=len(populations),
        )
        self.populations = [
            torch.as_tensor(p, dtype=torch.long, device=self.device)
            for p in populations
        ]
        self.initial_probs_seen = []
        self.scores_seen = []
        self.optimize_calls = 0

        self._uniform = torch.full(
            (int(horizon), int(n_actions)),
            1.0 / float(n_actions),
            dtype=torch.float32,
            device=self.device,
        )

        if final_probs is None:
            final_probs = self._uniform.clone()
        self.final_probs = torch.as_tensor(
            final_probs,
            dtype=torch.float32,
            device=self.device,
        )

    @property
    def uniform_probs(self):
        return self._uniform.clone()

    def optimize(self, objective_fn, initial_probs=None):
        self.optimize_calls += 1
        self.initial_probs_seen.append(
            None if initial_probs is None else initial_probs.detach().clone()
        )

        best_plan = None
        best_score = float("-inf")

        for iteration, plans in enumerate(self.populations):
            scores = torch.as_tensor(
                objective_fn(plans, iteration),
                dtype=torch.float32,
                device=self.device,
            )
            self.scores_seen.append(scores.detach().clone())
            score_t, idx_t = torch.max(scores, dim=0)
            score = float(score_t.item())
            if score > best_score:
                best_score = score
                best_plan = plans[int(idx_t.item())].detach().clone()

        return CEMResult(
            best_plan=best_plan,
            best_score=best_score,
            final_probs=self.final_probs.detach().clone(),
        )


class TestUGCEMPlanner(unittest.TestCase):

    def make_scripted(
        self,
        *,
        beta=0.5,
        update_stats=True,
        final_probs=None,
        populations=None,
    ):
        if populations is None:
            populations = [
                [[0, 0, 0, 0], [3, 0, 0, 0]],
                [[1, 0, 0, 0], [2, 0, 0, 0]],
            ]
        cem = ScriptedCEM(populations, final_probs=final_probs)
        rollout = FakeRolloutEvaluator()
        uncertainty = ScriptedUncertainty()
        planner = UGCEMPlanner(
            cem_optimizer=cem,
            rollout_evaluator=rollout,
            uncertainty=uncertainty,
            config=UGCEMPlannerConfig(
                beta=beta,
                update_uncertainty_stats=update_stats,
            ),
        )
        return planner, cem, rollout, uncertainty

    def test_score_formula_keeps_iteration_plus_one(self):
        planner, cem, _, _ = self.make_scripted(beta=0.5)
        planner.plan(state())

        # iteration 0:
        # a0=0 -> G=10, w=2 => 9
        # a0=3 -> G=13, w=5 => 10.5
        np.testing.assert_allclose(
            cem.scores_seen[0].cpu().numpy(),
            np.asarray([9.0, 10.5], dtype=np.float32),
        )

        # iteration 1 denominator must be 2:
        # a0=1 -> 11 - .5*3/2 = 10.25
        # a0=2 -> 12 - .5*4/2 = 11
        np.testing.assert_allclose(
            cem.scores_seen[1].cpu().numpy(),
            np.asarray([10.25, 11.0], dtype=np.float32),
        )

    def test_beta_zero_is_cem_apt_same_machinery(self):
        planner, cem, rollout, uncertainty = self.make_scripted(beta=0.0)
        result = planner.plan(state())
        np.testing.assert_allclose(
            cem.scores_seen[0].cpu().numpy(),
            np.asarray([10.0, 13.0], dtype=np.float32),
        )
        self.assertEqual(rollout.calls, 2)
        self.assertEqual(len(uncertainty.update_flags), 2)
        self.assertGreaterEqual(result.action_id, 0)

    def test_returns_best_sampled_plan_and_first_action(self):
        planner, _, _, _ = self.make_scripted(beta=0.5)
        result = planner.plan(state())
        self.assertTrue(torch.equal(result.best_plan, torch.tensor([2, 0, 0, 0])))
        self.assertEqual(result.action_id, 2)
        self.assertAlmostEqual(result.best_score, 11.0, places=6)
        self.assertAlmostEqual(result.expected_return, 12.0, places=6)
        self.assertAlmostEqual(result.uncertainty, 4.0, places=6)

    def test_mpc_warm_start_shifts_final_probs_and_appends_uniform(self):
        final_probs = np.asarray(
            [
                [0.10, 0.20, 0.30, 0.40],
                [0.40, 0.30, 0.20, 0.10],
                [0.25, 0.25, 0.25, 0.25],
                [0.55, 0.15, 0.15, 0.15],
            ],
            dtype=np.float32,
        )
        planner, cem, _, _ = self.make_scripted(final_probs=final_probs)

        first = planner.plan(state())
        self.assertFalse(first.warm_start_used)
        self.assertIsNone(cem.initial_probs_seen[0])

        expected_shift = np.vstack(
            [final_probs[1:], np.full((1, 4), 0.25, dtype=np.float32)]
        )
        np.testing.assert_allclose(
            planner.warm_start_probs.cpu().numpy(),
            expected_shift,
            rtol=0,
            atol=0,
        )

        second = planner.plan(state())
        self.assertTrue(second.warm_start_used)
        np.testing.assert_allclose(
            cem.initial_probs_seen[1].cpu().numpy(),
            expected_shift,
            rtol=0,
            atol=0,
        )

    def test_reset_episode_clears_only_warm_start(self):
        planner, _, _, uncertainty = self.make_scripted()
        planner.plan(state())
        self.assertIsNotNone(planner.warm_start_probs)
        self.assertEqual(uncertainty.reset_calls, 0)

        planner.reset_episode()
        self.assertIsNone(planner.warm_start_probs)
        self.assertEqual(uncertainty.reset_calls, 0)

    def test_reset_uncertainty_is_explicit(self):
        planner, _, _, uncertainty = self.make_scripted()
        planner.reset_uncertainty_stats()
        self.assertEqual(uncertainty.reset_calls, 1)

    def test_every_plan_call_replans(self):
        planner, cem, rollout, _ = self.make_scripted()
        planner.plan(state())
        planner.plan(state())
        self.assertEqual(cem.optimize_calls, 2)
        self.assertEqual(rollout.calls, 4)

    def test_iteration_history_and_required_debug_fields(self):
        planner, _, _, _ = self.make_scripted()
        result = planner.plan(state())

        self.assertEqual(len(result.iteration_history), 2)
        self.assertEqual([x.iteration for x in result.iteration_history], [0, 1])
        self.assertTrue(math.isfinite(result.best_score))
        self.assertTrue(math.isfinite(result.expected_return))
        self.assertTrue(math.isfinite(result.uncertainty))
        self.assertTrue(math.isfinite(result.final_probs_entropy))
        self.assertTrue(math.isfinite(result.planning_latency_sec))
        self.assertGreaterEqual(result.planning_latency_sec, 0.0)
        self.assertEqual(tuple(result.final_probs_entropy_by_step.shape), (4,))

    def test_entropy_matches_uniform_reference(self):
        uniform = np.full((4, 4), 0.25, dtype=np.float32)
        planner, _, _, _ = self.make_scripted(final_probs=uniform)
        result = planner.plan(state())
        self.assertAlmostEqual(result.final_probs_entropy, math.log(4.0), places=6)

    def test_update_uncertainty_stats_flag_is_forwarded(self):
        planner, _, _, uncertainty = self.make_scripted(update_stats=False)
        planner.plan(state())
        self.assertEqual(uncertainty.update_flags, [False, False])

    def test_invalid_beta_rejected(self):
        for beta in (-0.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                UGCEMPlannerConfig(beta=beta)

    def test_component_contract_guards(self):
        rollout = FakeRolloutEvaluator()
        uncertainty = ScriptedUncertainty()

        with self.assertRaises(ValueError):
            UGCEMPlanner(
                cem_optimizer=ScriptedCEM([[[0, 0, 0, 0]]], n_actions=5),
                rollout_evaluator=rollout,
                uncertainty=uncertainty,
            )

        with self.assertRaises(ValueError):
            UGCEMPlanner(
                cem_optimizer=ScriptedCEM([[[0, 0, 0]]], horizon=3),
                rollout_evaluator=rollout,
                uncertainty=uncertainty,
            )

        with self.assertRaises(ValueError):
            UGCEMPlanner(
                cem_optimizer=ScriptedCEM([[[0, 0, 0, 0]]]),
                rollout_evaluator=FakeRolloutEvaluator(ensemble_size=4),
                uncertainty=uncertainty,
            )

        meta_uncertainty = ScriptedUncertainty(device="meta")
        with self.assertRaises(ValueError):
            UGCEMPlanner(
                cem_optimizer=ScriptedCEM([[[0, 0, 0, 0]]]),
                rollout_evaluator=rollout,
                uncertainty=meta_uncertainty,
            )

    def test_real_categorical_cem_integration_smoke(self):
        cem = CategoricalCEMOptimizer(
            CategoricalCEMConfig(
                horizon=4,
                n_actions=4,
                population_size=32,
                num_iterations=3,
                elite_ratio=0.25,
                alpha=0.10,
                prob_floor=0.01,
                seed=7,
                device="cpu",
            )
        )
        rollout = FakeRolloutEvaluator(mode="count_restore")
        uncertainty = UGUncertainty(UGUncertaintyConfig(device="cpu"))
        planner = UGCEMPlanner(
            cem_optimizer=cem,
            rollout_evaluator=rollout,
            uncertainty=uncertainty,
            config=UGCEMPlannerConfig(beta=0.1),
        )

        result = planner.plan(state())
        self.assertEqual(tuple(result.best_plan.shape), (4,))
        self.assertTrue(0 <= result.action_id < 4)
        self.assertEqual(len(result.iteration_history), 3)
        self.assertTrue(torch.isfinite(result.final_probs).all())
        self.assertEqual(tuple(result.final_probs.shape), (4, 4))

    def test_warm_start_shape_and_finite_guard(self):
        planner, _, _, _ = self.make_scripted()

        with self.assertRaises(ValueError):
            planner._shift_final_probs(torch.ones(3, 4))

        bad = torch.full((4, 4), 0.25)
        bad[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            planner._shift_final_probs(bad)


if __name__ == "__main__":
    unittest.main()
