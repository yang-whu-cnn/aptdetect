import unittest
from types import SimpleNamespace

import numpy as np
import torch

from baselines.ug_cem_apt.categorical_cem import CEMResult
from baselines.ug_cem_apt.normalizer_warmup import (
    UGNormalizerWarmup,
    UGNormalizerWarmupConfig,
    WarmupState,
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


class WarmupCEM:
    def __init__(self, *, device="cpu"):
        self.device = torch.device(device)
        self.cfg = SimpleNamespace(
            horizon=4,
            n_actions=4,
            num_iterations=1,
        )
        self._uniform = torch.full(
            (4, 4),
            0.25,
            dtype=torch.float32,
            device=self.device,
        )

    @property
    def uniform_probs(self):
        return self._uniform.clone()

    def optimize(self, objective_fn, initial_probs=None):
        plans = torch.tensor(
            [
                [0, 1, 2, 3],
                [1, 2, 3, 0],
                [2, 3, 0, 1],
                [3, 0, 1, 2],
            ],
            dtype=torch.long,
            device=self.device,
        )
        scores = objective_fn(plans, 0)
        best_score, best_idx = torch.max(scores, dim=0)
        return CEMResult(
            best_plan=plans[int(best_idx.item())].detach().clone(),
            best_score=float(best_score.item()),
            final_probs=self._uniform.clone(),
        )


class WarmupRollout:
    def __init__(self, *, device="cpu"):
        self.device = torch.device(device)
        self.ensemble_size = 5
        self.config = SimpleNamespace(horizon=4)
        self.calls = 0

    def evaluate(self, current_state, plans):
        self.calls += 1
        plans = torch.as_tensor(plans, dtype=torch.long, device=self.device)
        n = int(plans.shape[0])

        next_states = torch.zeros(
            (4, n, 5, FORMAL_STATE_DIM),
            dtype=torch.float32,
            device=self.device,
        )

        base = torch.as_tensor(
            current_state,
            dtype=torch.float32,
            device=self.device,
        )

        for h in range(4):
            for m in range(5):
                next_states[h, :, m, :] = base
                next_states[h, :, m, 0] += float(h + 1) * 0.1
                next_states[h, :, m, 1] += float(m) * 0.2
                next_states[h, :, m, 2] += plans[:, h].float() * 0.05

        expected = plans.float().sum(dim=1) * 0.01
        member_returns = expected.unsqueeze(1).expand(n, 5).clone()

        return SharedRolloutResult(
            next_states=next_states,
            member_returns=member_returns,
            expected_return=expected,
        )


def make_planner():
    return UGCEMPlanner(
        cem_optimizer=WarmupCEM(),
        rollout_evaluator=WarmupRollout(),
        uncertainty=UGUncertainty(UGUncertaintyConfig(device="cpu")),
        config=UGCEMPlannerConfig(
            beta=0.1,
            update_uncertainty_stats=True,
        ),
    )


def make_state(index=0):
    state = np.zeros(FORMAL_STATE_DIM, dtype=np.float32)
    state[0] = float(index) * 0.01
    state[3] = 1.0
    return state


def record(index, *, split="train", seed=None, agent="blue_agent_0"):
    if seed is None:
        seed = 1000 + index if split == "train" else 3000 + index
    return WarmupState(
        state=make_state(index),
        split=split,
        episode_seed=seed,
        agent_name=agent,
    )


class TestUGNormalizerWarmup(unittest.TestCase):

    def test_default_policy_is_100_calls_and_online_ema(self):
        cfg = UGNormalizerWarmupConfig()
        self.assertEqual(cfg.planner_calls, 100)
        self.assertFalse(cfg.freeze_after_warmup)

    def test_warmup_state_rejects_validation_and_test(self):
        for split in ("validation", "test"):
            with self.assertRaises(ValueError):
                WarmupState(
                    state=make_state(),
                    split=split,
                    episode_seed=2000 if split == "validation" else 4000,
                    agent_name="blue_agent_0",
                )

    def test_warmup_state_rejects_seed_split_mismatch_and_test_seed_disguise(self):
        with self.assertRaises(ValueError):
            record(0, split="train", seed=3000)
        with self.assertRaises(ValueError):
            record(0, split="calibration", seed=1000)
        with self.assertRaises(ValueError):
            record(0, split="train", seed=4000)
        with self.assertRaises(ValueError):
            record(0, split="calibration", seed=4000)

    def test_warmup_state_validates_shape_finite_and_integer_seed(self):
        with self.assertRaises(ValueError):
            WarmupState(
                state=np.zeros(26, dtype=np.float32),
                split="train",
                episode_seed=1000,
                agent_name="blue_agent_0",
            )

        bad = make_state()
        bad[0] = np.nan
        with self.assertRaises(ValueError):
            WarmupState(
                state=bad,
                split="train",
                episode_seed=1000,
                agent_name="blue_agent_0",
            )

        with self.assertRaises(ValueError):
            WarmupState(
                state=make_state(),
                split="train",
                episode_seed=1000.0,
                agent_name="blue_agent_0",
            )

    def test_requires_enough_states(self):
        runner = UGNormalizerWarmup(
            planner=make_planner(),
            config=UGNormalizerWarmupConfig(planner_calls=3),
        )
        with self.assertRaises(ValueError):
            runner.run([record(0), record(1)])

    def test_warmup_forces_uniform_start_every_call(self):
        planner = make_planner()
        runner = UGNormalizerWarmup(
            planner=planner,
            config=UGNormalizerWarmupConfig(planner_calls=4),
        )
        report = runner.run([record(i) for i in range(4)])
        self.assertTrue(report.all_warm_starts_disabled)
        self.assertIsNone(planner.warm_start_probs)
        self.assertEqual(report.planner_calls, 4)

    def test_train_and_calibration_provenance_counts(self):
        planner = make_planner()
        runner = UGNormalizerWarmup(
            planner=planner,
            config=UGNormalizerWarmupConfig(planner_calls=4),
        )
        report = runner.run(
            [
                record(0, split="train", seed=1000),
                record(1, split="calibration", seed=3000),
                record(2, split="train", seed=1001),
                record(3, split="calibration", seed=3001),
            ]
        )
        self.assertEqual(report.train_calls, 2)
        self.assertEqual(report.calibration_calls, 2)
        self.assertEqual(report.unique_episode_seeds, (1000, 1001, 3000, 3001))
        self.assertEqual(report.agent_name, "blue_agent_0")

    def test_snapshot_is_initialized_finite_and_has_formal_shapes(self):
        runner = UGNormalizerWarmup(
            planner=make_planner(),
            config=UGNormalizerWarmupConfig(planner_calls=3),
        )
        report = runner.run([record(i) for i in range(3)])
        self.assertTrue(report.finite)
        self.assertEqual(tuple(report.snapshot.obs_mean.shape), (27,))
        self.assertEqual(tuple(report.snapshot.obs_std.shape), (27,))
        self.assertEqual(tuple(report.snapshot.horizon_std.shape), (4,))
        self.assertTrue(torch.isfinite(report.snapshot.obs_mean).all())
        self.assertTrue(torch.isfinite(report.snapshot.obs_std).all())
        self.assertTrue(torch.isfinite(report.snapshot.horizon_std).all())

    def test_main_mode_continues_online_ema_after_warmup(self):
        planner = make_planner()
        runner = UGNormalizerWarmup(
            planner=planner,
            config=UGNormalizerWarmupConfig(
                planner_calls=3,
                freeze_after_warmup=False,
            ),
        )
        report = runner.run([record(i) for i in range(3)])
        self.assertTrue(report.online_updates_after_warmup)
        self.assertTrue(planner.update_uncertainty_stats)

    def test_freeze_sensitivity_disables_future_ema_without_deleting_stats(self):
        planner = make_planner()
        runner = UGNormalizerWarmup(
            planner=planner,
            config=UGNormalizerWarmupConfig(
                planner_calls=3,
                freeze_after_warmup=True,
            ),
        )
        report = runner.run([record(i) for i in range(3)])
        self.assertFalse(report.online_updates_after_warmup)
        self.assertFalse(planner.update_uncertainty_stats)
        self.assertIsNotNone(planner.uncertainty.obs_mean)
        self.assertIsNotNone(planner.uncertainty.obs_std)
        self.assertIsNotNone(planner.uncertainty.horizon_std)

        before_mean = planner.uncertainty.obs_mean.detach().clone()
        before_std = planner.uncertainty.obs_std.detach().clone()
        before_horizon = planner.uncertainty.horizon_std.detach().clone()

        planner.plan(make_state(10))

        torch.testing.assert_close(
            planner.uncertainty.obs_mean,
            before_mean,
        )
        torch.testing.assert_close(
            planner.uncertainty.obs_std,
            before_std,
        )
        torch.testing.assert_close(
            planner.uncertainty.horizon_std,
            before_horizon,
        )

    def test_only_first_configured_number_of_states_are_used(self):
        planner = make_planner()
        runner = UGNormalizerWarmup(
            planner=planner,
            config=UGNormalizerWarmupConfig(planner_calls=2),
        )
        report = runner.run(
            [
                record(0, seed=1000),
                record(1, seed=1001),
                record(31, seed=1031),
            ]
        )
        self.assertEqual(report.planner_calls, 2)
        self.assertEqual(report.unique_episode_seeds, (1000, 1001))
        self.assertEqual(planner.rollout_evaluator.calls, 2)

    def test_one_runner_rejects_mixed_agents(self):
        runner = UGNormalizerWarmup(
            planner=make_planner(),
            config=UGNormalizerWarmupConfig(planner_calls=3),
        )
        with self.assertRaises(ValueError):
            runner.run(
                [
                    record(0, agent="blue_agent_0"),
                    record(1, agent="blue_agent_1"),
                    record(2, agent="blue_agent_0"),
                ]
            )

    def test_normalizers_are_not_implicitly_shared_between_planners(self):
        planner_a = make_planner()
        planner_b = make_planner()

        runner_a = UGNormalizerWarmup(
            planner=planner_a,
            config=UGNormalizerWarmupConfig(planner_calls=2),
        )
        runner_b = UGNormalizerWarmup(
            planner=planner_b,
            config=UGNormalizerWarmupConfig(planner_calls=2),
        )

        report_a = runner_a.run([record(0), record(1)])
        report_b = runner_b.run([record(20), record(21)])

        self.assertIsNot(planner_a.uncertainty, planner_b.uncertainty)
        self.assertFalse(
            torch.equal(
                report_a.snapshot.obs_mean,
                report_b.snapshot.obs_mean,
            )
        )

    def test_invalid_config_and_item_type_rejected(self):
        with self.assertRaises(ValueError):
            UGNormalizerWarmupConfig(planner_calls=0)
        with self.assertRaises(TypeError):
            UGNormalizerWarmupConfig(freeze_after_warmup=1)
        with self.assertRaises(TypeError):
            UGNormalizerWarmup(planner=object())

        runner = UGNormalizerWarmup(
            planner=make_planner(),
            config=UGNormalizerWarmupConfig(planner_calls=1),
        )
        with self.assertRaises(TypeError):
            runner.run([object()])


if __name__ == "__main__":
    unittest.main()
