import unittest
from types import SimpleNamespace

import numpy as np
import torch

from shared.action_contract import ACTION_CONTRACTS
from shared.formal_state import FORMAL_STATE_DIM
from shared.model_space_action import (
    ANY_TARGET_INDEX,
    canonicalize_requested_action,
)
from shared.rollout_evaluator import (
    SharedRolloutConfig,
    SharedRolloutEvaluator,
)
from shared.d27_projection import D27ProjectionContext


DURATIONS = {
    int(item.action_id): int(item.duration_ticks)
    for item in ACTION_CONTRACTS
}


class FakeWorldModel:
    def __init__(
        self,
        *,
        ensemble_size=3,
        device="cpu",
        target_mode="absolute",
        nonfinite=False,
    ):
        self.device = torch.device(device)
        self.config = SimpleNamespace(
            state_dim=FORMAL_STATE_DIM,
            n_actions=4,
            ensemble_size=int(ensemble_size),
            hidden_dim=128,
            target_mode=target_mode,
        )
        self.models = [object() for _ in range(int(ensemble_size))]
        self.calls = []
        self.nonfinite = bool(nonfinite)

    def predict_member_distribution_tensor(self, *args, **kwargs):
        raise AssertionError("aleatoric/distribution path must not be used")

    def predict_member_mean_tensor(self, member_idx, states, actions):
        self.calls.append((int(member_idx), int(states.shape[0])))
        out = states.clone()
        out[:, 0] = out[:, 0] + float(member_idx + 1)
        out[:, 1] = out[:, 1] + actions.to(dtype=torch.float32)
        if self.nonfinite:
            out[:, 2] = float("inf")
        return out


class FakeRewardPredictor:
    def __init__(self, *, device="cpu", state_dependent=False, nonfinite=False):
        self.device = torch.device(device)
        self.config = SimpleNamespace(
            state_dim=FORMAL_STATE_DIM,
            n_actions=4,
            hidden_dim=128,
        )
        self.calls = []
        self.state_dependent = bool(state_dependent)
        self.nonfinite = bool(nonfinite)

    def predict_tensor(self, states, actions, next_states):
        self.calls.append(int(states.shape[0]))
        reward = actions.to(dtype=torch.float32) + 1.0
        if self.state_dependent:
            reward = reward + 0.01 * next_states[:, 0]
        if self.nonfinite:
            reward = reward.clone()
            reward[0] = float("nan")
        return reward


def base_state(*, available=True):
    state = np.zeros(FORMAL_STATE_DIM, dtype=np.float32)
    state[ANY_TARGET_INDEX] = 1.0 if available else 0.0
    return state


def slow_reference(state, plans, *, members, gamma_tick, state_dependent=False):
    plans = np.asarray(plans, dtype=np.int64)
    n_plans, horizon = plans.shape
    next_states = np.zeros(
        (horizon, n_plans, members, FORMAL_STATE_DIM),
        dtype=np.float32,
    )
    member_returns = np.zeros((n_plans, members), dtype=np.float32)

    for plan_idx in range(n_plans):
        for member_idx in range(members):
            current = np.asarray(state, dtype=np.float32).copy()
            elapsed = 0
            total = 0.0

            for step in range(horizon):
                requested = int(plans[plan_idx, step])
                canonical = canonicalize_requested_action(current, requested)

                nxt = current.copy()
                nxt[0] += float(member_idx + 1)
                nxt[1] += float(canonical)

                reward = float(canonical + 1)
                if state_dependent:
                    reward += 0.01 * float(nxt[0])

                total += (float(gamma_tick) ** elapsed) * reward
                elapsed += DURATIONS[canonical]

                next_states[step, plan_idx, member_idx] = nxt
                current = nxt

            member_returns[plan_idx, member_idx] = total

    return next_states, member_returns, member_returns.mean(axis=1)


class TestSharedRolloutEvaluator(unittest.TestCase):

    def make_evaluator(
        self,
        *,
        members=3,
        gamma=0.99,
        device="cpu",
        state_dependent=False,
        wm_nonfinite=False,
        reward_nonfinite=False,
    ):
        wm = FakeWorldModel(
            ensemble_size=members,
            device=device,
            nonfinite=wm_nonfinite,
        )
        reward = FakeRewardPredictor(
            device=device,
            state_dependent=state_dependent,
            nonfinite=reward_nonfinite,
        )
        evaluator = SharedRolloutEvaluator(
            world_model=wm,
            reward_predictor=reward,
            config=SharedRolloutConfig(horizon=4, gamma_tick=gamma),
        )
        return evaluator, wm, reward

    def test_output_shapes_n1_d27(self):
        evaluator, _, _ = self.make_evaluator(members=5)
        result = evaluator.evaluate(base_state(), np.zeros((1, 4), dtype=np.int64))
        self.assertEqual(tuple(result.next_states.shape), (4, 1, 5, 27))
        self.assertEqual(tuple(result.member_returns.shape), (1, 5))
        self.assertEqual(tuple(result.expected_return.shape), (1,))

    def test_projection_is_applied_before_reward_and_recursive_child(self):
        evaluator, _, reward = self.make_evaluator(members=3)
        state=base_state(); state[0]=1.0; state[6]=.75
        context=D27ProjectionContext.from_root(state,root_tick=0,episode_steps=100)
        result=evaluator.evaluate(state,np.zeros((1,4),dtype=np.int64),projection_context=context)
        children=result.next_states[:,0]
        self.assertTrue(torch.all(children[...,0]==1)); self.assertTrue(torch.all(children[...,1:5]==0))
        self.assertTrue(torch.all(children[...,6]==.75)); self.assertTrue(torch.all(children>=0)); self.assertTrue(torch.all(children<=1))
        self.assertTrue(torch.allclose(children[0,:,5],torch.full((3,),.01)))
        self.assertTrue(torch.allclose(children[3,:,5],torch.full((3,),.04)))

    def test_output_shapes_n64(self):
        evaluator, _, _ = self.make_evaluator(members=5)
        plans = np.zeros((64, 4), dtype=np.int64)
        result = evaluator.evaluate(base_state(), plans)
        self.assertEqual(tuple(result.next_states.shape), (4, 64, 5, 27))
        self.assertEqual(tuple(result.member_returns.shape), (64, 5))

    def test_fixed_member_identity_through_horizon(self):
        evaluator, _, _ = self.make_evaluator(members=3)
        result = evaluator.evaluate(base_state(), np.zeros((1, 4), dtype=np.int64))
        values = result.next_states[:, 0, :, 0].cpu().numpy()
        expected = np.asarray(
            [[1, 2, 3], [2, 4, 6], [3, 6, 9], [4, 8, 12]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(values, expected, rtol=0, atol=0)

    def test_vectorized_forward_count_is_m_times_h(self):
        evaluator, wm, reward = self.make_evaluator(members=5)
        plans = np.tile(np.asarray([[0, 1, 2, 3]], dtype=np.int64), (64, 1))
        evaluator.evaluate(base_state(), plans)
        self.assertEqual(len(wm.calls), 5 * 4)
        self.assertEqual(len(reward.calls), 5 * 4)
        self.assertTrue(all(batch == 64 for _, batch in wm.calls))
        self.assertTrue(all(batch == 64 for batch in reward.calls))

    def test_deterministic_mean_path_uses_no_distribution_sampling(self):
        evaluator, _, _ = self.make_evaluator(members=3)
        first = evaluator.evaluate(base_state(), np.asarray([[0, 1, 2, 3]]))
        second = evaluator.evaluate(base_state(), np.asarray([[0, 1, 2, 3]]))
        torch.testing.assert_close(first.next_states, second.next_states)
        torch.testing.assert_close(first.member_returns, second.member_returns)

    def test_vectorized_matches_slow_reference(self):
        gamma = 0.93
        evaluator, _, _ = self.make_evaluator(
            members=3,
            gamma=gamma,
            state_dependent=True,
        )
        plans = np.asarray(
            [[0, 1, 2, 3], [3, 2, 1, 0], [1, 1, 0, 2]],
            dtype=np.int64,
        )
        result = evaluator.evaluate(base_state(), plans)
        slow_states, slow_returns, slow_expected = slow_reference(
            base_state(),
            plans,
            members=3,
            gamma_tick=gamma,
            state_dependent=True,
        )
        np.testing.assert_allclose(result.next_states.cpu().numpy(), slow_states, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(result.member_returns.cpu().numpy(), slow_returns, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(result.expected_return.cpu().numpy(), slow_expected, rtol=1e-6, atol=1e-6)

    def test_duration_aware_discount(self):
        evaluator, _, _ = self.make_evaluator(members=2, gamma=0.5)
        plans = np.asarray([[3, 0, 0, 0]], dtype=np.int64)
        result = evaluator.evaluate(base_state(), plans)
        expected = 4.0 + (0.5 ** 5) + (0.5 ** 6) + (0.5 ** 7)
        np.testing.assert_allclose(
            result.member_returns.cpu().numpy(),
            np.full((1, 2), expected, dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_canonical_fallback_affects_reward_and_duration(self):
        evaluator, _, _ = self.make_evaluator(members=2, gamma=0.5)
        plans = np.asarray([[3, 3, 3, 3]], dtype=np.int64)
        result = evaluator.evaluate(base_state(available=False), plans)
        expected = 1.0 + 0.5 + 0.25 + 0.125
        np.testing.assert_allclose(
            result.member_returns.cpu().numpy(),
            np.full((1, 2), expected, dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_expected_return_is_member_mean(self):
        evaluator, _, _ = self.make_evaluator(members=4, state_dependent=True)
        result = evaluator.evaluate(base_state(), np.asarray([[0, 1, 2, 3], [1, 0, 1, 0]]))
        torch.testing.assert_close(
            result.expected_return,
            result.member_returns.mean(dim=1),
        )

    def test_cpu_device_and_finite_output(self):
        evaluator, _, _ = self.make_evaluator(members=3, device="cpu")
        result = evaluator.evaluate(base_state(), np.asarray([[0, 1, 2, 3]]))
        self.assertEqual(result.next_states.device.type, "cpu")
        self.assertEqual(result.member_returns.device.type, "cpu")
        self.assertTrue(torch.isfinite(result.next_states).all())
        self.assertTrue(torch.isfinite(result.member_returns).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_device(self):
        evaluator, _, _ = self.make_evaluator(members=2, device="cuda")
        result = evaluator.evaluate(base_state(), np.asarray([[0, 1, 2, 3]]))
        self.assertEqual(result.next_states.device.type, "cuda")
        self.assertEqual(result.expected_return.device.type, "cuda")

    def test_nonfinite_and_contract_guards(self):
        evaluator, _, _ = self.make_evaluator()

        bad_state = base_state()
        bad_state[2] = np.nan
        with self.assertRaises(ValueError):
            evaluator.evaluate(bad_state, np.zeros((1, 4), dtype=np.int64))

        with self.assertRaises(ValueError):
            evaluator.evaluate(base_state(), np.zeros((1, 3), dtype=np.int64))

        with self.assertRaises(ValueError):
            evaluator.evaluate(base_state(), np.asarray([[0, 1, 2, 4]]))

        with self.assertRaises(ValueError):
            evaluator.evaluate(base_state(), np.asarray([[0.0, 1.5, 2.0, 3.0]]))

        bad_wm_eval, _, _ = self.make_evaluator(wm_nonfinite=True)
        with self.assertRaises(ValueError):
            bad_wm_eval.evaluate(base_state(), np.zeros((1, 4), dtype=np.int64))

        bad_reward_eval, _, _ = self.make_evaluator(reward_nonfinite=True)
        with self.assertRaises(ValueError):
            bad_reward_eval.evaluate(base_state(), np.zeros((1, 4), dtype=np.int64))

    def test_component_contract_guards(self):
        wm = FakeWorldModel(target_mode="delta")
        reward = FakeRewardPredictor()
        with self.assertRaises(ValueError):
            SharedRolloutEvaluator(world_model=wm, reward_predictor=reward)

        wm = FakeWorldModel(device="cpu")
        reward = FakeRewardPredictor(device="cuda" if torch.cuda.is_available() else "meta")
        with self.assertRaises(ValueError):
            SharedRolloutEvaluator(world_model=wm, reward_predictor=reward)


if __name__ == "__main__":
    unittest.main()
