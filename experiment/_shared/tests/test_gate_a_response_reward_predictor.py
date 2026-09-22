import tempfile
import unittest

from pathlib import Path

import numpy as np

from formal_experiments.data_collection.decision_replay import (
    DecisionEpochTransition,
    EXECUTED_DURATION,
)

from formal_experiments.evaluation.validate_response_reward_predictor import (
    build_reward_rollout_windows,
    discount_weights,
    discounted_sum,
    pearson,
    quality_gate,
)

from formal_experiments.training.response_reward_predictor import (
    ResponseRewardDataset,
    ResponseRewardPredictor,
    ResponseRewardPredictorConfig,
    ScalarNormalizer,
    build_response_reward_dataset,
)

from shared.action_contract import (
    get_action,
)

from shared.formal_state import (
    FORMAL_STATE_DIM,
)


def state(
    value,
):
    return np.full(
        (FORMAL_STATE_DIM,),
        value,
        dtype=np.float32,
    )


def transition(
    *,
    requested_action_id=0,
    executed_family="Sleep",
    decision_index=0,
    start_tick=0,
    start_value=0.0,
    next_value=0.1,
    response_reward=-1.0,
    completed=True,
):
    requested = get_action(
        requested_action_id
    )

    duration = (
        EXECUTED_DURATION[
            executed_family
        ]
    )

    dt = (
        duration
        if completed
        else max(
            1,
            duration - 1,
        )
    )

    return DecisionEpochTransition(
        episode_seed=2000,
        agent_name="blue_agent_0",
        decision_index=decision_index,

        global_tick_start=start_tick,
        global_tick_end=(
            start_tick + dt
        ),
        decision_dt=dt,

        state=state(
            start_value
        ),

        next_state=state(
            next_value
        ),

        requested_action_id=(
            requested_action_id
        ),

        requested_action_name=(
            requested.name
        ),

        requested_cyborg_family=(
            requested.cyborg_action
        ),

        requested_duration_ticks=(
            requested.duration_ticks
        ),

        executed_index=0,

        executed_label=(
            executed_family
        ),

        executed_action_family=(
            executed_family
        ),

        executed_duration_ticks=(
            duration
        ),

        target_host=(
            None
            if executed_family == "Sleep"
            else "host_a"
        ),

        fallback=(
            requested.cyborg_action
            != executed_family
        ),

        fallback_reason=(
            "no_valid_observable_target"
            if (
                requested.cyborg_action
                != executed_family
            )
            else None
        ),

        action_completed=(
            completed
        ),

        completed_action_success=(
            True
            if completed
            else None
        ),

        incident_event_ids=tuple(),
        incident_host_ids=tuple(),

        incident_active_ticks=1,
        incident_host_lwf_count=0,
        incident_host_lwf_raw_penalty=0.0,

        response_reward=float(
            response_reward
        ),

        official_reward=0.0,

        done=(
            not completed
        ),
    )


def tiny_dataset(
    n=64,
):
    rng = np.random.default_rng(
        123
    )

    states = rng.normal(
        size=(
            n,
            FORMAL_STATE_DIM,
        )
    ).astype(
        np.float32
    )

    actions = rng.integers(
        0,
        4,
        size=n,
        dtype=np.int64,
    )

    next_states = (
        states
        + 0.05
    ).astype(
        np.float32
    )

    rewards = (
        -0.5
        * states[
            :,
            0
        ]
        - actions.astype(
            np.float32
        )
    ).astype(
        np.float32
    )

    return ResponseRewardDataset(
        states=states,
        actions=actions,
        next_states=next_states,
        rewards=rewards,
    )


def tiny_config():
    return (
        ResponseRewardPredictorConfig(
            hidden_dim=16,
            lr=1e-3,
            batch_size=16,
            epochs=2,
            model_seed=123,
        )
    )


def chained_sequence(
    n=6,
):
    families = [
        "Sleep",
        "Analyse",
        "Remove",
        "Restore",
        "Sleep",
        "Analyse",
    ]

    ids = {
        "Sleep": 0,
        "Analyse": 1,
        "Remove": 2,
        "Restore": 3,
    }

    result = []

    tick = 0
    value = 0.0

    for index in range(
        n
    ):
        family = families[
            index
        ]

        duration = (
            EXECUTED_DURATION[
                family
            ]
        )

        next_value = (
            value
            + 0.1
        )

        result.append(
            transition(
                requested_action_id=(
                    ids[
                        family
                    ]
                ),
                executed_family=family,
                decision_index=index,
                start_tick=tick,
                start_value=value,
                next_value=next_value,
                response_reward=(
                    -float(
                        index + 1
                    )
                ),
            )
        )

        tick += duration
        value = next_value

    return result


class TestGateAResponseRewardPredictor(
    unittest.TestCase
):

    def test_dataset_uses_executed_action(
        self,
    ):
        item = transition(
            requested_action_id=3,
            executed_family="Sleep",
        )

        dataset = (
            build_response_reward_dataset(
                [item]
            )
        )

        self.assertEqual(
            dataset.actions.tolist(),
            [0],
        )

    def test_incomplete_is_dropped(
        self,
    ):
        good = transition()

        bad = transition(
            requested_action_id=3,
            executed_family="Restore",
            completed=False,
        )

        dataset = (
            build_response_reward_dataset(
                [
                    good,
                    bad,
                ]
            )
        )

        self.assertEqual(
            dataset.n_samples,
            1,
        )

    def test_incomplete_can_be_rejected(
        self,
    ):
        bad = transition(
            requested_action_id=3,
            executed_family="Restore",
            completed=False,
        )

        with self.assertRaises(
            ValueError
        ):
            build_response_reward_dataset(
                [bad],
                drop_incomplete=False,
            )

    def test_scalar_normalizer_round_trip(
        self,
    ):
        values = np.asarray(
            [
                -10.0,
                -2.0,
                0.0,
                -4.0,
            ],
            dtype=np.float32,
        )

        normalizer = (
            ScalarNormalizer.fit(
                values
            )
        )

        recovered = (
            normalizer
            .denormalize_np(
                normalizer
                .normalize_np(
                    values
                )
            )
        )

        np.testing.assert_allclose(
            recovered,
            values,
            atol=1e-6,
        )

    def test_fit_is_finite(
        self,
    ):
        predictor = (
            ResponseRewardPredictor(
                tiny_config()
            )
        )

        summary = predictor.fit(
            tiny_dataset()
        )

        self.assertTrue(
            np.isfinite(
                summary.final_loss
            )
        )

    def test_prediction_shape(
        self,
    ):
        dataset = tiny_dataset()

        predictor = (
            ResponseRewardPredictor(
                tiny_config()
            )
        )

        predictor.fit(
            dataset
        )

        result = predictor.predict(
            dataset.states[:7],
            dataset.actions[:7],
            dataset.next_states[:7],
        )

        self.assertEqual(
            result.shape,
            (7,),
        )

        self.assertTrue(
            np.all(
                np.isfinite(
                    result
                )
            )
        )

    def test_checkpoint_round_trip(
        self,
    ):
        dataset = tiny_dataset()

        predictor = (
            ResponseRewardPredictor(
                tiny_config()
            )
        )

        predictor.fit(
            dataset
        )

        before = predictor.predict(
            dataset.states[:5],
            dataset.actions[:5],
            dataset.next_states[:5],
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = (
                Path(tmp)
                / "reward.pt"
            )

            predictor.save_checkpoint(
                path
            )

            loaded = (
                ResponseRewardPredictor
                .load_checkpoint(
                    path
                )
            )

            after = loaded.predict(
                dataset.states[:5],
                dataset.actions[:5],
                dataset.next_states[:5],
            )

        np.testing.assert_allclose(
            before,
            after,
            rtol=0.0,
            atol=0.0,
        )

    def test_discount_weights_are_tick_aware(
        self,
    ):
        result = discount_weights(
            [
                1,
                2,
                3,
            ],
            0.9,
        )

        np.testing.assert_allclose(
            result,
            [
                1.0,
                0.9,
                0.9 ** 3,
            ],
        )

    def test_discounted_sum(
        self,
    ):
        result = discounted_sum(
            [
                -1.0,
                -2.0,
                -3.0,
            ],
            [
                1,
                2,
                3,
            ],
            0.9,
        )

        expected = (
            -1.0
            -2.0 * 0.9
            -3.0 * (
                0.9 ** 3
            )
        )

        self.assertAlmostEqual(
            result,
            expected,
        )

    def test_h4_window_count(
        self,
    ):
        windows = (
            build_reward_rollout_windows(
                chained_sequence(
                    6
                ),
                horizon=4,
            )
        )

        self.assertEqual(
            len(windows),
            3,
        )

    def test_incomplete_breaks_h4_window(
        self,
    ):
        sequence = (
            chained_sequence(
                5
            )
        )

        last = sequence[
            -1
        ]

        sequence.append(
            transition(
                requested_action_id=3,
                executed_family="Restore",
                decision_index=5,
                start_tick=(
                    last.global_tick_end
                ),
                start_value=0.5,
                next_value=0.6,
                completed=False,
            )
        )

        windows = (
            build_reward_rollout_windows(
                sequence,
                horizon=4,
            )
        )

        self.assertEqual(
            len(windows),
            2,
        )

    def test_pearson_sign(
        self,
    ):
        x = np.arange(
            10,
            dtype=np.float64,
        )

        self.assertGreater(
            pearson(
                x,
                x,
            ),
            0.99,
        )

        self.assertLess(
            pearson(
                x,
                -x,
            ),
            -0.99,
        )

    def test_quality_gate_pass(
        self,
    ):
        one_step = {
            "rmse": 1.0,
            "train_mean_baseline_rmse":
                2.0,
        }

        wm_h4 = {
            "rmse": 2.0,
            "constant_baseline_rmse":
                3.0,
            "spearman": 0.5,
            "positive_episode_spearman_count":
                6,
        }

        result = quality_gate(
            one_step=one_step,
            wm_h4=wm_h4,
        )

        self.assertTrue(
            result[
                "pass"
            ]
        )

    def test_quality_gate_rejects_weak_value_relation(
        self,
    ):
        one_step = {
            "rmse": 1.0,
            "train_mean_baseline_rmse":
                2.0,
        }

        wm_h4 = {
            "rmse": 2.0,
            "constant_baseline_rmse":
                3.0,
            "spearman": 0.2,
            "positive_episode_spearman_count":
                8,
        }

        result = quality_gate(
            one_step=one_step,
            wm_h4=wm_h4,
        )

        self.assertFalse(
            result[
                "pass"
            ]
        )


if __name__ == "__main__":
    unittest.main()