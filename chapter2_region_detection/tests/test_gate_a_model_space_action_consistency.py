import unittest

import numpy as np
import torch

from formal_experiments.evaluation.audit_model_space_action_consistency import (
    DEFAULT_REPORT_OUT,
    DEFAULT_REWARD_MODEL,
    DEFAULT_TRAIN_REPLAY,
    DEFAULT_VALIDATION_REPLAY,
    DEFAULT_WORLD_MODEL,
    quality_gate,
)

from shared.model_space_action import (
    ANY_TARGET_INDEX,
    canonicalize_requested_action,
    canonicalize_requested_tensor,
)

from shared.formal_state import (
    FORMAL_STATE_DIM,
)


def state(
    available: float,
):
    result = np.zeros(
        FORMAL_STATE_DIM,
        dtype=np.float32,
    )

    result[
        ANY_TARGET_INDEX
    ] = float(
        available
    )

    return result


class TestGateAModelSpaceActionConsistency(
    unittest.TestCase
):

    def test_noop_always_sleep(
        self,
    ):
        self.assertEqual(
            canonicalize_requested_action(
                state(0.0),
                0,
            ),
            0,
        )

        self.assertEqual(
            canonicalize_requested_action(
                state(1.0),
                0,
            ),
            0,
        )

    def test_targeted_without_target_falls_back(
        self,
    ):
        for action_id in (
            1,
            2,
            3,
        ):
            self.assertEqual(
                canonicalize_requested_action(
                    state(0.0),
                    action_id,
                ),
                0,
            )

    def test_targeted_with_target_is_kept(
        self,
    ):
        for action_id in (
            1,
            2,
            3,
        ):
            self.assertEqual(
                canonicalize_requested_action(
                    state(1.0),
                    action_id,
                ),
                action_id,
            )

    def test_threshold_is_half(
        self,
    ):
        self.assertEqual(
            canonicalize_requested_action(
                state(0.49),
                3,
            ),
            0,
        )

        self.assertEqual(
            canonicalize_requested_action(
                state(0.50),
                3,
            ),
            3,
        )

    def test_tensor_matches_scalar(
        self,
    ):
        states = np.stack(
            [
                state(0.0),
                state(1.0),
                state(0.0),
                state(1.0),
            ]
        )

        requested = np.asarray(
            [
                0,
                1,
                2,
                3,
            ],
            dtype=np.int64,
        )

        tensor_result = (
            canonicalize_requested_tensor(
                torch.as_tensor(
                    states
                ),

                torch.as_tensor(
                    requested
                ),
            )
            .cpu()
            .numpy()
        )

        scalar_result = np.asarray(
            [
                canonicalize_requested_action(
                    states[index],
                    int(
                        requested[
                            index
                        ]
                    ),
                )
                for index
                in range(
                    len(
                        requested
                    )
                )
            ],
            dtype=np.int64,
        )

        np.testing.assert_array_equal(
            tensor_result,
            scalar_result,
        )

    def test_invalid_action_rejected(
        self,
    ):
        with self.assertRaises(
            ValueError
        ):
            canonicalize_requested_action(
                state(1.0),
                4,
            )

    def test_formal_defaults_use_v2_artifacts(
        self,
    ):
        self.assertEqual(
            DEFAULT_TRAIN_REPLAY,
            "outputs/formal_replay_v2/train.jsonl",
        )

        self.assertEqual(
            DEFAULT_VALIDATION_REPLAY,
            "outputs/formal_replay_v2/validation.jsonl",
        )

        self.assertEqual(
            DEFAULT_WORLD_MODEL,
            (
                "outputs/world_model_v2/"
                "a4_5b/"
                "world_model_absolute.pt"
            ),
        )

        self.assertEqual(
            DEFAULT_REWARD_MODEL,
            (
                "outputs/world_model_v2/"
                "a4_5c/"
                "response_reward_predictor.pt"
            ),
        )

        self.assertEqual(
            DEFAULT_REPORT_OUT,
            (
                "outputs/world_model_v2/"
                "a4_6a/"
                "action_consistency_report.json"
            ),
        )

    def test_formal_defaults_do_not_bind_legacy_artifacts(
        self,
    ):
        formal_paths = (
            DEFAULT_TRAIN_REPLAY,
            DEFAULT_VALIDATION_REPLAY,
            DEFAULT_WORLD_MODEL,
            DEFAULT_REWARD_MODEL,
            DEFAULT_REPORT_OUT,
        )

        for path in formal_paths:
            self.assertNotIn(
                "outputs/formal_replay/",
                path,
            )

            self.assertNotIn(
                "outputs/world_model/",
                path,
            )

    def test_quality_gate(
        self,
    ):
        mapping = {
            "accuracy": 1.0,
            "feature_one_but_fallback": 0,
            "feature_zero_but_nonfallback": 0,
        }

        integrated = {
            "h4_state_rmse": 0.2,
            "persistence_state_rmse": 0.3,

            "h4_value_rmse": 5.0,
            "constant_value_baseline_rmse": 10.0,

            "h4_value_spearman": 0.6,

            "positive_episode_value_spearman_count":
                8,
        }

        result = quality_gate(
            train_mapping=mapping,
            validation_mapping=mapping,
            integrated=integrated,
        )

        self.assertTrue(
            result[
                "pass"
            ]
        )


if __name__ == "__main__":
    unittest.main()