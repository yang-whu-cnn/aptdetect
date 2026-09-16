import unittest

from formal_experiments.data_collection.collect_cc4_formal_replay import (
    DEFAULT_SPLIT_SEEDS,
    observation_success_bool,
    parse_seed_csv,
    seeds_for_split,
    stratified_requested_action_id,
)

from shared.formal_state import (
    BLUE_AGENTS,
)


class FakeSuccess:
    def __init__(
        self,
        name,
    ):
        self.name = name


class TestGateAFormalReplayCollection(
    unittest.TestCase
):

    def test_frozen_seed_counts(
        self,
    ):
        self.assertEqual(
            len(
                DEFAULT_SPLIT_SEEDS[
                    "train"
                ]
            ),
            32,
        )

        self.assertEqual(
            len(
                DEFAULT_SPLIT_SEEDS[
                    "validation"
                ]
            ),
            8,
        )

        self.assertEqual(
            len(
                DEFAULT_SPLIT_SEEDS[
                    "calibration"
                ]
            ),
            8,
        )

        self.assertEqual(
            len(
                DEFAULT_SPLIT_SEEDS[
                    "test"
                ]
            ),
            20,
        )

    def test_seed_splits_are_disjoint(
        self,
    ):
        names = [
            "train",
            "validation",
            "calibration",
            "test",
        ]

        for i, left in enumerate(
            names
        ):
            for right in names[
                i + 1:
            ]:
                with self.subTest(
                    left=left,
                    right=right,
                ):
                    self.assertTrue(
                        set(
                            DEFAULT_SPLIT_SEEDS[
                                left
                            ]
                        )
                        .isdisjoint(
                            DEFAULT_SPLIT_SEEDS[
                                right
                            ]
                        )
                    )

    def test_parse_seed_csv(
        self,
    ):
        self.assertEqual(
            parse_seed_csv(
                "10, 20,30"
            ),
            [
                10,
                20,
                30,
            ],
        )

    def test_duplicate_seed_rejected(
        self,
    ):
        with self.assertRaises(
            ValueError
        ):
            parse_seed_csv(
                "10,10"
            )

    def test_a45_only_train_validation(
        self,
    ):
        with self.assertRaises(
            ValueError
        ):
            seeds_for_split(
                "test",
                None,
            )

        with self.assertRaises(
            ValueError
        ):
            seeds_for_split(
                "calibration",
                None,
            )

    def test_stratified_policy_exact_four_cycle(
        self,
    ):
        for agent in BLUE_AGENTS:

            actions = [
                stratified_requested_action_id(
                    episode_seed=1000,
                    agent_name=agent,
                    decision_index=index,
                )
                for index in range(
                    8
                )
            ]

            counts = {
                action_id:
                    actions.count(
                        action_id
                    )
                for action_id
                in range(
                    4
                )
            }

            self.assertEqual(
                counts,
                {
                    0: 2,
                    1: 2,
                    2: 2,
                    3: 2,
                },
            )

    def test_observation_success_parser(
        self,
    ):
        self.assertIs(
            observation_success_bool(
                {
                    "success": True
                }
            ),
            True,
        )

        self.assertIs(
            observation_success_bool(
                {
                    "success": False
                }
            ),
            False,
        )

        self.assertIs(
            observation_success_bool(
                {
                    "success":
                        FakeSuccess(
                            "TRUE"
                        )
                }
            ),
            True,
        )

        self.assertIs(
            observation_success_bool(
                {
                    "success":
                        FakeSuccess(
                            "FALSE"
                        )
                }
            ),
            False,
        )

        self.assertIsNone(
            observation_success_bool(
                {
                    "success":
                        FakeSuccess(
                            "IN_PROGRESS"
                        )
                }
            )
        )


if __name__ == "__main__":
    unittest.main()