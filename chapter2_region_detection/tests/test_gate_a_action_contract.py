import unittest

from shared.action_contract import (
    ACTION_CONTRACTS,
    N_ACTIONS,
    ID2ACTION,
    NAME2ID,
    get_action,
    get_action_id,
    get_cyborg_action_type,
    get_action_duration,
)


class TestGateAActionContract(
    unittest.TestCase
):

    def test_number_of_actions(
        self,
    ):
        self.assertEqual(
            N_ACTIONS,
            4,
        )

        self.assertEqual(
            len(ACTION_CONTRACTS),
            4,
        )

    def test_numeric_id_order(
        self,
    ):
        expected_names = [
            "no_op",
            "analyse",
            "remove",
            "restore",
        ]

        expected_ids = [
            0,
            1,
            2,
            3,
        ]

        self.assertEqual(
            [
                action.name
                for action
                in ACTION_CONTRACTS
            ],
            expected_names,
        )

        self.assertEqual(
            [
                action.action_id
                for action
                in ACTION_CONTRACTS
            ],
            expected_ids,
        )

    def test_name_to_id_contract(
        self,
    ):
        expected = {
            "no_op": 0,
            "analyse": 1,
            "remove": 2,
            "restore": 3,
        }

        self.assertEqual(
            NAME2ID,
            expected,
        )

        for (
            name,
            action_id,
        ) in expected.items():
            self.assertEqual(
                get_action_id(
                    name
                ),
                action_id,
            )

    def test_id_to_action_contract(
        self,
    ):
        for action_id in range(
            N_ACTIONS
        ):
            action = get_action(
                action_id
            )

            self.assertEqual(
                action.action_id,
                action_id,
            )

            self.assertIs(
                ID2ACTION[
                    action_id
                ],
                action,
            )

    def test_paper_semantics(
        self,
    ):
        self.assertIn(
            "no operation",
            get_action(0)
            .description
            .lower(),
        )

        self.assertIn(
            "intrusion investigation",
            get_action(1)
            .description
            .lower(),
        )

        self.assertIn(
            "user-level compromise removal",
            get_action(2)
            .description
            .lower(),
        )

        self.assertIn(
            "host reimaging",
            get_action(3)
            .description
            .lower(),
        )

    def test_cyborg_action_types(
        self,
    ):
        expected = [
            "Sleep",
            "Analyse",
            "Remove",
            "Restore",
        ]

        got = [
            get_cyborg_action_type(i)
            for i in range(
                N_ACTIONS
            )
        ]

        self.assertEqual(
            got,
            expected,
        )

    def test_action_durations(
        self,
    ):
        expected = [
            1,
            2,
            3,
            5,
        ]

        got = [
            get_action_duration(i)
            for i in range(
                N_ACTIONS
            )
        ]

        self.assertEqual(
            got,
            expected,
        )

    def test_removed_control_traffic_rejected(
        self,
    ):
        with self.assertRaises(
            ValueError
        ):
            get_action_id(
                "control_traffic"
            )

    def test_legacy_action_names_rejected(
        self,
    ):
        legacy_names = [
            "monitor",
            "light_evidence",
            "heavy_evidence",
            "local_mitigate",
            "strong_mitigate",
        ]

        for name in legacy_names:
            with self.assertRaises(
                ValueError
            ):
                get_action_id(
                    name
                )

    def test_invalid_action_id(
        self,
    ):
        for bad_id in (
            -1,
            4,
            100,
        ):
            with self.assertRaises(
                ValueError
            ):
                get_action(
                    bad_id
                )

    def test_non_integer_action_id_rejected(
        self,
    ):
        bad_ids = (
            1.5,
            "1",
            None,
            True,
        )

        for bad_id in bad_ids:
            with self.assertRaises(
                ValueError
            ):
                get_action(
                    bad_id
                )


if __name__ == "__main__":
    unittest.main()