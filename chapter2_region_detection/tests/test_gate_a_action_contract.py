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
            5,
        )

        self.assertEqual(
            len(ACTION_CONTRACTS),
            5,
        )

    def test_numeric_id_order(
        self,
    ):
        expected = [
            "no_op",
            "analyse",
            "control_traffic",
            "remove",
            "restore",
        ]

        got = [
            action.name
            for action
            in ACTION_CONTRACTS
        ]

        self.assertEqual(
            got,
            expected,
        )

        self.assertEqual(
            [
                action.action_id
                for action
                in ACTION_CONTRACTS
            ],
            [
                0,
                1,
                2,
                3,
                4,
            ],
        )

    def test_name_to_id_contract(
        self,
    ):
        expected = {
            "no_op": 0,
            "analyse": 1,
            "control_traffic": 2,
            "remove": 3,
            "restore": 4,
        }

        self.assertEqual(
            NAME2ID,
            expected,
        )

        for name, action_id in (
            expected.items()
        ):
            self.assertEqual(
                get_action_id(name),
                action_id,
            )

    def test_id_to_action_contract(
        self,
    ):
        for action_id in range(5):
            action = get_action(
                action_id
            )

            self.assertEqual(
                action.action_id,
                action_id,
            )

            self.assertIs(
                ID2ACTION[action_id],
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
            "traffic blocking",
            get_action(2)
            .description
            .lower(),
        )

        self.assertIn(
            "user-level compromise removal",
            get_action(3)
            .description
            .lower(),
        )

        self.assertIn(
            "host reimaging",
            get_action(4)
            .description
            .lower(),
        )

    def test_cyborg_action_types(
        self,
    ):
        expected = [
            "Sleep",
            "Analyse",
            "BlockTraffic",
            "Remove",
            "Restore",
        ]

        got = [
            get_cyborg_action_type(i)
            for i in range(5)
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
            1,
            3,
            5,
        ]

        got = [
            get_action_duration(i)
            for i in range(5)
        ]

        self.assertEqual(
            got,
            expected,
        )

    def test_invalid_action_id(
        self,
    ):
        for bad_id in (
            -1,
            5,
            100,
        ):
            with self.assertRaises(
                ValueError
            ):
                get_action(
                    bad_id
                )

    def test_invalid_action_name(
        self,
    ):
        with self.assertRaises(
            ValueError
        ):
            get_action_id(
                "heavy_evidence"
            )


if __name__ == "__main__":
    unittest.main()