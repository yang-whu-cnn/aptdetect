import unittest

from shared.cyborg_target_resolver import (
    observable_target_family_availability,
    resolve_sleep_action,
    resolve_target_action,
    valid_actions,
)


class TestGateACyborgTargetResolver(
    unittest.TestCase
):

    def setUp(
        self,
    ):
        self.labels = [
            "Monitor",

            (
                "[Invalid] "
                "Analyse host_invalid"
            ),

            "Sleep",

            "Analyse host_b",
            "Analyse host_a",

            "Remove host_b",
            "Remove host_a",

            "Restore host_b",
            "Restore host_a",

            "BlockTrafficZone",
        ]

        self.mask = [
            True,
            False,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
        ]

    # =========================================================
    # 1
    # =========================================================

    def test_valid_actions_filter_other_families(
        self,
    ):
        got = valid_actions(
            self.labels,
            self.mask,
        )

        families = [
            item.family
            for item
            in got
        ]

        self.assertEqual(
            families,
            [
                "Sleep",
                "Analyse",
                "Analyse",
                "Remove",
                "Remove",
                "Restore",
                "Restore",
            ],
        )

    # =========================================================
    # 2
    # =========================================================

    def test_invalid_padded_action_excluded(
        self,
    ):
        got = valid_actions(
            self.labels,
            self.mask,
        )

        labels = {
            item.label
            for item
            in got
        }

        self.assertNotIn(
            (
                "[Invalid] "
                "Analyse host_invalid"
            ),
            labels,
        )

    # =========================================================
    # 3
    # =========================================================

    def test_sleep_is_dynamic_not_fixed_index(
        self,
    ):
        got = resolve_sleep_action(
            self.labels,
            self.mask,
        )

        self.assertEqual(
            got.index,
            2,
        )

        self.assertEqual(
            got.family,
            "Sleep",
        )

        self.assertIsNone(
            got.target_host
        )

    # =========================================================
    # 4
    # =========================================================

    def test_highest_observable_score_selected(
        self,
    ):
        got = resolve_target_action(
            family="Analyse",

            labels=self.labels,

            mask=self.mask,

            observable_host_scores={
                "host_a": 0.30,
                "host_b": 0.90,
            },
        )

        self.assertIsNotNone(
            got
        )

        self.assertEqual(
            got.target_host,
            "host_b",
        )

        self.assertEqual(
            got.index,
            3,
        )

    # =========================================================
    # 5
    # =========================================================

    def test_equal_score_uses_host_name_tie_break(
        self,
    ):
        got = resolve_target_action(
            family="Analyse",

            labels=self.labels,

            mask=self.mask,

            observable_host_scores={
                "host_b": 0.50,
                "host_a": 0.50,
            },
        )

        self.assertIsNotNone(
            got
        )

        self.assertEqual(
            got.target_host,
            "host_a",
        )

        self.assertEqual(
            got.index,
            4,
        )

    # =========================================================
    # 6
    # =========================================================

    def test_requested_family_is_respected(
        self,
    ):
        scores = {
            "host_a": 0.90,
            "host_b": 0.10,
        }

        analyse = resolve_target_action(
            "Analyse",
            self.labels,
            self.mask,
            scores,
        )

        remove = resolve_target_action(
            "Remove",
            self.labels,
            self.mask,
            scores,
        )

        restore = resolve_target_action(
            "Restore",
            self.labels,
            self.mask,
            scores,
        )

        self.assertEqual(
            analyse.index,
            4,
        )

        self.assertEqual(
            remove.index,
            6,
        )

        self.assertEqual(
            restore.index,
            8,
        )

    # =========================================================
    # 7
    # =========================================================

    def test_non_candidate_host_is_not_hallucinated(
        self,
    ):
        got = resolve_target_action(
            family="Remove",

            labels=self.labels,

            mask=self.mask,

            observable_host_scores={
                "host_not_in_actions":
                    1.0,
            },
        )

        self.assertIsNone(
            got
        )

    # =========================================================
    # 8
    # =========================================================

    def test_no_observable_evidence_returns_none(
        self,
    ):
        got = resolve_target_action(
            family="Restore",

            labels=self.labels,

            mask=self.mask,

            observable_host_scores=None,
        )

        self.assertIsNone(
            got
        )

    # =========================================================
    # 9
    # =========================================================

    def test_nonfinite_score_rejected(
        self,
    ):
        with self.assertRaises(
            ValueError
        ):
            resolve_target_action(
                family="Analyse",

                labels=self.labels,

                mask=self.mask,

                observable_host_scores={
                    "host_a":
                        float(
                            "nan"
                        ),
                },
            )

    # =========================================================
    # 10
    # =========================================================

    def test_labels_mask_length_mismatch_rejected(
        self,
    ):
        with self.assertRaises(
            ValueError
        ):
            valid_actions(
                [
                    "Sleep",
                    "Analyse host_a",
                ],

                [
                    True,
                ],
            )

    # =========================================================
    # 11
    # =========================================================

    def test_unsupported_target_family_rejected(
        self,
    ):
        with self.assertRaises(
            ValueError
        ):
            resolve_target_action(
                family=(
                    "BlockTrafficZone"
                ),

                labels=self.labels,

                mask=self.mask,

                observable_host_scores={
                    "host_a": 1.0,
                },
            )

    # =========================================================
    # 12
    # =========================================================

    def test_no_valid_sleep_rejected(
        self,
    ):
        labels = [
            "Sleep",
            "Analyse host_a",
        ]

        mask = [
            False,
            True,
        ]

        with self.assertRaises(
            RuntimeError
        ):
            resolve_sleep_action(
                labels,
                mask,
            )

    # =========================================================
    # 13
    #
    # A4.1c regression:
    #
    # 有 evidence 不代表 wrapper 中有合法 target。
    # =========================================================

    def test_observable_evidence_without_valid_host_is_unavailable(
        self,
    ):
        got = (
            observable_target_family_availability(
                labels=self.labels,

                mask=self.mask,

                observable_host_scores={
                    "host_invalid":
                        1.0,
                },
            )
        )

        self.assertEqual(
            got,
            {
                "Analyse": False,
                "Remove": False,
                "Restore": False,
            },
        )

    # =========================================================
    # 14
    # =========================================================

    def test_valid_observable_host_available_for_all_targeted_families(
        self,
    ):
        got = (
            observable_target_family_availability(
                labels=self.labels,

                mask=self.mask,

                observable_host_scores={
                    "host_a":
                        1.0,
                },
            )
        )

        self.assertEqual(
            got,
            {
                "Analyse": True,
                "Remove": True,
                "Restore": True,
            },
        )


if __name__ == "__main__":
    unittest.main()