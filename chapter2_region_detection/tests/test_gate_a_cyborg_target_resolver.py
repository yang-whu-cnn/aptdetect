import unittest

from shared.cyborg_target_resolver import (
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
        # 故意不使用真实 CC4 的 49 / 145。
        #
        # 这样可以证明 resolver
        # 没有偷偷依赖固定 raw index。
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
            True,   # Monitor
            False,  # invalid Analyse
            True,   # Sleep
            True,   # Analyse b
            True,   # Analyse a
            True,   # Remove b
            True,   # Remove a
            True,   # Restore b
            True,   # Restore a
            True,   # traffic action
        ]

    # =========================================================
    # 1. only four-action valid candidates survive
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
            for item in got
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
    # 2. invalid padded action excluded
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
            for item in got
        }

        self.assertNotIn(
            (
                "[Invalid] "
                "Analyse host_invalid"
            ),
            labels,
        )

    # =========================================================
    # 3. Sleep resolved dynamically
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
    # 4. highest observable score
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
    # 5. deterministic tie break
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

        # host_a 字典序优先，
        # 即使它 raw index 更大。
        self.assertEqual(
            got.target_host,
            "host_a",
        )

        self.assertEqual(
            got.index,
            4,
        )

    # =========================================================
    # 6. action family respected
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
    # 7. unknown observable host ignored
    # =========================================================
    def test_non_candidate_host_is_not_hallucinated(
        self,
    ):
        got = resolve_target_action(
            family="Remove",
            labels=self.labels,
            mask=self.mask,
            observable_host_scores={
                "host_not_in_actions": 1.0,
            },
        )

        self.assertIsNone(
            got
        )

    # =========================================================
    # 8. no observable evidence -> no target
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
    # 9. invalid score rejected
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
                    "host_a": float(
                        "nan"
                    ),
                },
            )

    # =========================================================
    # 10. malformed action-space rejected
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
    # 11. unsupported family rejected
    # =========================================================
    def test_unsupported_target_family_rejected(
        self,
    ):
        with self.assertRaises(
            ValueError
        ):
            resolve_target_action(
                family="BlockTrafficZone",
                labels=self.labels,
                mask=self.mask,
                observable_host_scores={
                    "host_a": 1.0,
                },
            )

    # =========================================================
    # 12. no valid Sleep is explicit failure
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


if __name__ == "__main__":
    unittest.main()