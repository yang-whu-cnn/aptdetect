import unittest

from shared.cyborg_action_adapter import (
    CybORGActionAdapter,
)


class FakeBlueFixedActionWrapper:
    """
    A3.2/A3.3 unit test fake。

    不依赖 CybORG 安装，
    只模拟正式 adapter 真正使用的：

        action_labels(agent)
        action_mask(agent)
    """

    def __init__(
        self,
        labels,
        mask,
    ):
        self._labels = list(
            labels
        )

        self._mask = list(
            mask
        )

    def action_labels(
        self,
        agent_name,
    ):
        return list(
            self._labels
        )

    def action_mask(
        self,
        agent_name,
    ):
        return list(
            self._mask
        )


class TestGateACyborgActionAdapter(
    unittest.TestCase
):

    def setUp(
        self,
    ):
        self.adapter = (
            CybORGActionAdapter()
        )

        # Sleep 故意放在 index 2，
        # 防止实现偷偷写死 49 / 145。
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
            "AllowTrafficZone",
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

        self.env = (
            FakeBlueFixedActionWrapper(
                self.labels,
                self.mask,
            )
        )

    # =========================================================
    # 1. no_op -> Sleep
    # =========================================================
    def test_no_op_maps_to_dynamic_sleep(
        self,
    ):
        got = self.adapter.resolve(
            env=self.env,
            agent_name="blue_agent_0",
            action_id=0,
        )

        self.assertEqual(
            got.requested_action_name,
            "no_op",
        )

        self.assertEqual(
            got.requested_cyborg_family,
            "Sleep",
        )

        self.assertEqual(
            got.executed_index,
            2,
        )

        self.assertEqual(
            got.executed_label,
            "Sleep",
        )

        self.assertEqual(
            got.executed_action_family,
            "Sleep",
        )

        self.assertIsNone(
            got.target_host
        )

        self.assertFalse(
            got.fallback
        )

    # =========================================================
    # 2. no_op is not Monitor
    # =========================================================
    def test_no_op_never_maps_to_monitor(
        self,
    ):
        got = self.adapter.resolve(
            self.env,
            "blue_agent_0",
            0,
        )

        self.assertNotEqual(
            got.executed_label,
            "Monitor",
        )

    # =========================================================
    # 3. analyse
    # =========================================================
    def test_analyse_uses_observable_host_score(
        self,
    ):
        got = self.adapter.resolve(
            env=self.env,
            agent_name="blue_agent_0",
            action_id=1,
            observable_host_scores={
                "host_a": 0.20,
                "host_b": 0.90,
            },
        )

        self.assertEqual(
            got.requested_action_name,
            "analyse",
        )

        self.assertEqual(
            got.executed_index,
            3,
        )

        self.assertEqual(
            got.executed_action_family,
            "Analyse",
        )

        self.assertEqual(
            got.target_host,
            "host_b",
        )

        self.assertFalse(
            got.fallback
        )

    # =========================================================
    # 4. remove
    # =========================================================
    def test_remove_uses_same_shared_resolver(
        self,
    ):
        got = self.adapter.resolve(
            env=self.env,
            agent_name="blue_agent_1",
            action_id=2,
            observable_host_scores={
                "host_a": 0.80,
                "host_b": 0.10,
            },
        )

        self.assertEqual(
            got.executed_index,
            6,
        )

        self.assertEqual(
            got.executed_action_family,
            "Remove",
        )

        self.assertEqual(
            got.target_host,
            "host_a",
        )

        self.assertFalse(
            got.fallback
        )

    # =========================================================
    # 5. restore
    # =========================================================
    def test_restore_uses_same_shared_resolver(
        self,
    ):
        got = self.adapter.resolve(
            env=self.env,
            agent_name="blue_agent_2",
            action_id=3,
            observable_host_scores={
                "host_a": 0.10,
                "host_b": 0.70,
            },
        )

        self.assertEqual(
            got.executed_index,
            7,
        )

        self.assertEqual(
            got.executed_action_family,
            "Restore",
        )

        self.assertEqual(
            got.target_host,
            "host_b",
        )

        self.assertFalse(
            got.fallback
        )

    # =========================================================
    # 6. no target -> fallback Sleep
    # =========================================================
    def test_no_observable_target_falls_back_to_sleep(
        self,
    ):
        got = self.adapter.resolve(
            env=self.env,
            agent_name="blue_agent_0",
            action_id=2,
            observable_host_scores=None,
        )

        # requested 仍然必须保留 remove。
        self.assertEqual(
            got.requested_action_id,
            2,
        )

        self.assertEqual(
            got.requested_action_name,
            "remove",
        )

        # executed 才是 Sleep。
        self.assertEqual(
            got.executed_index,
            2,
        )

        self.assertEqual(
            got.executed_action_family,
            "Sleep",
        )

        self.assertIsNone(
            got.target_host
        )

        self.assertTrue(
            got.fallback
        )

        self.assertEqual(
            got.fallback_reason,
            "no_valid_observable_target",
        )

    # =========================================================
    # 7. invalid padded target cannot execute
    # =========================================================
    def test_invalid_padded_target_cannot_be_selected(
        self,
    ):
        got = self.adapter.resolve(
            env=self.env,
            agent_name="blue_agent_0",
            action_id=1,
            observable_host_scores={
                # 分数非常高，
                # 但 wrapper mask=False。
                "host_invalid": 999.0,
            },
        )

        self.assertEqual(
            got.executed_action_family,
            "Sleep",
        )

        self.assertTrue(
            got.fallback
        )

    # =========================================================
    # 8. deterministic tie
    # =========================================================
    def test_equal_scores_are_deterministic(
        self,
    ):
        first = self.adapter.resolve(
            self.env,
            "blue_agent_0",
            1,
            {
                "host_b": 0.5,
                "host_a": 0.5,
            },
        )

        second = self.adapter.resolve(
            self.env,
            "blue_agent_0",
            1,
            {
                "host_a": 0.5,
                "host_b": 0.5,
            },
        )

        self.assertEqual(
            first,
            second,
        )

        self.assertEqual(
            first.target_host,
            "host_a",
        )

    # =========================================================
    # 9. raw action indices are not hardcoded
    # =========================================================
    def test_different_layout_still_resolves(
        self,
    ):
        env = (
            FakeBlueFixedActionWrapper(
                labels=[
                    "Sleep",
                    "Restore another_host",
                    "Monitor",
                    "Analyse another_host",
                    "Remove another_host",
                ],
                mask=[
                    True,
                    True,
                    True,
                    True,
                    True,
                ],
            )
        )

        no_op = self.adapter.resolve(
            env,
            "blue_agent_4",
            0,
        )

        analyse = self.adapter.resolve(
            env,
            "blue_agent_4",
            1,
            {
                "another_host": 1.0,
            },
        )

        remove = self.adapter.resolve(
            env,
            "blue_agent_4",
            2,
            {
                "another_host": 1.0,
            },
        )

        restore = self.adapter.resolve(
            env,
            "blue_agent_4",
            3,
            {
                "another_host": 1.0,
            },
        )

        self.assertEqual(
            no_op.executed_index,
            0,
        )

        self.assertEqual(
            analyse.executed_index,
            3,
        )

        self.assertEqual(
            remove.executed_index,
            4,
        )

        self.assertEqual(
            restore.executed_index,
            1,
        )

    # =========================================================
    # 10. metadata preserves agent
    # =========================================================
    def test_result_preserves_agent_name(
        self,
    ):
        got = self.adapter.resolve(
            self.env,
            "blue_agent_4",
            0,
        )

        self.assertEqual(
            got.agent_name,
            "blue_agent_4",
        )

    # =========================================================
    # 11. old action id rejected
    # =========================================================
    def test_old_five_action_id_rejected(
        self,
    ):
        with self.assertRaises(
            ValueError
        ):
            self.adapter.resolve(
                self.env,
                "blue_agent_0",
                4,
            )

    # =========================================================
    # 12. malformed wrapper rejected
    # =========================================================
    def test_labels_mask_length_mismatch_rejected(
        self,
    ):
        env = (
            FakeBlueFixedActionWrapper(
                labels=[
                    "Sleep",
                    "Analyse host_a",
                ],
                mask=[
                    True,
                ],
            )
        )

        with self.assertRaises(
            ValueError
        ):
            self.adapter.resolve(
                env,
                "blue_agent_0",
                0,
            )

    # =========================================================
    # 13. fallback requires real valid Sleep
    # =========================================================
    def test_fallback_fails_if_no_valid_sleep_exists(
        self,
    ):
        env = (
            FakeBlueFixedActionWrapper(
                labels=[
                    "Sleep",
                    "Analyse host_a",
                ],
                mask=[
                    False,
                    True,
                ],
            )
        )

        with self.assertRaises(
            RuntimeError
        ):
            self.adapter.resolve(
                env,
                "blue_agent_0",
                2,
                observable_host_scores=None,
            )


if __name__ == "__main__":
    unittest.main()