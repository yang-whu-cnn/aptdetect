import unittest

from shared.local_action_adapter import (
    LocalActionAdapter,
    LocalActionAdapterConfig,
    LocalThreatState,
)


class TestGateALocalActionAdapter(
    unittest.TestCase
):

    def setUp(
        self,
    ):
        self.adapter = (
            LocalActionAdapter()
        )

        self.state = LocalThreatState(
            auth_pressure=0.80,
            scan_pressure=0.60,
            process_pressure=0.70,
            outbound_pressure=0.90,
            visibility=0.20,
        )

    def test_no_op_has_no_active_effect(
        self,
    ):
        got = self.adapter.apply(
            self.state,
            0,
        )

        self.assertEqual(
            got,
            self.state,
        )

    def test_analyse_only_increases_visibility(
        self,
    ):
        got = self.adapter.apply(
            self.state,
            1,
        )

        self.assertEqual(
            got.auth_pressure,
            self.state.auth_pressure,
        )

        self.assertEqual(
            got.scan_pressure,
            self.state.scan_pressure,
        )

        self.assertEqual(
            got.process_pressure,
            self.state.process_pressure,
        )

        self.assertEqual(
            got.outbound_pressure,
            self.state.outbound_pressure,
        )

        self.assertGreater(
            got.visibility,
            self.state.visibility,
        )

    def test_remove_targets_user_compromise(
        self,
    ):
        got = self.adapter.apply(
            self.state,
            2,
        )

        self.assertLess(
            got.auth_pressure,
            self.state.auth_pressure,
        )

        self.assertLess(
            got.process_pressure,
            self.state.process_pressure,
        )

        self.assertEqual(
            got.scan_pressure,
            self.state.scan_pressure,
        )

        self.assertEqual(
            got.outbound_pressure,
            self.state.outbound_pressure,
        )

    def test_restore_targets_host_compromise(
        self,
    ):
        got = self.adapter.apply(
            self.state,
            3,
        )

        self.assertLess(
            got.auth_pressure,
            self.state.auth_pressure,
        )

        self.assertLess(
            got.process_pressure,
            self.state.process_pressure,
        )

        self.assertLess(
            got.outbound_pressure,
            self.state.outbound_pressure,
        )

        self.assertEqual(
            got.scan_pressure,
            self.state.scan_pressure,
        )

    def test_actions_are_semantically_distinct_not_strength_levels(
        self,
    ):
        analyse = self.adapter.apply(
            self.state,
            1,
        )

        remove = self.adapter.apply(
            self.state,
            2,
        )

        restore = self.adapter.apply(
            self.state,
            3,
        )

        # analyse 只增加 visibility。
        self.assertEqual(
            analyse.auth_pressure,
            self.state.auth_pressure,
        )

        self.assertGreater(
            analyse.visibility,
            self.state.visibility,
        )

        # remove 针对用户级 compromise，
        # 不直接改变 outbound。
        self.assertLess(
            remove.auth_pressure,
            self.state.auth_pressure,
        )

        self.assertEqual(
            remove.outbound_pressure,
            self.state.outbound_pressure,
        )

        # restore 是 host reimage，
        # 会影响 outbound，
        # 但不清除外部 scan pressure。
        self.assertLess(
            restore.outbound_pressure,
            self.state.outbound_pressure,
        )

        self.assertEqual(
            restore.scan_pressure,
            self.state.scan_pressure,
        )

    def test_input_state_is_immutable(
        self,
    ):
        original = self.state

        _ = self.adapter.apply(
            self.state,
            3,
        )

        self.assertEqual(
            self.state,
            original,
        )

    def test_removed_old_action_id_rejected(
        self,
    ):
        with self.assertRaises(
            ValueError
        ):
            self.adapter.apply(
                self.state,
                4,
            )

    def test_invalid_state_rejected(
        self,
    ):
        bad = LocalThreatState(
            auth_pressure=1.2,
            scan_pressure=0.5,
            process_pressure=0.5,
            outbound_pressure=0.5,
            visibility=0.5,
        )

        with self.assertRaises(
            ValueError
        ):
            self.adapter.apply(
                bad,
                0,
            )

    def test_invalid_config_rejected(
        self,
    ):
        with self.assertRaises(
            ValueError
        ):
            LocalActionAdapter(
                LocalActionAdapterConfig(
                    remove_auth_multiplier=1.5,
                )
            )


if __name__ == "__main__":
    unittest.main()