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

    def test_control_traffic_targets_network(
        self,
    ):
        got = self.adapter.apply(
            self.state,
            2,
        )

        self.assertEqual(
            got.auth_pressure,
            self.state.auth_pressure,
        )

        self.assertEqual(
            got.process_pressure,
            self.state.process_pressure,
        )

        self.assertLess(
            got.scan_pressure,
            self.state.scan_pressure,
        )

        self.assertLess(
            got.outbound_pressure,
            self.state.outbound_pressure,
        )

    def test_remove_targets_user_compromise(
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
            4,
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

        # 外部扫描压力不应因为
        # reimage host 自动消失。
        self.assertEqual(
            got.scan_pressure,
            self.state.scan_pressure,
        )

    def test_actions_are_not_numeric_strength_levels(
        self,
    ):
        analyse = self.adapter.apply(
            self.state,
            1,
        )

        control = self.adapter.apply(
            self.state,
            2,
        )

        remove = self.adapter.apply(
            self.state,
            3,
        )

        # action 2 并不是 action 1
        # 的“更强版本”。
        self.assertEqual(
            analyse.outbound_pressure,
            self.state.outbound_pressure,
        )

        self.assertLess(
            control.outbound_pressure,
            self.state.outbound_pressure,
        )

        # action 3 也不是简单继续增强
        # action 2 的网络效果。
        self.assertEqual(
            remove.outbound_pressure,
            self.state.outbound_pressure,
        )

    def test_input_state_is_immutable(
        self,
    ):
        original = self.state

        _ = self.adapter.apply(
            self.state,
            4,
        )

        self.assertEqual(
            self.state,
            original,
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
                    control_outbound_multiplier=1.5,
                )
            )


if __name__ == "__main__":
    unittest.main()