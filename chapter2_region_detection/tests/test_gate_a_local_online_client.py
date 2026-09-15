import unittest

import numpy as np

from shared.local_online_client import (
    LocalOnlineConfig,
    SharedLocalOnlineClient,
)

from src.state_summary import (
    FEATURE_KEYS,
    SlidingWindowSummarizer,
)


class TestGateALocalOnlineClient(
    unittest.TestCase
):

    def _make_client(
        self,
        seed=123,
        region_id=0,
        max_steps=10,
    ):
        return SharedLocalOnlineClient(
            LocalOnlineConfig(
                seed=seed,
                region_id=region_id,
                max_steps=max_steps,
            )
        )

    def test_reset_observation_schema_and_no_latent_leak(
        self,
    ):
        client = self._make_client()

        obs = client.reset(
            seed=7
        )

        self.assertIn(
            "t",
            obs,
        )
        self.assertIn(
            "events",
            obs,
        )
        self.assertIn(
            "risk_proxy",
            obs,
        )
        self.assertIn(
            "region_id",
            obs,
        )
        self.assertIn(
            "region_name",
            obs,
        )

        self.assertEqual(
            obs["t"],
            0,
        )

        self.assertGreaterEqual(
            obs["risk_proxy"],
            0.0,
        )
        self.assertLessEqual(
            obs["risk_proxy"],
            1.0,
        )

        forbidden = {
            "auth_pressure",
            "scan_pressure",
            "process_pressure",
            "outbound_pressure",
            "visibility",
            "latent_state",
        }

        self.assertTrue(
            forbidden.isdisjoint(
                obs.keys()
            )
        )

    def test_same_seed_same_reset(
        self,
    ):
        c1 = self._make_client(
            seed=10
        )
        c2 = self._make_client(
            seed=10
        )

        obs1 = c1.reset(
            seed=99
        )
        obs2 = c2.reset(
            seed=99
        )

        self.assertEqual(
            obs1,
            obs2,
        )

    def test_same_seed_same_action_sequence(
        self,
    ):
        c1 = self._make_client(
            seed=10
        )
        c2 = self._make_client(
            seed=10
        )

        self.assertEqual(
            c1.reset(seed=55),
            c2.reset(seed=55),
        )

        actions = [
            1,
            2,
            3,
            0,
            4,
        ]

        for action_id in actions:
            out1 = c1.step(
                action_id
            )
            out2 = c2.step(
                action_id
            )

            self.assertEqual(
                out1,
                out2,
            )

    def test_no_op_keeps_only_natural_dynamics(
        self,
    ):
        client = self._make_client()

        client.reset(
            seed=123
        )

        before = (
            client._latent_state
        )

        client.step(
            0
        )

        after = (
            client._latent_state
        )

        # no_op 本身没有主动处置，
        # 但环境自然攻击过程仍然前进。
        self.assertNotEqual(
            before,
            after,
        )

    def test_analyse_adds_visibility_not_direct_mitigation(
        self,
    ):
        no_op = self._make_client()
        analyse = self._make_client()

        no_op.reset(seed=321)
        analyse.reset(seed=321)

        no_op.step(0)
        analyse.step(1)

        s0 = no_op._latent_state
        s1 = analyse._latent_state

        self.assertAlmostEqual(
            s0.auth_pressure,
            s1.auth_pressure,
        )
        self.assertAlmostEqual(
            s0.scan_pressure,
            s1.scan_pressure,
        )
        self.assertAlmostEqual(
            s0.process_pressure,
            s1.process_pressure,
        )
        self.assertAlmostEqual(
            s0.outbound_pressure,
            s1.outbound_pressure,
        )

        self.assertGreater(
            s1.visibility,
            s0.visibility,
        )

    def test_control_traffic_targets_network_components(
        self,
    ):
        no_op = self._make_client()
        control = self._make_client()

        no_op.reset(seed=321)
        control.reset(seed=321)

        no_op.step(0)
        control.step(2)

        base = no_op._latent_state
        got = control._latent_state

        self.assertAlmostEqual(
            got.auth_pressure,
            base.auth_pressure,
        )
        self.assertAlmostEqual(
            got.process_pressure,
            base.process_pressure,
        )

        self.assertLess(
            got.scan_pressure,
            base.scan_pressure,
        )
        self.assertLess(
            got.outbound_pressure,
            base.outbound_pressure,
        )

    def test_remove_targets_user_level_compromise(
        self,
    ):
        no_op = self._make_client()
        remove = self._make_client()

        no_op.reset(seed=321)
        remove.reset(seed=321)

        no_op.step(0)
        remove.step(3)

        base = no_op._latent_state
        got = remove._latent_state

        self.assertLess(
            got.auth_pressure,
            base.auth_pressure,
        )
        self.assertLess(
            got.process_pressure,
            base.process_pressure,
        )

        self.assertAlmostEqual(
            got.scan_pressure,
            base.scan_pressure,
        )
        self.assertAlmostEqual(
            got.outbound_pressure,
            base.outbound_pressure,
        )

    def test_restore_targets_host_compromise(
        self,
    ):
        no_op = self._make_client()
        restore = self._make_client()

        no_op.reset(seed=321)
        restore.reset(seed=321)

        no_op.step(0)
        restore.step(4)

        base = no_op._latent_state
        got = restore._latent_state

        self.assertLess(
            got.auth_pressure,
            base.auth_pressure,
        )
        self.assertLess(
            got.process_pressure,
            base.process_pressure,
        )
        self.assertLess(
            got.outbound_pressure,
            base.outbound_pressure,
        )

        # host reimage 不会让外部扫描者消失。
        self.assertAlmostEqual(
            got.scan_pressure,
            base.scan_pressure,
        )

    def test_reward_is_gate_b_placeholder_and_info_has_no_latent(
        self,
    ):
        client = self._make_client()
        client.reset(seed=5)

        _, reward, _, info = (
            client.step(2)
        )

        self.assertEqual(
            reward,
            0.0,
        )

        self.assertEqual(
            info["reward_status"],
            "placeholder_zero_until_gate_b",
        )

        forbidden = {
            "auth_pressure",
            "scan_pressure",
            "process_pressure",
            "outbound_pressure",
            "visibility",
            "latent_state",
        }

        self.assertTrue(
            forbidden.isdisjoint(
                info.keys()
            )
        )

    def test_done_after_max_steps(
        self,
    ):
        client = self._make_client(
            max_steps=2
        )

        client.reset(
            seed=8
        )

        _, _, done1, _ = (
            client.step(0)
        )

        _, _, done2, _ = (
            client.step(0)
        )

        self.assertFalse(
            done1
        )
        self.assertTrue(
            done2
        )

        with self.assertRaises(
            RuntimeError
        ):
            client.step(
                0
            )

    def test_observation_is_compatible_with_state_summary(
        self,
    ):
        client = self._make_client()

        summarizer = (
            SlidingWindowSummarizer(
                window_size=20,
                max_text_len=512,
            )
        )

        obs = client.reset(
            seed=77
        )

        summarizer.update(
            obs["events"]
        )

        summary = summarizer.build(
            risk_proxy=obs[
                "risk_proxy"
            ]
        )

        self.assertEqual(
            summary.vec.shape,
            (
                len(FEATURE_KEYS),
            ),
        )

        self.assertEqual(
            len(FEATURE_KEYS),
            8,
        )

        self.assertTrue(
            np.isfinite(
                summary.vec
            ).all()
        )

        obs2, _, _, _ = (
            client.step(2)
        )

        summarizer.update(
            obs2["events"]
        )

        summary2 = summarizer.build(
            risk_proxy=obs2[
                "risk_proxy"
            ]
        )

        self.assertEqual(
            summary2.vec.shape,
            (8,),
        )

        self.assertTrue(
            np.isfinite(
                summary2.vec
            ).all()
        )

    def test_region_profiles_are_distinct(
        self,
    ):
        c0 = self._make_client(
            region_id=0
        )
        c1 = self._make_client(
            region_id=1
        )

        c0.reset(seed=11)
        c1.reset(seed=11)

        self.assertNotEqual(
            c0._latent_state,
            c1._latent_state,
        )

    def test_invalid_config_rejected(
        self,
    ):
        with self.assertRaises(
            ValueError
        ):
            SharedLocalOnlineClient(
                LocalOnlineConfig(
                    max_steps=0
                )
            )

        with self.assertRaises(
            ValueError
        ):
            SharedLocalOnlineClient(
                LocalOnlineConfig(
                    region_id=99
                )
            )

        with self.assertRaises(
            ValueError
        ):
            SharedLocalOnlineClient(
                LocalOnlineConfig(
                    visibility_floor=1.5
                )
            )


if __name__ == "__main__":
    unittest.main()