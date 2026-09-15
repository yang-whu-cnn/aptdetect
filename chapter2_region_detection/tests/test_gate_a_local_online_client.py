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

    # =========================================================
    # 1. reset observation schema
    # =========================================================
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

        # planner 不允许直接看到
        # local simulator 私有 latent state。
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

    # =========================================================
    # 2. reset reproducibility
    # =========================================================
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

    # =========================================================
    # 3. full trajectory reproducibility
    # =========================================================
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

        # A=4:
        # 0 no_op
        # 1 analyse
        # 2 remove
        # 3 restore
        actions = [
            1,
            2,
            3,
            0,
            2,
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

    # =========================================================
    # 4. no-op
    # =========================================================
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

        # no_op 本身没有主动响应效果，
        # 但自然攻击过程仍继续演化，
        # 所以前后状态不要求相同。
        self.assertNotEqual(
            before,
            after,
        )

    # =========================================================
    # 5. analyse
    # =========================================================
    def test_analyse_adds_visibility_not_direct_mitigation(
        self,
    ):
        no_op = self._make_client()
        analyse = self._make_client()

        no_op.reset(
            seed=321
        )

        analyse.reset(
            seed=321
        )

        # 使用完全相同随机轨迹，
        # 唯一差异是 action。
        no_op.step(
            0
        )

        analyse.step(
            1
        )

        base = (
            no_op._latent_state
        )

        got = (
            analyse._latent_state
        )

        # analyse 不直接消除 threat。
        self.assertAlmostEqual(
            got.auth_pressure,
            base.auth_pressure,
        )

        self.assertAlmostEqual(
            got.scan_pressure,
            base.scan_pressure,
        )

        self.assertAlmostEqual(
            got.process_pressure,
            base.process_pressure,
        )

        self.assertAlmostEqual(
            got.outbound_pressure,
            base.outbound_pressure,
        )

        # analyse 的主要作用：
        # 提升当前异常的可见性。
        self.assertGreater(
            got.visibility,
            base.visibility,
        )

    # =========================================================
    # 6. remove
    # =========================================================
    def test_remove_targets_user_level_compromise(
        self,
    ):
        no_op = self._make_client()
        remove = self._make_client()

        no_op.reset(
            seed=321
        )

        remove.reset(
            seed=321
        )

        no_op.step(
            0
        )

        remove.step(
            2
        )

        base = (
            no_op._latent_state
        )

        got = (
            remove._latent_state
        )

        # remove 主要用于
        # user-level compromise removal。
        self.assertLess(
            got.auth_pressure,
            base.auth_pressure,
        )

        self.assertLess(
            got.process_pressure,
            base.process_pressure,
        )

        # 不把 remove 当作
        # 全局风险统一衰减动作。
        self.assertAlmostEqual(
            got.scan_pressure,
            base.scan_pressure,
        )

        self.assertAlmostEqual(
            got.outbound_pressure,
            base.outbound_pressure,
        )

    # =========================================================
    # 7. restore
    # =========================================================
    def test_restore_targets_host_compromise(
        self,
    ):
        no_op = self._make_client()
        restore = self._make_client()

        no_op.reset(
            seed=321
        )

        restore.reset(
            seed=321
        )

        no_op.step(
            0
        )

        restore.step(
            3
        )

        base = (
            no_op._latent_state
        )

        got = (
            restore._latent_state
        )

        # host reimaging 应明显缓解
        # host compromise。
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

        # restore 主机不会让
        # 外部攻击者停止扫描。
        self.assertAlmostEqual(
            got.scan_pressure,
            base.scan_pressure,
        )

    # =========================================================
    # 8. old five-action ID must be rejected
    # =========================================================
    def test_removed_old_action_id_rejected(
        self,
    ):
        client = self._make_client()

        client.reset(
            seed=321
        )

        # A=4 后合法 ID 只有 0..3。
        #
        # 旧五动作 contract 中：
        # 4 = restore
        #
        # 现在必须明确失败，
        # 防止旧 replay/action ID 被静默复用。
        with self.assertRaises(
            ValueError
        ):
            client.step(
                4
            )

    # =========================================================
    # 9. reward boundary
    # =========================================================
    def test_reward_is_gate_b_placeholder_and_info_has_no_latent(
        self,
    ):
        client = self._make_client()

        client.reset(
            seed=5
        )

        _, reward, _, info = (
            client.step(
                2
            )
        )

        # A2R 阶段仍不在 local simulator
        # 中实现正式 response reward。
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

        # info 可以记录执行了哪个
        # high-level action，
        # 但不能泄露内部威胁状态。
        self.assertEqual(
            info["action_id"],
            2,
        )

        self.assertEqual(
            info["action_name"],
            "remove",
        )

    # =========================================================
    # 10. episode termination
    # =========================================================
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
            client.step(
                0
            )
        )

        _, _, done2, _ = (
            client.step(
                0
            )
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

    # =========================================================
    # 11. StateSummary compatibility
    # =========================================================
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

        # 当前 local-development state
        # 仍保持 D=8。
        self.assertEqual(
            len(FEATURE_KEYS),
            8,
        )

        self.assertTrue(
            np.isfinite(
                summary.vec
            ).all()
        )

        # 使用合法的新 action 2 = remove。
        obs2, _, _, _ = (
            client.step(
                2
            )
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

    # =========================================================
    # 12. region profile distinction
    # =========================================================
    def test_region_profiles_are_distinct(
        self,
    ):
        c0 = self._make_client(
            region_id=0
        )

        c1 = self._make_client(
            region_id=1
        )

        c0.reset(
            seed=11
        )

        c1.reset(
            seed=11
        )

        self.assertNotEqual(
            c0._latent_state,
            c1._latent_state,
        )

    # =========================================================
    # 13. invalid config
    # =========================================================
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