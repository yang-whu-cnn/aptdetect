import unittest

import torch

from baselines.ug_cem_apt.uncertainty import (
    UGUncertainty,
    UGUncertaintyConfig,
)


class TestUGUncertainty(
    unittest.TestCase
):

    def test_identical_members_have_zero_uncertainty(
        self,
    ):
        ug = UGUncertainty(
            UGUncertaintyConfig()
        )

        # base:
        # [H,N,D]
        base = torch.randn(
            4,
            6,
            8,
        )

        # 每个 ensemble member
        # 预测完全相同。
        #
        # [H,N,D]
        # ->
        # [H,N,M,D]
        next_states = (
            base
            .unsqueeze(2)
            .repeat(
                1,
                1,
                5,
                1,
            )
        )

        uncertainty = ug.compute(
            next_states
        )

        torch.testing.assert_close(
            uncertainty,
            torch.zeros(6),
            atol=1e-6,
            rtol=0,
        )

    def test_more_member_disagreement_means_more_uncertainty(
        self,
    ):
        cfg = UGUncertaintyConfig()

        low_ug = UGUncertainty(
            cfg
        )

        high_ug = UGUncertainty(
            cfg
        )

        # [M]
        low_offsets = torch.tensor(
            [
                -0.10,
                -0.05,
                0.00,
                0.05,
                0.10,
            ]
        )

        high_offsets = (
            10.0
            * low_offsets
        )

        # [H,N,M,D]
        low = (
            low_offsets
            .view(
                1,
                1,
                5,
                1,
            )
            .expand(
                4,
                6,
                5,
                8,
            )
            .clone()
        )

        high = (
            high_offsets
            .view(
                1,
                1,
                5,
                1,
            )
            .expand(
                4,
                6,
                5,
                8,
            )
            .clone()
        )

        # 不更新 statistics，
        # 两边使用完全相同的初始 normalizer，
        # 这样比较只反映 member disagreement。
        low_u = low_ug.compute(
            low,
            update_stats=False,
        )

        high_u = high_ug.compute(
            high,
            update_stats=False,
        )

        self.assertGreater(
            float(high_u.mean()),
            float(low_u.mean()),
        )

    def test_output_shape(
        self,
    ):
        ug = UGUncertainty(
            UGUncertaintyConfig()
        )

        next_states = torch.randn(
            4,
            7,
            5,
            8,
        )

        uncertainty = ug.compute(
            next_states
        )

        self.assertEqual(
            tuple(
                uncertainty.shape
            ),
            (7,),
        )

    def test_uncertainty_is_finite(
        self,
    ):
        ug = UGUncertainty(
            UGUncertaintyConfig()
        )

        # 连续多次调用，
        # 检查 running normalizer
        # 不会制造 NaN / Inf。
        for _ in range(30):
            next_states = torch.randn(
                4,
                16,
                5,
                8,
            )

            uncertainty = ug.compute(
                next_states
            )

            self.assertTrue(
                bool(
                    torch.isfinite(
                        uncertainty
                    ).all()
                )
            )

    def test_running_statistics_update_correctly(
        self,
    ):
        alpha = 0.2
        eps = 1e-8

        ug = UGUncertainty(
            UGUncertaintyConfig(
                alpha=alpha,
                eps=eps,
            )
        )

        next_states = torch.arange(
            2 * 3 * 4 * 2,
            dtype=torch.float32,
        ).reshape(
            2,
            3,
            4,
            2,
        )

        # 官方初始化
        old_obs_mean = torch.zeros(
            2
        )

        old_obs_std = torch.full(
            (2,),
            0.1,
        )

        old_horizon_std = torch.full(
            (2,),
            0.01,
        )

        batch_mean = (
            next_states.mean(
                dim=(0, 1, 2)
            )
        )

        batch_std = (
            next_states.std(
                dim=(0, 1, 2)
            )
        )

        expected_obs_mean = (
            (1.0 - alpha)
            * old_obs_mean
            +
            alpha
            * batch_mean
        )

        expected_obs_std = (
            (1.0 - alpha)
            * old_obs_std
            +
            alpha
            * batch_std
        )

        normalized = (
            next_states
            - expected_obs_mean.view(
                1,
                1,
                1,
                2,
            )
        ) / (
            expected_obs_std
            .clamp_min(eps)
            .view(
                1,
                1,
                1,
                2,
            )
        )

        next_obs_std = (
            normalized.std(
                dim=2
            )
            .mean(
                dim=2
            )
        )

        horizon_batch = (
            next_obs_std.mean(
                dim=1
            )
        )

        expected_horizon_std = (
            (1.0 - alpha)
            * old_horizon_std
            +
            alpha
            * horizon_batch
        )

        ug.compute(
            next_states,
            update_stats=True,
        )

        torch.testing.assert_close(
            ug.obs_mean,
            expected_obs_mean,
        )

        torch.testing.assert_close(
            ug.obs_std,
            expected_obs_std,
        )

        torch.testing.assert_close(
            ug.horizon_std,
            expected_horizon_std,
        )

    def test_reset(
        self,
    ):
        ug = UGUncertainty(
            UGUncertaintyConfig()
        )

        ug.compute(
            torch.randn(
                4,
                8,
                5,
                8,
            )
        )

        self.assertIsNotNone(
            ug.obs_mean
        )

        self.assertIsNotNone(
            ug.obs_std
        )

        self.assertIsNotNone(
            ug.horizon_std
        )

        ug.reset()

        self.assertIsNone(
            ug.obs_mean
        )

        self.assertIsNone(
            ug.obs_std
        )

        self.assertIsNone(
            ug.horizon_std
        )

    def test_update_stats_false_keeps_statistics_unchanged(
        self,
    ):
        ug = UGUncertainty(
            UGUncertaintyConfig()
        )

        # 先正常初始化并更新一次。
        ug.compute(
            torch.randn(
                4,
                8,
                5,
                8,
            ),
            update_stats=True,
        )

        old_mean = (
            ug.obs_mean.clone()
        )

        old_std = (
            ug.obs_std.clone()
        )

        old_horizon_std = (
            ug.horizon_std.clone()
        )

        # 输入完全不同的数据，
        # 但禁止更新 statistics。
        ug.compute(
            torch.randn(
                4,
                8,
                5,
                8,
            )
            * 100.0,
            update_stats=False,
        )

        torch.testing.assert_close(
            ug.obs_mean,
            old_mean,
        )

        torch.testing.assert_close(
            ug.obs_std,
            old_std,
        )

        torch.testing.assert_close(
            ug.horizon_std,
            old_horizon_std,
        )

    def test_update_stats_false_uses_official_initial_values(
        self,
    ):
        ug = UGUncertainty(
            UGUncertaintyConfig()
        )

        ug.compute(
            torch.randn(
                4,
                8,
                5,
                8,
            ),
            update_stats=False,
        )

        torch.testing.assert_close(
            ug.obs_mean,
            torch.zeros(8),
        )

        torch.testing.assert_close(
            ug.obs_std,
            torch.full(
                (8,),
                0.1,
            ),
        )

        torch.testing.assert_close(
            ug.horizon_std,
            torch.full(
                (4,),
                0.01,
            ),
        )

    def test_matches_official_reference_formula(
            self,
    ):
        alpha = 0.2
        eps = 1e-8

        ug = UGUncertainty(
            UGUncertaintyConfig(
                alpha=alpha,
                eps=eps,
            )
        )

        # canonical layout:
        #
        # [H,N,M,D]
        next_states = torch.arange(
            2 * 3 * 4 * 2,
            dtype=torch.float32,
        ).reshape(
            2,
            3,
            4,
            2,
        )

        H, N, M, D = (
            next_states.shape
        )

        # --------------------------------
        # 以下直接模拟 UG 官方源码
        # --------------------------------

        # 官方在 reshape 之前的 next_obs
        # 可以视为：
        #
        # [H, N*M, D]
        official_next_obs = (
            next_states.reshape(
                H,
                N * M,
                D,
            )
        )

        # 官方初始化值
        obs_mean = torch.zeros(
            D
        )

        obs_std = torch.full(
            (D,),
            0.1,
        )

        horizon_std = torch.full(
            (H,),
            0.01,
        )

        # 官方：
        #
        # next_obs.mean(dim=(0,1))
        # next_obs.std(dim=(0,1))
        obs_mean = (
                (1.0 - alpha)
                * obs_mean
                +
                alpha
                * official_next_obs.mean(
            dim=(0, 1)
        )
        )

        obs_std = (
                (1.0 - alpha)
                * obs_std
                +
                alpha
                * official_next_obs.std(
            dim=(0, 1)
        )
        )

        # 官方 reshape：
        #
        # [H, population, particle, D]
        official_next_obs = (
            official_next_obs.reshape(
                H,
                N,
                M,
                D,
            )
        )

        next_obs_norm = (
                                official_next_obs
                                - obs_mean
                        ) / obs_std.clamp_min(
            eps
        )

        # 官方：
        #
        # std(dim=particle/model)
        # mean(dim=state)
        next_obs_std = (
            next_obs_norm.std(
                dim=2
            )
            .mean(
                dim=2
            )
        )

        # [H]
        horizon_std = (
                (1.0 - alpha)
                * horizon_std
                +
                alpha
                * next_obs_std.mean(
            dim=1
        )
        )

        # 官方：
        #
        # self.next_obs_std
        # /= self.horizon_std.unsqueeze(1)
        normalized_std = (
                next_obs_std
                /
                horizon_std
                .clamp_min(eps)
                .unsqueeze(1)
        )

        # 官方：
        #
        # pop_std =
        # self.next_obs_std.mean(dim=0)
        expected_uncertainty = (
            normalized_std.mean(
                dim=0
            )
        )

        got = ug.compute(
            next_states,
            update_stats=True,
        )

        torch.testing.assert_close(
            got,
            expected_uncertainty,
            atol=1e-6,
            rtol=1e-6,
        )

        torch.testing.assert_close(
            ug.obs_mean,
            obs_mean,
            atol=1e-6,
            rtol=1e-6,
        )

        torch.testing.assert_close(
            ug.obs_std,
            obs_std,
            atol=1e-6,
            rtol=1e-6,
        )

        torch.testing.assert_close(
            ug.horizon_std,
            horizon_std,
            atol=1e-6,
            rtol=1e-6,
        )

    def test_horizon_or_state_dim_change_raises(
            self,
    ):
        # -----------------------------
        # H 发生改变
        # -----------------------------
        ug_h = UGUncertainty(
            UGUncertaintyConfig()
        )

        ug_h.compute(
            torch.randn(
                4,
                8,
                5,
                8,
            )
        )

        with self.assertRaises(
                ValueError
        ):
            ug_h.compute(
                torch.randn(
                    3,
                    8,
                    5,
                    8,
                )
            )

        # -----------------------------
        # D 发生改变
        # -----------------------------
        ug_d = UGUncertainty(
            UGUncertaintyConfig()
        )

        ug_d.compute(
            torch.randn(
                4,
                8,
                5,
                8,
            )
        )

        with self.assertRaises(
                ValueError
        ):
            ug_d.compute(
                torch.randn(
                    4,
                    8,
                    5,
                    7,
                )
            )

    def test_invalid_config(
            self,
    ):
        with self.assertRaises(
                ValueError
        ):
            UGUncertainty(
                UGUncertaintyConfig(
                    alpha=-0.1
                )
            )

        with self.assertRaises(
                ValueError
        ):
            UGUncertainty(
                UGUncertaintyConfig(
                    alpha=1.1
                )
            )

        with self.assertRaises(
                ValueError
        ):
            UGUncertainty(
                UGUncertaintyConfig(
                    alpha=float("nan")
                )
            )

        with self.assertRaises(
                ValueError
        ):
            UGUncertainty(
                UGUncertaintyConfig(
                    eps=0.0
                )
            )

        with self.assertRaises(
                ValueError
        ):
            UGUncertainty(
                UGUncertaintyConfig(
                    eps=float("nan")
                )
            )


    def test_invalid_input(
        self,
    ):
        ug = UGUncertainty(
            UGUncertaintyConfig()
        )

        # 不是 [H,N,M,D]
        with self.assertRaises(
            ValueError
        ):
            ug.compute(
                torch.randn(
                    4,
                    5,
                    8,
                )
            )

        # M = 1，
        # 无法计算 ensemble disagreement。
        with self.assertRaises(
            ValueError
        ):
            ug.compute(
                torch.randn(
                    4,
                    8,
                    1,
                    8,
                )
            )

        # 非有限输入
        bad = torch.randn(
            4,
            8,
            5,
            8,
        )

        bad[
            0,
            0,
            0,
            0,
        ] = float(
            "nan"
        )

        with self.assertRaises(
            ValueError
        ):
            ug.compute(
                bad
            )


if __name__ == "__main__":
    unittest.main()