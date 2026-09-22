import unittest

import torch

from baselines.ug_cem_apt.categorical_cem import (
    CategoricalCEMConfig,
    CategoricalCEMOptimizer,
)


class TestCategoricalCEM(
    unittest.TestCase
):

    def test_uniform_probability_shape_and_sum(
        self,
    ):
        cem = CategoricalCEMOptimizer(
            CategoricalCEMConfig()
        )

        self.assertEqual(
            tuple(
                cem.uniform_probs.shape
            ),
            (4, 5),
        )

        torch.testing.assert_close(
            cem.uniform_probs.sum(
                dim=1
            ),
            torch.ones(4),
            atol=1e-7,
            rtol=0,
        )

    def test_sample_shape_dtype_range(
        self,
    ):
        cem = CategoricalCEMOptimizer(
            CategoricalCEMConfig(
                population_size=32
            )
        )

        plans = (
            cem._sample_population(
                cem.uniform_probs
            )
        )

        self.assertEqual(
            tuple(plans.shape),
            (32, 4),
        )

        self.assertEqual(
            plans.dtype,
            torch.long,
        )

        self.assertGreaterEqual(
            int(plans.min()),
            0,
        )

        self.assertLess(
            int(plans.max()),
            5,
        )

    def test_elite_num_uses_ceil(
        self,
    ):
        cem = CategoricalCEMOptimizer(
            CategoricalCEMConfig(
                population_size=10,
                elite_ratio=0.21,
            )
        )

        # ceil(10 * 0.21)
        # =
        # ceil(2.1)
        # =
        # 3
        self.assertEqual(
            cem.elite_num,
            3,
        )

    def test_elite_frequency(
        self,
    ):
        cem = CategoricalCEMOptimizer(
            CategoricalCEMConfig(
                horizon=2,
                n_actions=3,
                prob_floor=0.0,
            )
        )

        elite_plans = torch.tensor(
            [
                [0, 1],
                [0, 2],
                [1, 2],
                [0, 2],
            ],
            dtype=torch.long,
        )

        got = cem._elite_frequencies(
            elite_plans
        )

        expected = torch.tensor(
            [
                [
                    0.75,
                    0.25,
                    0.00,
                ],
                [
                    0.00,
                    0.25,
                    0.75,
                ],
            ]
        )

        torch.testing.assert_close(
            got,
            expected,
        )

    def test_probability_floor_is_normalized_and_idempotent(
        self,
    ):
        cem = CategoricalCEMOptimizer(
            CategoricalCEMConfig(
                horizon=2,
                n_actions=5,
                prob_floor=0.01,
            )
        )

        probs = torch.tensor(
            [
                [
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.2,
                    0.2,
                    0.2,
                    0.2,
                    0.2,
                ],
            ]
        )

        q = (
            cem._apply_probability_floor(
                probs
            )
        )

        self.assertTrue(
            bool(
                (
                    q
                    >= 0.01 - 1e-7
                ).all()
            )
        )

        torch.testing.assert_close(
            q.sum(
                dim=1
            ),
            torch.ones(2),
            atol=1e-7,
            rtol=0,
        )

        # 再执行一次 floor，
        # 概率不应该继续改变。
        q2 = (
            cem._apply_probability_floor(
                q
            )
        )

        torch.testing.assert_close(
            q2,
            q,
            atol=1e-7,
            rtol=0,
        )

    def test_alpha_semantics(
        self,
    ):
        old_probs = torch.tensor(
            [
                [
                    0.5,
                    0.3,
                    0.2,
                ],
                [
                    0.2,
                    0.3,
                    0.5,
                ],
            ]
        )

        elite_freq = torch.tensor(
            [
                [
                    0.0,
                    1.0,
                    0.0,
                ],
                [
                    1.0,
                    0.0,
                    0.0,
                ],
            ]
        )

        # alpha = 0
        #
        # new = elite
        cem_alpha0 = (
            CategoricalCEMOptimizer(
                CategoricalCEMConfig(
                    horizon=2,
                    n_actions=3,
                    alpha=0.0,
                    prob_floor=0.0,
                )
            )
        )

        got0 = (
            cem_alpha0._update_probs(
                old_probs,
                elite_freq,
            )
        )

        torch.testing.assert_close(
            got0,
            elite_freq,
        )

        # alpha = 1
        #
        # new = old
        cem_alpha1 = (
            CategoricalCEMOptimizer(
                CategoricalCEMConfig(
                    horizon=2,
                    n_actions=3,
                    alpha=1.0,
                    prob_floor=0.0,
                )
            )
        )

        got1 = (
            cem_alpha1._update_probs(
                old_probs,
                elite_freq,
            )
        )

        torch.testing.assert_close(
            got1,
            old_probs,
        )

    def test_reproducibility(
        self,
    ):
        cfg = CategoricalCEMConfig(
            population_size=128,
            num_iterations=5,
            seed=123,
        )

        target = torch.tensor(
            [1, 2, 3, 4],
            dtype=torch.long,
        )

        def objective(
            plans,
            iteration,
        ):
            return (
                plans
                == target.to(
                    plans.device
                )
            ).sum(
                dim=1
            ).float()

        cem1 = (
            CategoricalCEMOptimizer(
                cfg
            )
        )

        cem2 = (
            CategoricalCEMOptimizer(
                cfg
            )
        )

        r1 = cem1.optimize(
            objective
        )

        r2 = cem2.optimize(
            objective
        )

        torch.testing.assert_close(
            r1.best_plan,
            r2.best_plan,
        )

        self.assertEqual(
            r1.best_score,
            r2.best_score,
        )

        torch.testing.assert_close(
            r1.final_probs,
            r2.final_probs,
        )

    def test_finds_target(
        self,
    ):
        target = torch.tensor(
            [4, 3, 2, 1],
            dtype=torch.long,
        )

        cfg = CategoricalCEMConfig(
            horizon=4,
            n_actions=5,
            population_size=256,
            num_iterations=6,
            elite_ratio=0.20,
            alpha=0.10,
            prob_floor=0.01,
            seed=42,
        )

        cem = (
            CategoricalCEMOptimizer(
                cfg
            )
        )

        def objective(
            plans,
            iteration,
        ):
            # 每个位置答对一个动作，
            # 加 1 分。
            #
            # 唯一最高分：
            #
            # [4,3,2,1]
            #
            return (
                plans
                == target.to(
                    plans.device
                )
            ).sum(
                dim=1
            ).float()

        result = cem.optimize(
            objective
        )

        torch.testing.assert_close(
            result.best_plan,
            target,
        )

        self.assertEqual(
            result.best_score,
            4.0,
        )

    def test_initial_probs_control_first_sampling(
        self,
    ):
        cfg = CategoricalCEMConfig(
            horizon=4,
            n_actions=5,
            population_size=20,
            num_iterations=1,
            prob_floor=0.0,
            seed=7,
        )

        cem = (
            CategoricalCEMOptimizer(
                cfg
            )
        )

        # 所有位置强制 action=2
        initial_probs = torch.zeros(
            4,
            5,
        )

        initial_probs[:, 2] = 1.0

        seen = {}

        def objective(
            plans,
            iteration,
        ):
            seen[
                "plans"
            ] = plans.clone()

            return torch.zeros(
                plans.shape[0],
                device=plans.device,
            )

        cem.optimize(
            objective,
            initial_probs=(
                initial_probs
            ),
        )

        self.assertTrue(
            bool(
                (
                    seen["plans"]
                    == 2
                ).all()
            )
        )

    def test_no_implicit_final_probs_carryover(
        self,
    ):
        cfg = CategoricalCEMConfig(
            horizon=4,
            n_actions=5,
            population_size=64,
            num_iterations=1,
            prob_floor=0.0,
            seed=9,
        )

        cem = (
            CategoricalCEMOptimizer(
                cfg
            )
        )

        forced = torch.zeros(
            4,
            5,
        )

        forced[:, 4] = 1.0

        # 第一次：
        # 强制从 action=4 开始
        cem.optimize(
            lambda plans, iteration:
            torch.zeros(
                plans.shape[0],
                device=plans.device,
            ),
            initial_probs=forced,
        )

        seen = {}

        def second_objective(
            plans,
            iteration,
        ):
            seen[
                "plans"
            ] = plans.clone()

            return torch.zeros(
                plans.shape[0],
                device=plans.device,
            )

        # 第二次没有传 initial_probs。
        #
        # 应重新从 uniform 开始，
        # 而不是自动继承上次 final_probs。
        cem.optimize(
            second_objective
        )

        self.assertFalse(
            bool(
                (
                    seen["plans"]
                    == 4
                ).all()
            )
        )

    def test_iteration_is_passed(
        self,
    ):
        cfg = CategoricalCEMConfig(
            num_iterations=4,
            population_size=8,
        )

        cem = (
            CategoricalCEMOptimizer(
                cfg
            )
        )

        seen_iterations = []

        def objective(
            plans,
            iteration,
        ):
            seen_iterations.append(
                iteration
            )

            return torch.zeros(
                plans.shape[0],
                device=plans.device,
            )

        cem.optimize(
            objective
        )

        self.assertEqual(
            seen_iterations,
            [
                0,
                1,
                2,
                3,
            ],
        )

    def test_invalid_config(
        self,
    ):
        with self.assertRaises(
            ValueError
        ):
            CategoricalCEMOptimizer(
                CategoricalCEMConfig(
                    horizon=0
                )
            )

        with self.assertRaises(
            ValueError
        ):
            CategoricalCEMOptimizer(
                CategoricalCEMConfig(
                    elite_ratio=float(
                        "nan"
                    )
                )
            )

        # 5 * 0.2 = 1
        #
        # 不再有剩余概率质量
        with self.assertRaises(
            ValueError
        ):
            CategoricalCEMOptimizer(
                CategoricalCEMConfig(
                    n_actions=5,
                    prob_floor=0.2,
                )
            )

    def test_invalid_initial_probs(
        self,
    ):
        cem = (
            CategoricalCEMOptimizer(
                CategoricalCEMConfig()
            )
        )

        def objective(
            plans,
            iteration,
        ):
            return torch.zeros(
                plans.shape[0],
                device=plans.device,
            )

        # shape 错误
        with self.assertRaises(
            ValueError
        ):
            cem.optimize(
                objective,
                torch.ones(
                    3,
                    5,
                ),
            )

        # 负概率
        negative = torch.ones(
            4,
            5,
        )

        negative[
            0,
            0,
        ] = -1.0

        with self.assertRaises(
            ValueError
        ):
            cem.optimize(
                objective,
                negative,
            )

        # 某一行全 0
        zero_row = torch.ones(
            4,
            5,
        )

        zero_row[0] = 0.0

        with self.assertRaises(
            ValueError
        ):
            cem.optimize(
                objective,
                zero_row,
            )

        # NaN
        nan_probs = torch.ones(
            4,
            5,
        )

        nan_probs[
            0,
            0,
        ] = float(
            "nan"
        )

        with self.assertRaises(
            ValueError
        ):
            cem.optimize(
                objective,
                nan_probs,
            )

    def test_bad_objective_shape(
        self,
    ):
        cem = (
            CategoricalCEMOptimizer(
                CategoricalCEMConfig(
                    population_size=8
                )
            )
        )

        def bad_objective(
            plans,
            iteration,
        ):
            return torch.zeros(
                8,
                1,
            )

        with self.assertRaises(
            ValueError
        ):
            cem.optimize(
                bad_objective
            )

    def test_too_few_finite_scores_for_elites(
            self,
    ):
        cfg = CategoricalCEMConfig(
            population_size=8,
            num_iterations=1,
            elite_ratio=0.5,
            seed=0,
        )

        cem = CategoricalCEMOptimizer(
            cfg
        )

        # elite_num =
        # ceil(8 * 0.5)
        # = 4
        #
        # 但这里只提供2个finite score
        def objective(
                plans,
                iteration,
        ):
            return torch.tensor(
                [
                    1.0,
                    0.5,
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                ],
                device=plans.device,
            )

        with self.assertRaises(
                ValueError
        ):
            cem.optimize(
                objective
            )

    def test_all_nonfinite_scores(
        self,
    ):
        cem = (
            CategoricalCEMOptimizer(
                CategoricalCEMConfig(
                    population_size=8
                )
            )
        )

        def bad_objective(
            plans,
            iteration,
        ):
            return torch.full(
                (8,),
                float("nan"),
            )

        with self.assertRaises(
            ValueError
        ):
            cem.optimize(
                bad_objective
            )


if __name__ == "__main__":
    unittest.main()