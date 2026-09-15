from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite
from typing import Callable, Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CategoricalCEMConfig:
    """
    离散动作空间 CEM 的配置。

    当前模块只关心：
        H: planning horizon
        A: action number
        N: population size

    它不关心具体动作语义、reward、world model 或 CC4。
    """

    horizon: int = 4
    n_actions: int = 5

    population_size: int = 64
    num_iterations: int = 4

    elite_ratio: float = 0.30

    # 与 UG 官方 CEM 保持同样的 alpha 语义：
    #
    # new = alpha * old
    #       + (1 - alpha) * elite
    #
    alpha: float = 0.10

    # 防止某动作概率永久变成 0
    prob_floor: float = 0.01

    seed: int = 0
    device: str = "cpu"


@dataclass
class CEMResult:
    """
    一次 CEM 搜索的输出。
    """

    # 所有 CEM iteration 中真正评估过的最佳计划
    # shape = [H]
    best_plan: torch.Tensor

    # best_plan 对应的 objective score
    best_score: float

    # 最后一轮更新后的 categorical probability
    # shape = [H, A]
    final_probs: torch.Tensor


class CategoricalCEMOptimizer:
    """
    离散动作空间下的 Cross-Entropy Method。

    probability matrix:

        shape = [H, A]

    例如 H=4, A=5：

        t0  p(a0) p(a1) p(a2) p(a3) p(a4)
        t1  ...
        t2  ...
        t3  ...

    每轮流程：

        probability
            ↓
        sample N plans
            ↓
        objective_fn
            ↓
        choose elites
            ↓
        elite action frequencies
            ↓
        update probability
    """

    def __init__(
        self,
        cfg: CategoricalCEMConfig,
    ):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self._validate_config()

        # 均匀初始概率：
        #
        # 每个位置的 5 个动作均为 0.2
        #
        self._uniform_probs = torch.full(
            (
                cfg.horizon,
                cfg.n_actions,
            ),
            1.0 / cfg.n_actions,
            dtype=torch.float32,
            device=self.device,
        )

        # 独立随机数生成器
        #
        # 不修改全局 torch random state
        #
        self.generator = torch.Generator(
            device=self.device
        )

        self.generator.manual_seed(
            cfg.seed
        )

    @property
    def uniform_probs(self) -> torch.Tensor:
        """
        返回均匀概率矩阵的副本。

        shape:
            [H, A]
        """

        return self._uniform_probs.clone()

    @property
    def elite_num(self) -> int:
        """
        elite 数量。

        与 UG 官方源码一致：

            ceil(
                population_size
                * elite_ratio
            )
        """

        return max(
            1,
            int(
                ceil(
                    self.cfg.population_size
                    * self.cfg.elite_ratio
                )
            ),
        )

    def _validate_config(self) -> None:
        """
        检查配置是否合法。
        """

        c = self.cfg

        if c.horizon <= 0:
            raise ValueError(
                "horizon must be > 0"
            )

        if c.n_actions <= 1:
            raise ValueError(
                "n_actions must be > 1"
            )

        if c.population_size <= 0:
            raise ValueError(
                "population_size must be > 0"
            )

        if c.num_iterations <= 0:
            raise ValueError(
                "num_iterations must be > 0"
            )

        if (
            not isfinite(
                float(c.elite_ratio)
            )
            or not (
                0.0
                < c.elite_ratio
                <= 1.0
            )
        ):
            raise ValueError(
                "elite_ratio must be in (0, 1]"
            )

        if (
            not isfinite(
                float(c.alpha)
            )
            or not (
                0.0
                <= c.alpha
                <= 1.0
            )
        ):
            raise ValueError(
                "alpha must be in [0, 1]"
            )

        if (
            not isfinite(
                float(c.prob_floor)
            )
            or c.prob_floor < 0.0
        ):
            raise ValueError(
                "prob_floor must be finite "
                "and >= 0"
            )

        if (
            c.n_actions
            * c.prob_floor
            >= 1.0
        ):
            raise ValueError(
                "n_actions * prob_floor "
                "must be < 1"
                "finite_mask"
            )

    def _apply_probability_floor(
        self,
        probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        给 categorical probability 设置最小概率。

        使用幂等的 lower-bound projection。

        第一步：

            p = normalize(p)

        第二步：

            residual
                = max(p - floor, 0)

        第三步：

            q
            =
            floor
            +
            (1 - A*floor)
            * residual
            / sum(residual)

        好处：

        1. 每个动作概率 >= floor
        2. 每行和 = 1
        3. 如果输入本来已经满足 floor，
           再执行一次不会继续改变概率
        """

        probs = torch.as_tensor(
            probs,
            dtype=torch.float32,
            device=self.device,
        )

        expected_shape = (
            self.cfg.horizon,
            self.cfg.n_actions,
        )

        if (
            tuple(probs.shape)
            != expected_shape
        ):
            raise ValueError(
                "probability matrix must "
                f"have shape {expected_shape}, "
                f"got {tuple(probs.shape)}"
            )

        if not torch.isfinite(
            probs
        ).all():
            raise ValueError(
                "probabilities must be finite"
            )

        if (
            probs < 0
        ).any():
            raise ValueError(
                "probabilities cannot "
                "be negative"
            )

        row_sums = probs.sum(
            dim=1,
            keepdim=True,
        )

        if (
            row_sums <= 0
        ).any():
            raise ValueError(
                "each probability row "
                "must have positive mass"
            )

        # 先归一化
        p = probs / row_sums

        floor = self.cfg.prob_floor

        if floor == 0.0:
            return p

        # 去掉 floor 后剩余的有效概率
        residual = torch.clamp(
            p - floor,
            min=0.0,
        )

        residual_sum = residual.sum(
            dim=1,
            keepdim=True,
        )

        # 在 A*floor < 1 且 p 行和为1的情况下，
        # 理论上 residual_sum 一定 > 0。
        #
        # 这里保留检查避免隐藏数值错误。
        if (
            residual_sum <= 0
        ).any():
            raise RuntimeError(
                "invalid probability-floor "
                "residual"
            )

        remaining_mass = (
            1.0
            - self.cfg.n_actions
            * floor
        )

        q = (
            floor
            + remaining_mass
            * (
                residual
                / residual_sum
            )
        )

        return q

    def _prepare_initial_probs(
        self,
        initial_probs: Optional[
            torch.Tensor
        ],
    ) -> torch.Tensor:
        """
        为一次 optimize() 准备初始概率。

        initial_probs=None：

            使用 uniform。

        initial_probs 给定：

            使用调用方传入的概率。

        注意：

        本类不会自动使用上一次 optimize
        的 final_probs。

        MPC warm-start 将由未来的
        planner.py 显式传入。
        """

        if initial_probs is None:
            return self.uniform_probs

        return self._apply_probability_floor(
            initial_probs
        )

    def _sample_population(
        self,
        probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        从 categorical probability 中
        采样 N 条计划。

        输入：

            probs
            [H, A]

        torch.multinomial 输出：

            [H, N]

        转置之后：

            plans
            [N, H]

        plans dtype：

            torch.long
        """

        sampled = torch.multinomial(
            probs,
            num_samples=(
                self.cfg.population_size
            ),
            replacement=True,
            generator=self.generator,
        )

        plans = (
            sampled
            .transpose(0, 1)
            .contiguous()
            .long()
        )

        return plans

    def _elite_frequencies(
        self,
        elite_plans: torch.Tensor,
    ) -> torch.Tensor:
        """
        统计 elite plans 在每个时间位置
        上选择各动作的频率。

        输入：

            elite_plans
            [K, H]

        输出：

            elite_freq
            [H, A]
        """

        if elite_plans.ndim != 2:
            raise ValueError(
                "elite_plans must have "
                "shape [K,H]"
            )

        if (
            elite_plans.shape[1]
            != self.cfg.horizon
        ):
            raise ValueError(
                "elite_plans horizon "
                "does not match config"
            )

        if (
            elite_plans.shape[0]
            <= 0
        ):
            raise ValueError(
                "elite_plans cannot be empty"
            )

        elite_plans = elite_plans.to(
            device=self.device,
            dtype=torch.long,
        )

        if (
            elite_plans < 0
        ).any() or (
            elite_plans
            >= self.cfg.n_actions
        ).any():
            raise ValueError(
                "elite_plans contain "
                "invalid action ids"
            )

        # [K,H]
        #
        # one_hot
        # ↓
        #
        # [K,H,A]
        #
        # 对 K 求平均
        # ↓
        #
        # [H,A]
        #
        elite_freq = F.one_hot(
            elite_plans,
            num_classes=(
                self.cfg.n_actions
            ),
        ).float().mean(
            dim=0
        )

        return elite_freq

    def _update_probs(
        self,
        old_probs: torch.Tensor,
        elite_freq: torch.Tensor,
    ) -> torch.Tensor:
        """
        根据 elite distribution 更新概率。

        与 UG 官方源码 alpha 方向一致：

            p_new
            =
            alpha * p_old
            +
            (1-alpha) * p_elite
        """

        expected_shape = (
            self.cfg.horizon,
            self.cfg.n_actions,
        )

        if (
            tuple(old_probs.shape)
            != expected_shape
            or
            tuple(elite_freq.shape)
            != expected_shape
        ):
            raise ValueError(
                "old_probs and elite_freq "
                "must have shape [H,A]"
            )

        new_probs = (
            self.cfg.alpha
            * old_probs
            +
            (
                1.0
                - self.cfg.alpha
            )
            * elite_freq
        )

        return (
            self._apply_probability_floor(
                new_probs
            )
        )

    def optimize(
        self,
        objective_fn: Callable[
            [
                torch.Tensor,
                int,
            ],
            torch.Tensor,
        ],
        initial_probs: Optional[
            torch.Tensor
        ] = None,
    ) -> CEMResult:
        """
        执行一次完整 CEM 搜索。

        objective_fn 接口：

            objective_fn(
                plans,
                iteration,
            )

        输入：

            plans:
                [N,H]

            iteration:
                当前 CEM iteration，
                从 0 开始。

        输出：

            scores:
                [N]

        score 越大越好。

        iteration 参数必须保留，
        因为以后 UG uncertainty 使用：

            score
            =
            predicted_return
            -
            beta * uncertainty
            / (iteration + 1)
        """

        probs = (
            self._prepare_initial_probs(
                initial_probs
            )
        )

        best_plan = None
        best_score = float(
            "-inf"
        )

        for iteration in range(
            self.cfg.num_iterations
        ):
            # --------------------------------
            # 1. 采样候选计划
            # --------------------------------

            plans = (
                self._sample_population(
                    probs
                )
            )

            # --------------------------------
            # 2. 给每条计划评分
            # --------------------------------

            scores = torch.as_tensor(
                objective_fn(
                    plans,
                    iteration,
                ),
                dtype=torch.float32,
                device=self.device,
            )

            expected_shape = (
                self.cfg.population_size,
            )

            if (
                tuple(scores.shape)
                != expected_shape
            ):
                raise ValueError(
                    "objective_fn must "
                    "return shape "
                    f"{expected_shape}, "
                    f"got "
                    f"{tuple(scores.shape)}"
                )

            # --------------------------------
            # 3. 处理 NaN / Inf
            # --------------------------------

            finite_mask = torch.isfinite(
                scores
            )

            finite_count = int(
                finite_mask.sum().item()
            )

            if finite_count == 0:
                raise ValueError(
                    "all candidate scores "
                    "are non-finite"
                )

            if finite_count < self.elite_num:
                raise ValueError(
                    "not enough finite candidate "
                    "scores to form elite set: "
                    f"{finite_count} finite scores, "
                    f"but elite_num="
                    f"{self.elite_num}"
                )

            scores = torch.where(
                finite_mask,
                scores,
                torch.full_like(
                    scores,
                    float("-inf"),
                ),
            )

            # --------------------------------
            # 4. 保存全局最佳 sampled plan
            # --------------------------------

            (
                iter_best_score_tensor,
                iter_best_idx_tensor,
            ) = torch.max(
                scores,
                dim=0,
            )

            iter_best_score = float(
                iter_best_score_tensor.item()
            )

            if (
                iter_best_score
                > best_score
            ):
                best_score = (
                    iter_best_score
                )

                best_plan = (
                    plans[
                        int(
                            iter_best_idx_tensor
                            .item()
                        )
                    ]
                    .detach()
                    .clone()
                )

            # --------------------------------
            # 5. 选择 elite
            # --------------------------------

            elite_idx = torch.topk(
                scores,
                k=self.elite_num,
                largest=True,
                sorted=False,
            ).indices

            elite_plans = (
                plans[elite_idx]
            )

            # --------------------------------
            # 6. elite categorical frequency
            # --------------------------------

            elite_freq = (
                self._elite_frequencies(
                    elite_plans
                )
            )

            # --------------------------------
            # 7. 更新 categorical probability
            # --------------------------------

            probs = (
                self._update_probs(
                    probs,
                    elite_freq,
                )
            )

        if best_plan is None:
            raise RuntimeError(
                "CEM failed to produce "
                "a best plan"
            )

        return CEMResult(
            best_plan=best_plan,
            best_score=best_score,
            final_probs=(
                probs
                .detach()
                .clone()
            ),
        )