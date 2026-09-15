from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch


@dataclass(frozen=True)
class UGUncertaintyConfig:
    """
    UG trajectory uncertainty 的配置。

    alpha:
        running statistics 的指数移动平均系数。

    eps:
        防止标准差过小导致除 0。

    device:
        uncertainty 计算设备。
    """

    alpha: float = 0.01
    eps: float = 1e-8
    device: str = "cpu"


class UGUncertainty:
    """
    Uncertainty-Guided Planning 中的
    trajectory disagreement estimator。

    输入：

        next_states
        shape = [H, N, M, D]

    其中：

        H = planning horizon
        N = candidate plans
        M = ensemble members
        D = state dimension

    输出：

        uncertainty
        shape = [N]

    每个 candidate plan 最终对应一个
    trajectory uncertainty。
    """

    def __init__(
        self,
        cfg: UGUncertaintyConfig,
    ):
        self.cfg = cfg
        self.device = torch.device(
            cfg.device
        )

        self._validate_config()

        # 与官方源码一样，
        # 第一次看到输入 shape 后再初始化。
        self.obs_mean = None
        self.obs_std = None
        self.horizon_std = None

        self._horizon = None
        self._state_dim = None

    def _validate_config(
        self,
    ) -> None:
        """
        检查配置。
        """

        if (
            not isfinite(
                float(self.cfg.alpha)
            )
            or not (
                0.0
                <= self.cfg.alpha
                <= 1.0
            )
        ):
            raise ValueError(
                "alpha must be finite "
                "and in [0, 1]"
            )

        if (
            not isfinite(
                float(self.cfg.eps)
            )
            or self.cfg.eps <= 0.0
        ):
            raise ValueError(
                "eps must be finite "
                "and > 0"
            )

    def _ensure_initialized(
        self,
        horizon: int,
        state_dim: int,
    ) -> None:
        """
        根据第一次输入的 H、D
        初始化 running statistics。

        对应 UG 官方初始化：

            obs_mean = 0
            obs_std = 0.1
            horizon_std = 0.01
        """

        if self.obs_mean is None:
            self._horizon = horizon
            self._state_dim = state_dim

            self.obs_mean = torch.zeros(
                state_dim,
                dtype=torch.float32,
                device=self.device,
            )

            self.obs_std = torch.full(
                (state_dim,),
                0.1,
                dtype=torch.float32,
                device=self.device,
            )

            self.horizon_std = torch.full(
                (horizon,),
                0.01,
                dtype=torch.float32,
                device=self.device,
            )

            return

        # normalizer 初始化之后，
        # H 和 D 不允许悄悄发生变化。
        if horizon != self._horizon:
            raise ValueError(
                "planning horizon changed: "
                f"expected {self._horizon}, "
                f"got {horizon}"
            )

        if state_dim != self._state_dim:
            raise ValueError(
                "state dimension changed: "
                f"expected {self._state_dim}, "
                f"got {state_dim}"
            )

    def reset(
        self,
    ) -> None:
        """
        清空 running statistics。

        下一次 compute() 时重新使用：

            obs_mean = 0
            obs_std = 0.1
            horizon_std = 0.01
        """

        self.obs_mean = None
        self.obs_std = None
        self.horizon_std = None

        self._horizon = None
        self._state_dim = None

    def compute(
        self,
        next_states: torch.Tensor,
        update_stats: bool = True,
    ) -> torch.Tensor:
        """
        计算每条 candidate plan 的
        UG trajectory uncertainty。

        Parameters
        ----------
        next_states:
            shape = [H, N, M, D]

        update_stats:
            True:
                按官方方式更新
                obs_mean / obs_std /
                horizon_std。

            False:
                不进行 EMA statistics 更新。

                如果当前 normalizer 尚未初始化，
                仍会创建官方默认值：

                    obs_mean = 0
                    obs_std = 0.1
                    horizon_std = 0.01

        Returns
        -------
        uncertainty:
            shape = [N]
        """

        next_states = torch.as_tensor(
            next_states,
            dtype=torch.float32,
            device=self.device,
        )

        if next_states.ndim != 4:
            raise ValueError(
                "next_states must have "
                "shape [H,N,M,D]"
            )

        (
            horizon,
            population_size,
            ensemble_size,
            state_dim,
        ) = next_states.shape

        if horizon <= 0:
            raise ValueError(
                "H must be > 0"
            )

        if population_size <= 0:
            raise ValueError(
                "N must be > 0"
            )

        # std(dim=M) 至少需要两个 member。
        #
        # 正式系统 M=5。
        if ensemble_size <= 1:
            raise ValueError(
                "M must be > 1 "
                "to measure ensemble "
                "disagreement"
            )

        if state_dim <= 0:
            raise ValueError(
                "D must be > 0"
            )

        if not torch.isfinite(
            next_states
        ).all():
            raise ValueError(
                "next_states must be finite"
            )

        self._ensure_initialized(
            horizon=horizon,
            state_dim=state_dim,
        )

        # uncertainty 本身只用于规划，
        # 不参与 world-model 反向传播。
        next_states = (
            next_states
            .detach()
        )

        alpha = self.cfg.alpha

        # --------------------------------
        # 1. 更新全局 state statistics
        # --------------------------------
        #
        # 输入已经是：
        #
        # [H,N,M,D]
        #
        # 因此对 H,N,M 求统计，
        # 最终留下 D：
        #
        # [D]
        #
        if update_stats:
            batch_mean = (
                next_states.mean(
                    dim=(0, 1, 2)
                )
            )

            # 官方源码直接使用 torch.std，
            # 因此这里保持相同语义。
            batch_std = (
                next_states.std(
                    dim=(0, 1, 2)
                )
            )

            self.obs_mean = (
                (1.0 - alpha)
                * self.obs_mean
                +
                alpha
                * batch_mean
            )

            self.obs_std = (
                (1.0 - alpha)
                * self.obs_std
                +
                alpha
                * batch_std
            )

        # --------------------------------
        # 2. state normalization
        # --------------------------------

        obs_denom = (
            self.obs_std.clamp_min(
                self.cfg.eps
            )
        )

        normalized_states = (
            next_states
            - self.obs_mean.view(
                1,
                1,
                1,
                state_dim,
            )
        ) / obs_denom.view(
            1,
            1,
            1,
            state_dim,
        )

        # --------------------------------
        # 3. ensemble disagreement
        # --------------------------------
        #
        # [H,N,M,D]
        #
        # std over M
        #
        # ↓
        #
        # [H,N,D]
        #
        member_std = (
            normalized_states.std(
                dim=2
            )
        )

        # --------------------------------
        # 4. 对状态维 D 求平均
        # --------------------------------
        #
        # [H,N,D]
        #
        # ↓
        #
        # [H,N]
        #
        next_obs_std = (
            member_std.mean(
                dim=2
            )
        )

        # --------------------------------
        # 5. 更新 horizon normalizer
        # --------------------------------
        #
        # 对 N 求平均：
        #
        # [H,N]
        #
        # ↓
        #
        # [H]
        #
        if update_stats:
            horizon_batch = (
                next_obs_std.mean(
                    dim=1
                )
            )

            self.horizon_std = (
                (1.0 - alpha)
                * self.horizon_std
                +
                alpha
                * horizon_batch
            )

        # --------------------------------
        # 6. 按 horizon 归一化
        # --------------------------------

        horizon_denom = (
            self.horizon_std.clamp_min(
                self.cfg.eps
            )
            .unsqueeze(1)
        )

        normalized_disagreement = (
            next_obs_std
            / horizon_denom
        )

        # --------------------------------
        # 7. 对 horizon 求平均
        # --------------------------------
        #
        # [H,N]
        #
        # ↓
        #
        # [N]
        #
        uncertainty = (
            normalized_disagreement.mean(
                dim=0
            )
        )

        if not torch.isfinite(
            uncertainty
        ).all():
            raise RuntimeError(
                "computed uncertainty "
                "contains NaN or Inf"
            )

        return uncertainty