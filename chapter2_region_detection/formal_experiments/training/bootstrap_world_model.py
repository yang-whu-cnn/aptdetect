from __future__ import annotations

from dataclasses import (
    asdict,
    dataclass,
)
from math import isfinite
from pathlib import Path
from typing import (
    Literal,
    Sequence,
)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from formal_experiments.data_collection.decision_replay import (
    DecisionEpochTransition,
)

from shared.action_contract import (
    ACTION_CONTRACTS,
    N_ACTIONS,
)

from shared.formal_state import (
    FORMAL_STATE_DIM,
)


# ============================================================
# Executed low-level family -> formal categorical action
# ============================================================

EXECUTED_FAMILY_TO_ACTION_ID = {
    action.cyborg_action:
        int(action.action_id)
    for action in ACTION_CONTRACTS
}


if len(
    EXECUTED_FAMILY_TO_ACTION_ID
) != N_ACTIONS:
    raise RuntimeError(
        "each formal action must map "
        "to one unique CybORG family"
    )


# ============================================================
# Dataset
# ============================================================

@dataclass(frozen=True)
class WorldModelDataset:
    states: np.ndarray
    actions: np.ndarray
    next_states: np.ndarray

    @property
    def n_samples(
        self,
    ) -> int:
        return int(
            self.states.shape[0]
        )


def _state_matrix(
    name: str,
    value,
) -> np.ndarray:
    array = np.asarray(
        value,
        dtype=np.float32,
    )

    if (
        array.ndim != 2
        or array.shape[1]
        != FORMAL_STATE_DIM
    ):
        raise ValueError(
            f"{name} must have shape "
            f"(N,{FORMAL_STATE_DIM}); "
            f"got {array.shape}"
        )

    if not np.all(
        np.isfinite(array)
    ):
        raise ValueError(
            f"{name} contains "
            "non-finite values"
        )

    return array.copy()


def _action_array(
    value,
) -> np.ndarray:
    array = np.asarray(
        value,
        dtype=np.int64,
    )

    if array.ndim != 1:
        raise ValueError(
            "actions must be 1-D"
        )

    if array.size == 0:
        raise ValueError(
            "actions must not be empty"
        )

    if (
        np.any(array < 0)
        or np.any(
            array >= N_ACTIONS
        )
    ):
        raise ValueError(
            "actions contain invalid "
            "categorical IDs"
        )

    return array.copy()


def validate_dataset(
    dataset: WorldModelDataset,
) -> WorldModelDataset:
    if not isinstance(
        dataset,
        WorldModelDataset,
    ):
        raise TypeError(
            "dataset must be "
            "WorldModelDataset"
        )

    states = _state_matrix(
        "states",
        dataset.states,
    )

    next_states = _state_matrix(
        "next_states",
        dataset.next_states,
    )

    actions = _action_array(
        dataset.actions
    )

    n = states.shape[0]

    if n == 0:
        raise ValueError(
            "world-model dataset "
            "must not be empty"
        )

    if (
        next_states.shape[0] != n
        or actions.shape[0] != n
    ):
        raise ValueError(
            "states/actions/next_states "
            "length mismatch"
        )

    return WorldModelDataset(
        states=states,
        actions=actions,
        next_states=next_states,
    )


def build_world_model_dataset(
    transitions: Sequence[
        DecisionEpochTransition
    ],
    *,
    drop_incomplete: bool = True,
) -> WorldModelDataset:
    """
    Decision replay -> dynamics dataset.

    关键 contract：

    1. dynamics 学 executed action，
       不是 requested action；

    2. fallback Restore -> Sleep
       会作为 action 0 / Sleep 学习；

    3. terminal mid-action
       action_completed=False
       默认从 dynamics training 删除；

    4. 完整 transition 必须满足：
       decision_dt == executed_duration_ticks。
    """

    states = []
    actions = []
    next_states = []

    for transition in transitions:
        if not isinstance(
            transition,
            DecisionEpochTransition,
        ):
            raise TypeError(
                "all transitions must be "
                "DecisionEpochTransition"
            )

        if not transition.action_completed:
            if drop_incomplete:
                continue

            raise ValueError(
                "incomplete action cannot "
                "be used as standard "
                "p(s_next|s,a) target"
            )

        if (
            transition.decision_dt
            !=
            transition.executed_duration_ticks
        ):
            raise ValueError(
                "completed transition has "
                "decision_dt inconsistent "
                "with executed duration"
            )

        family = str(
            transition
            .executed_action_family
        )

        if (
            family
            not in
            EXECUTED_FAMILY_TO_ACTION_ID
        ):
            raise ValueError(
                "unsupported executed "
                f"action family: {family}"
            )

        action_id = (
            EXECUTED_FAMILY_TO_ACTION_ID[
                family
            ]
        )

        states.append(
            np.asarray(
                transition.state,
                dtype=np.float32,
            )
        )

        actions.append(
            int(action_id)
        )

        next_states.append(
            np.asarray(
                transition.next_state,
                dtype=np.float32,
            )
        )

    if not states:
        raise ValueError(
            "no completed transitions "
            "available for world model"
        )

    return validate_dataset(
        WorldModelDataset(
            states=np.stack(
                states,
                axis=0,
            ),
            actions=np.asarray(
                actions,
                dtype=np.int64,
            ),
            next_states=np.stack(
                next_states,
                axis=0,
            ),
        )
    )


# ============================================================
# Train-only state normalizer
# ============================================================

@dataclass(frozen=True)
class StateNormalizer:
    mean: np.ndarray
    std: np.ndarray
    eps: float

    @classmethod
    def fit(
        cls,
        states: np.ndarray,
        next_states: np.ndarray,
        *,
        eps: float = 1e-6,
    ) -> "StateNormalizer":
        eps = float(eps)

        if (
            not isfinite(eps)
            or eps <= 0.0
        ):
            raise ValueError(
                "eps must be finite "
                "and > 0"
            )

        states = _state_matrix(
            "states",
            states,
        )

        next_states = _state_matrix(
            "next_states",
            next_states,
        )

        if (
            states.shape[0]
            != next_states.shape[0]
        ):
            raise ValueError(
                "states/next_states "
                "length mismatch"
            )

        # Both are TRAIN transitions only.
        #
        # Using both ends gives one common
        # normalization space for s and s_next.
        matrix = np.concatenate(
            [
                states,
                next_states,
            ],
            axis=0,
        ).astype(
            np.float64,
            copy=False,
        )

        mean = np.mean(
            matrix,
            axis=0,
        )

        std = np.std(
            matrix,
            axis=0,
        )

        std = np.maximum(
            std,
            eps,
        )

        return cls(
            mean=mean.astype(
                np.float32
            ),
            std=std.astype(
                np.float32
            ),
            eps=eps,
        )

    def normalize_np(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        x = _state_matrix(
            "state matrix",
            x,
        )

        return (
            (x - self.mean)
            / self.std
        ).astype(
            np.float32
        )

    def denormalize_np(
        self,
        z: np.ndarray,
    ) -> np.ndarray:
        z = _state_matrix(
            "normalized matrix",
            z,
        )

        return (
            z * self.std
            + self.mean
        ).astype(
            np.float32
        )

    def normalize_tensor(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        mean = torch.as_tensor(
            self.mean,
            dtype=x.dtype,
            device=x.device,
        )

        std = torch.as_tensor(
            self.std,
            dtype=x.dtype,
            device=x.device,
        )

        return (
            x - mean
        ) / std

    def denormalize_tensor(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        mean = torch.as_tensor(
            self.mean,
            dtype=z.dtype,
            device=z.device,
        )

        std = torch.as_tensor(
            self.std,
            dtype=z.dtype,
            device=z.device,
        )

        return (
            z * std
            + mean
        )

    def variance_to_raw_tensor(
        self,
        var_z: torch.Tensor,
    ) -> torch.Tensor:
        std = torch.as_tensor(
            self.std,
            dtype=var_z.dtype,
            device=var_z.device,
        )

        return (
            var_z
            * std.square()
        )


# ============================================================
# Config
# ============================================================

TargetMode = Literal[
    "absolute",
    "delta",
]


@dataclass(frozen=True)
class BootstrapWorldModelConfig:
    state_dim: int = FORMAL_STATE_DIM
    n_actions: int = N_ACTIONS

    ensemble_size: int = 5

    hidden_dim: int = 128

    lr: float = 3e-4
    batch_size: int = 256
    epochs: int = 50

    state_std_eps: float = 1e-6

    logvar_min: float = -10.0
    logvar_max: float = 3.0

    target_mode: TargetMode = (
        "absolute"
    )

    model_seed: int = 20260916
    bootstrap_seed: int = 20261916

    def __post_init__(
        self,
    ) -> None:
        if self.state_dim != FORMAL_STATE_DIM:
            raise ValueError(
                "formal state_dim must be "
                f"{FORMAL_STATE_DIM}"
            )

        if self.n_actions != N_ACTIONS:
            raise ValueError(
                "formal n_actions must be "
                f"{N_ACTIONS}"
            )

        for name in (
            "ensemble_size",
            "hidden_dim",
            "batch_size",
            "epochs",
        ):
            value = getattr(
                self,
                name,
            )

            if (
                isinstance(value, bool)
                or not isinstance(
                    value,
                    int,
                )
                or value <= 0
            ):
                raise ValueError(
                    f"{name} must be "
                    "a positive int"
                )

        if (
            not isfinite(
                float(self.lr)
            )
            or self.lr <= 0
        ):
            raise ValueError(
                "lr must be finite "
                "and > 0"
            )

        if (
            not isfinite(
                float(
                    self.state_std_eps
                )
            )
            or self.state_std_eps <= 0
        ):
            raise ValueError(
                "state_std_eps must be "
                "finite and > 0"
            )

        if (
            self.logvar_min
            >= self.logvar_max
        ):
            raise ValueError(
                "logvar_min must be "
                "< logvar_max"
            )

        if self.target_mode not in (
            "absolute",
            "delta",
        ):
            raise ValueError(
                "target_mode must be "
                "'absolute' or 'delta'"
            )


# ============================================================
# Probabilistic member
# ============================================================

class ProbabilisticDynamicsMember(
    nn.Module
):
    """
    Diagonal-Gaussian dynamics:

        normalized s
        +
        one-hot executed action
        ->
        target mean + log variance
    """

    def __init__(
        self,
        *,
        state_dim: int,
        n_actions: int,
        hidden_dim: int,
        logvar_min: float,
        logvar_max: float,
    ):
        super().__init__()

        self.state_dim = int(
            state_dim
        )

        self.n_actions = int(
            n_actions
        )

        self.logvar_min = float(
            logvar_min
        )

        self.logvar_max = float(
            logvar_max
        )

        input_dim = (
            self.state_dim
            + self.n_actions
        )

        self.backbone = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(),
        )

        self.mean_head = nn.Linear(
            hidden_dim,
            self.state_dim,
        )

        self.logvar_head = nn.Linear(
            hidden_dim,
            self.state_dim,
        )

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        if (
            states.ndim != 2
            or states.shape[1]
            != self.state_dim
        ):
            raise ValueError(
                "states tensor has "
                "invalid shape"
            )

        if (
            actions.ndim != 1
            or actions.shape[0]
            != states.shape[0]
        ):
            raise ValueError(
                "actions tensor has "
                "invalid shape"
            )

        actions = actions.to(
            dtype=torch.long
        )

        if actions.numel() > 0:
            if (
                int(actions.min()) < 0
                or int(actions.max())
                >= self.n_actions
            ):
                raise ValueError(
                    "invalid action ID"
                )

        action_one_hot = F.one_hot(
            actions,
            num_classes=self.n_actions,
        ).to(
            dtype=states.dtype
        )

        x = torch.cat(
            [
                states,
                action_one_hot,
            ],
            dim=-1,
        )

        h = self.backbone(
            x
        )

        mean = self.mean_head(
            h
        )

        logvar = torch.clamp(
            self.logvar_head(
                h
            ),
            min=self.logvar_min,
            max=self.logvar_max,
        )

        return (
            mean,
            logvar,
        )


def gaussian_nll(
    target: torch.Tensor,
    mean: torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor:
    if (
        target.shape != mean.shape
        or mean.shape != logvar.shape
    ):
        raise ValueError(
            "target/mean/logvar "
            "shape mismatch"
        )

    return (
        0.5
        * (
            logvar
            +
            (target - mean).square()
            * torch.exp(
                -logvar
            )
        )
    ).mean()


# ============================================================
# Training summary
# ============================================================

@dataclass(frozen=True)
class BootstrapTrainingSummary:
    n_samples: int

    member_final_losses: tuple[
        float,
        ...,
    ]

    bootstrap_unique_fractions: tuple[
        float,
        ...,
    ]


# ============================================================
# Ensemble
# ============================================================

class BootstrapProbabilisticWorldModel:
    FORMAT_VERSION = 1

    def __init__(
        self,
        config: (
            BootstrapWorldModelConfig
            | None
        ) = None,
        *,
        device: str = "cpu",
    ):
        self.config = (
            config
            if config is not None
            else BootstrapWorldModelConfig()
        )

        self.device = torch.device(
            device
        )

        self.normalizer: (
            StateNormalizer | None
        ) = None

        self.model_seeds = tuple(
            int(
                self.config.model_seed
                + member
            )
            for member in range(
                self.config.ensemble_size
            )
        )

        self.bootstrap_seeds = tuple(
            int(
                self.config.bootstrap_seed
                + member
            )
            for member in range(
                self.config.ensemble_size
            )
        )

        self.models: list[
            ProbabilisticDynamicsMember
        ] = []

        self.optimizers: list[
            torch.optim.Optimizer
        ] = []

        for seed in self.model_seeds:
            torch.manual_seed(
                seed
            )

            model = (
                ProbabilisticDynamicsMember(
                    state_dim=(
                        self.config.state_dim
                    ),
                    n_actions=(
                        self.config.n_actions
                    ),
                    hidden_dim=(
                        self.config.hidden_dim
                    ),
                    logvar_min=(
                        self.config.logvar_min
                    ),
                    logvar_max=(
                        self.config.logvar_max
                    ),
                )
                .to(
                    self.device
                )
            )

            self.models.append(
                model
            )

            self.optimizers.append(
                torch.optim.Adam(
                    model.parameters(),
                    lr=self.config.lr,
                )
            )

        self._bootstrap_indices: list[
            np.ndarray
        ] = []

        self.training_summary: (
            BootstrapTrainingSummary
            | None
        ) = None

    # --------------------------------------------------------
    # bootstrap audit
    # --------------------------------------------------------

    @property
    def bootstrap_indices(
        self,
    ) -> tuple[
        np.ndarray,
        ...,
    ]:
        return tuple(
            item.copy()
            for item
            in self._bootstrap_indices
        )

    # --------------------------------------------------------
    # Fit
    # --------------------------------------------------------

    def fit(
        self,
        dataset: WorldModelDataset,
    ) -> BootstrapTrainingSummary:
        dataset = validate_dataset(
            dataset
        )

        n = dataset.n_samples

        self.normalizer = (
            StateNormalizer.fit(
                dataset.states,
                dataset.next_states,
                eps=(
                    self.config
                    .state_std_eps
                ),
            )
        )

        z_state = (
            self.normalizer
            .normalize_np(
                dataset.states
            )
        )

        z_next = (
            self.normalizer
            .normalize_np(
                dataset.next_states
            )
        )

        if (
            self.config.target_mode
            == "absolute"
        ):
            targets = z_next
        else:
            targets = (
                z_next - z_state
            ).astype(
                np.float32
            )

        states_t = torch.as_tensor(
            z_state,
            dtype=torch.float32,
            device=self.device,
        )

        actions_t = torch.as_tensor(
            dataset.actions,
            dtype=torch.long,
            device=self.device,
        )

        targets_t = torch.as_tensor(
            targets,
            dtype=torch.float32,
            device=self.device,
        )

        self._bootstrap_indices = []

        final_losses: list[
            float
        ] = []

        unique_fractions: list[
            float
        ] = []

        for member_idx, (
            model,
            optimizer,
        ) in enumerate(
            zip(
                self.models,
                self.optimizers,
            )
        ):
            bootstrap_rng = (
                np.random.default_rng(
                    self.bootstrap_seeds[
                        member_idx
                    ]
                )
            )

            bootstrap_idx = (
                bootstrap_rng.integers(
                    low=0,
                    high=n,
                    size=n,
                    dtype=np.int64,
                )
            )

            self._bootstrap_indices.append(
                bootstrap_idx.copy()
            )

            unique_fraction = (
                np.unique(
                    bootstrap_idx
                ).size
                / float(n)
            )

            unique_fractions.append(
                float(
                    unique_fraction
                )
            )

            # Independent shuffling stream.
            shuffle_rng = (
                np.random.default_rng(
                    self.bootstrap_seeds[
                        member_idx
                    ]
                    + 1_000_003
                )
            )

            model.train()

            last_epoch_losses = []

            for _epoch in range(
                self.config.epochs
            ):
                order = (
                    shuffle_rng.permutation(
                        n
                    )
                )

                epoch_losses = []

                for start in range(
                    0,
                    n,
                    self.config.batch_size,
                ):
                    positions = order[
                        start:
                        start
                        + self.config.batch_size
                    ]

                    batch_indices = (
                        bootstrap_idx[
                            positions
                        ]
                    )

                    idx_t = torch.as_tensor(
                        batch_indices,
                        dtype=torch.long,
                        device=self.device,
                    )

                    state_batch = (
                        states_t[
                            idx_t
                        ]
                    )

                    action_batch = (
                        actions_t[
                            idx_t
                        ]
                    )

                    target_batch = (
                        targets_t[
                            idx_t
                        ]
                    )

                    mean, logvar = model(
                        state_batch,
                        action_batch,
                    )

                    loss = gaussian_nll(
                        target_batch,
                        mean,
                        logvar,
                    )

                    if not torch.isfinite(
                        loss
                    ):
                        raise RuntimeError(
                            "non-finite "
                            "world-model loss"
                        )

                    optimizer.zero_grad(
                        set_to_none=True
                    )

                    loss.backward()

                    optimizer.step()

                    epoch_losses.append(
                        float(
                            loss.detach()
                            .cpu()
                            .item()
                        )
                    )

                last_epoch_losses = (
                    epoch_losses
                )

            model.eval()

            if not last_epoch_losses:
                raise RuntimeError(
                    "no training batches "
                    "were produced"
                )

            final_losses.append(
                float(
                    np.mean(
                        last_epoch_losses
                    )
                )
            )

        summary = (
            BootstrapTrainingSummary(
                n_samples=n,

                member_final_losses=tuple(
                    final_losses
                ),

                bootstrap_unique_fractions=tuple(
                    unique_fractions
                ),
            )
        )

        self.training_summary = (
            summary
        )

        return summary

    # --------------------------------------------------------
    # Prediction helpers
    # --------------------------------------------------------

    def _require_fitted(
        self,
    ) -> StateNormalizer:
        if self.normalizer is None:
            raise RuntimeError(
                "fit() or load_checkpoint() "
                "must be called first"
            )

        return self.normalizer

    def _prepare_tensor_inputs(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        if not isinstance(
            states,
            torch.Tensor,
        ):
            raise TypeError(
                "states must be "
                "torch.Tensor"
            )

        if not isinstance(
            actions,
            torch.Tensor,
        ):
            raise TypeError(
                "actions must be "
                "torch.Tensor"
            )

        states = states.to(
            device=self.device,
            dtype=torch.float32,
        )

        actions = actions.to(
            device=self.device,
            dtype=torch.long,
        )

        if (
            states.ndim != 2
            or states.shape[1]
            != self.config.state_dim
        ):
            raise ValueError(
                "states have invalid shape"
            )

        if (
            actions.ndim != 1
            or actions.shape[0]
            != states.shape[0]
        ):
            raise ValueError(
                "actions have invalid shape"
            )

        if not torch.isfinite(
            states
        ).all():
            raise ValueError(
                "states contain "
                "non-finite values"
            )

        if actions.numel() > 0:
            if (
                int(actions.min()) < 0
                or int(actions.max())
                >= self.config.n_actions
            ):
                raise ValueError(
                    "actions contain "
                    "invalid IDs"
                )

        return (
            states,
            actions,
        )

    @torch.no_grad()
    def predict_member_distribution_tensor(
        self,
        member_idx: int,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Vectorized prediction for one fixed member.

        Returns raw-space:
            next_mean [B,D]
            next_variance [B,D]

        Step 4 fixed-member rollout
        会直接复用这个接口。
        """

        normalizer = (
            self._require_fitted()
        )

        if (
            isinstance(
                member_idx,
                bool,
            )
            or not isinstance(
                member_idx,
                int,
            )
            or not (
                0
                <= member_idx
                < len(self.models)
            )
        ):
            raise ValueError(
                "invalid member_idx"
            )

        (
            states,
            actions,
        ) = self._prepare_tensor_inputs(
            states,
            actions,
        )

        z_state = (
            normalizer
            .normalize_tensor(
                states
            )
        )

        target_mean, logvar = (
            self.models[
                member_idx
            ](
                z_state,
                actions,
            )
        )

        if (
            self.config.target_mode
            == "absolute"
        ):
            next_z_mean = (
                target_mean
            )
        else:
            next_z_mean = (
                z_state
                + target_mean
            )

        next_mean = (
            normalizer
            .denormalize_tensor(
                next_z_mean
            )
        )

        target_var_z = torch.exp(
            logvar
        )

        next_var = (
            normalizer
            .variance_to_raw_tensor(
                target_var_z
            )
        )

        return (
            next_mean,
            next_var,
        )

    @torch.no_grad()
    def predict_member_mean_tensor(
        self,
        member_idx: int,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        mean, _variance = (
            self
            .predict_member_distribution_tensor(
                member_idx,
                states,
                actions,
            )
        )

        return mean

    @torch.no_grad()
    def predict_ensemble(
        self,
        states: np.ndarray,
        actions: np.ndarray,
    ) -> dict[
        str,
        np.ndarray,
    ]:
        """
        Returns:

        member_means:
            [M,B,D]

        member_variances:
            [M,B,D]

        mean:
            [B,D]

        epistemic_variance:
            variance of member means

        aleatoric_variance:
            mean member predicted variance

        total_variance:
            epistemic + aleatoric
        """

        states = _state_matrix(
            "states",
            states,
        )

        actions = _action_array(
            actions
        )

        if (
            states.shape[0]
            != actions.shape[0]
        ):
            raise ValueError(
                "states/actions "
                "length mismatch"
            )

        states_t = torch.as_tensor(
            states,
            dtype=torch.float32,
            device=self.device,
        )

        actions_t = torch.as_tensor(
            actions,
            dtype=torch.long,
            device=self.device,
        )

        member_means = []
        member_variances = []

        for member_idx in range(
            len(self.models)
        ):
            mean, variance = (
                self
                .predict_member_distribution_tensor(
                    member_idx,
                    states_t,
                    actions_t,
                )
            )

            member_means.append(
                mean
            )

            member_variances.append(
                variance
            )

        means_t = torch.stack(
            member_means,
            dim=0,
        )

        variances_t = torch.stack(
            member_variances,
            dim=0,
        )

        mean_t = torch.mean(
            means_t,
            dim=0,
        )

        epistemic_t = torch.var(
            means_t,
            dim=0,
            unbiased=False,
        )

        aleatoric_t = torch.mean(
            variances_t,
            dim=0,
        )

        total_t = (
            epistemic_t
            + aleatoric_t
        )

        return {
            "member_means":
                means_t.cpu()
                .numpy()
                .astype(
                    np.float32
                ),

            "member_variances":
                variances_t.cpu()
                .numpy()
                .astype(
                    np.float32
                ),

            "mean":
                mean_t.cpu()
                .numpy()
                .astype(
                    np.float32
                ),

            "epistemic_variance":
                epistemic_t.cpu()
                .numpy()
                .astype(
                    np.float32
                ),

            "aleatoric_variance":
                aleatoric_t.cpu()
                .numpy()
                .astype(
                    np.float32
                ),

            "total_variance":
                total_t.cpu()
                .numpy()
                .astype(
                    np.float32
                ),
        }

    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    def save_checkpoint(
        self,
        path: str | Path,
    ) -> None:
        normalizer = (
            self._require_fitted()
        )

        path = Path(
            path
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "format_version":
                self.FORMAT_VERSION,

            "config":
                asdict(
                    self.config
                ),

            "normalizer": {
                "mean":
                    normalizer.mean,

                "std":
                    normalizer.std,

                "eps":
                    normalizer.eps,
            },

            "model_seeds":
                self.model_seeds,

            "bootstrap_seeds":
                self.bootstrap_seeds,

            "model_state_dicts": [
                model.state_dict()
                for model in self.models
            ],
        }

        torch.save(
            payload,
            path,
        )

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str = "cpu",
    ) -> (
        "BootstrapProbabilisticWorldModel"
    ):
        payload = torch.load(
            Path(path),
            map_location=device,
        )

        if (
            payload.get(
                "format_version"
            )
            != cls.FORMAT_VERSION
        ):
            raise ValueError(
                "unsupported checkpoint "
                "format_version"
            )

        config = (
            BootstrapWorldModelConfig(
                **payload[
                    "config"
                ]
            )
        )

        instance = cls(
            config=config,
            device=device,
        )

        normalizer_payload = (
            payload[
                "normalizer"
            ]
        )

        instance.normalizer = (
            StateNormalizer(
                mean=np.asarray(
                    normalizer_payload[
                        "mean"
                    ],
                    dtype=np.float32,
                ),

                std=np.asarray(
                    normalizer_payload[
                        "std"
                    ],
                    dtype=np.float32,
                ),

                eps=float(
                    normalizer_payload[
                        "eps"
                    ]
                ),
            )
        )

        state_dicts = (
            payload[
                "model_state_dicts"
            ]
        )

        if (
            len(state_dicts)
            != len(instance.models)
        ):
            raise ValueError(
                "checkpoint ensemble size "
                "does not match config"
            )

        for model, state_dict in zip(
            instance.models,
            state_dicts,
        ):
            model.load_state_dict(
                state_dict
            )

            model.eval()

        return instance