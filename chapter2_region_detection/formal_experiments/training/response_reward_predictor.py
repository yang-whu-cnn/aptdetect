from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from formal_experiments.data_collection.decision_replay import (
    DecisionEpochTransition,
)

from formal_experiments.training.bootstrap_world_model import (
    EXECUTED_FAMILY_TO_ACTION_ID,
    StateNormalizer,
)

from shared.action_contract import N_ACTIONS
from shared.formal_state import FORMAL_STATE_DIM


# ============================================================
# Dataset
# ============================================================

@dataclass(frozen=True)
class ResponseRewardDataset:
    states: np.ndarray
    actions: np.ndarray
    next_states: np.ndarray
    rewards: np.ndarray

    @property
    def n_samples(self) -> int:
        return int(
            self.states.shape[0]
        )


def _state_matrix(
    name: str,
    value,
) -> np.ndarray:
    value = np.asarray(
        value,
        dtype=np.float32,
    )

    if (
        value.ndim != 2
        or value.shape[1]
        != FORMAL_STATE_DIM
    ):
        raise ValueError(
            f"{name} must have shape "
            f"(N,{FORMAL_STATE_DIM}); "
            f"got {value.shape}"
        )

    if not np.all(
        np.isfinite(value)
    ):
        raise ValueError(
            f"{name} contains "
            "non-finite values"
        )

    return value.copy()


def _action_array(
    value,
) -> np.ndarray:
    value = np.asarray(
        value,
        dtype=np.int64,
    )

    if value.ndim != 1:
        raise ValueError(
            "actions must be 1-D"
        )

    if value.size == 0:
        raise ValueError(
            "actions must not be empty"
        )

    if (
        np.any(value < 0)
        or np.any(
            value >= N_ACTIONS
        )
    ):
        raise ValueError(
            "invalid action IDs"
        )

    return value.copy()


def _reward_array(
    value,
) -> np.ndarray:
    value = np.asarray(
        value,
        dtype=np.float32,
    )

    if value.ndim != 1:
        raise ValueError(
            "rewards must be 1-D"
        )

    if value.size == 0:
        raise ValueError(
            "rewards must not be empty"
        )

    if not np.all(
        np.isfinite(value)
    ):
        raise ValueError(
            "rewards contain "
            "non-finite values"
        )

    return value.copy()


def validate_reward_dataset(
    dataset: ResponseRewardDataset,
) -> ResponseRewardDataset:
    if not isinstance(
        dataset,
        ResponseRewardDataset,
    ):
        raise TypeError(
            "dataset must be "
            "ResponseRewardDataset"
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

    rewards = _reward_array(
        dataset.rewards
    )

    n = states.shape[0]

    if (
        next_states.shape[0] != n
        or actions.shape[0] != n
        or rewards.shape[0] != n
    ):
        raise ValueError(
            "dataset length mismatch"
        )

    return ResponseRewardDataset(
        states=states,
        actions=actions,
        next_states=next_states,
        rewards=rewards,
    )


def build_response_reward_dataset(
    transitions: Sequence[
        DecisionEpochTransition
    ],
    *,
    drop_incomplete: bool = True,
) -> ResponseRewardDataset:
    """
    Planner-visible reward-model dataset.

    Inputs:
        state
        executed/canonical action
        next_state

    Label:
        frozen A4.3 response_reward

    Hidden incident identity / truth is NOT input.
    """

    states = []
    actions = []
    next_states = []
    rewards = []

    for item in transitions:

        if not isinstance(
            item,
            DecisionEpochTransition,
        ):
            raise TypeError(
                "all transitions must be "
                "DecisionEpochTransition"
            )

        if not item.action_completed:

            if drop_incomplete:
                continue

            raise ValueError(
                "incomplete transition "
                "cannot enter reward model"
            )

        if (
            item.decision_dt
            != item.executed_duration_ticks
        ):
            raise ValueError(
                "completed transition "
                "duration mismatch"
            )

        family = str(
            item.executed_action_family
        )

        if (
            family
            not in
            EXECUTED_FAMILY_TO_ACTION_ID
        ):
            raise ValueError(
                "unsupported executed "
                f"family: {family}"
            )

        reward = float(
            item.response_reward
        )

        if not isfinite(
            reward
        ):
            raise ValueError(
                "non-finite response reward"
            )

        states.append(
            np.asarray(
                item.state,
                dtype=np.float32,
            )
        )

        actions.append(
            int(
                EXECUTED_FAMILY_TO_ACTION_ID[
                    family
                ]
            )
        )

        next_states.append(
            np.asarray(
                item.next_state,
                dtype=np.float32,
            )
        )

        rewards.append(
            reward
        )

    if not states:
        raise ValueError(
            "no completed transitions "
            "for reward model"
        )

    return validate_reward_dataset(
        ResponseRewardDataset(
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

            rewards=np.asarray(
                rewards,
                dtype=np.float32,
            ),
        )
    )


# ============================================================
# Reward normalization
# ============================================================

@dataclass(frozen=True)
class ScalarNormalizer:
    mean: float
    std: float
    eps: float

    @classmethod
    def fit(
        cls,
        values,
        *,
        eps: float = 1e-6,
    ) -> "ScalarNormalizer":

        values = _reward_array(
            values
        )

        eps = float(
            eps
        )

        if (
            not isfinite(eps)
            or eps <= 0.0
        ):
            raise ValueError(
                "eps must be > 0"
            )

        mean = float(
            np.mean(
                values,
                dtype=np.float64,
            )
        )

        std = float(
            np.std(
                values,
                dtype=np.float64,
            )
        )

        std = max(
            std,
            eps,
        )

        return cls(
            mean=mean,
            std=std,
            eps=eps,
        )

    def normalize_np(
        self,
        values,
    ) -> np.ndarray:
        values = np.asarray(
            values,
            dtype=np.float32,
        )

        return (
            (
                values
                - self.mean
            )
            / self.std
        ).astype(
            np.float32
        )

    def denormalize_np(
        self,
        values,
    ) -> np.ndarray:
        values = np.asarray(
            values,
            dtype=np.float32,
        )

        return (
            values
            * self.std
            + self.mean
        ).astype(
            np.float32
        )

    def denormalize_tensor(
        self,
        values: torch.Tensor,
    ) -> torch.Tensor:
        return (
            values
            * float(
                self.std
            )
            + float(
                self.mean
            )
        )


# ============================================================
# Config
# ============================================================

@dataclass(frozen=True)
class ResponseRewardPredictorConfig:
    state_dim: int = FORMAL_STATE_DIM
    n_actions: int = N_ACTIONS

    hidden_dim: int = 128

    lr: float = 3e-4
    batch_size: int = 256
    epochs: int = 50

    state_std_eps: float = 1e-6
    reward_std_eps: float = 1e-6

    model_seed: int = 20260917

    def __post_init__(
        self,
    ) -> None:

        if (
            self.state_dim
            != FORMAL_STATE_DIM
        ):
            raise ValueError(
                "state_dim contract changed"
            )

        if (
            self.n_actions
            != N_ACTIONS
        ):
            raise ValueError(
                "n_actions contract changed"
            )

        for name in (
            "hidden_dim",
            "batch_size",
            "epochs",
        ):
            value = getattr(
                self,
                name,
            )

            if (
                isinstance(
                    value,
                    bool,
                )
                or not isinstance(
                    value,
                    int,
                )
                or value <= 0
            ):
                raise ValueError(
                    f"{name} must be "
                    "positive int"
                )

        if (
            not isfinite(
                float(
                    self.lr
                )
            )
            or self.lr <= 0.0
        ):
            raise ValueError(
                "lr must be > 0"
            )


# ============================================================
# Network
# ============================================================

class ResponseRewardNetwork(
    nn.Module
):
    """
    r_hat(s, a, s_next)

    Only planner/model-space variables.
    """

    def __init__(
        self,
        *,
        state_dim: int,
        n_actions: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.state_dim = int(
            state_dim
        )

        self.n_actions = int(
            n_actions
        )

        input_dim = (
            2
            * self.state_dim
            + self.n_actions
        )

        self.network = nn.Sequential(
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

            nn.Linear(
                hidden_dim,
                1,
            ),
        )

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
    ) -> torch.Tensor:

        if (
            states.ndim != 2
            or next_states.ndim != 2
            or states.shape
            != next_states.shape
            or states.shape[1]
            != self.state_dim
        ):
            raise ValueError(
                "invalid state shape"
            )

        if (
            actions.ndim != 1
            or actions.shape[0]
            != states.shape[0]
        ):
            raise ValueError(
                "invalid action shape"
            )

        actions = actions.to(
            dtype=torch.long
        )

        if actions.numel() > 0:
            if (
                int(
                    actions.min()
                ) < 0
                or int(
                    actions.max()
                )
                >= self.n_actions
            ):
                raise ValueError(
                    "invalid action ID"
                )

        one_hot = F.one_hot(
            actions,
            num_classes=self.n_actions,
        ).to(
            dtype=states.dtype
        )

        x = torch.cat(
            [
                states,
                one_hot,
                next_states,
            ],
            dim=1,
        )

        return (
            self.network(
                x
            )
            .squeeze(
                -1
            )
        )


# ============================================================
# Training
# ============================================================

@dataclass(frozen=True)
class ResponseRewardTrainingSummary:
    n_samples: int
    final_loss: float
    reward_mean: float
    reward_std: float


class ResponseRewardPredictor:
    FORMAT_VERSION = 1

    def __init__(
        self,
        config: (
            ResponseRewardPredictorConfig
            | None
        ) = None,
        *,
        device: str = "cpu",
    ):
        self.config = (
            config
            if config is not None
            else ResponseRewardPredictorConfig()
        )

        self.device = torch.device(
            device
        )

        torch.manual_seed(
            int(
                self.config.model_seed
            )
        )

        self.model = (
            ResponseRewardNetwork(
                state_dim=(
                    self.config.state_dim
                ),
                n_actions=(
                    self.config.n_actions
                ),
                hidden_dim=(
                    self.config.hidden_dim
                ),
            )
            .to(
                self.device
            )
        )

        self.optimizer = (
            torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.lr,
            )
        )

        self.state_normalizer: (
            StateNormalizer | None
        ) = None

        self.reward_normalizer: (
            ScalarNormalizer | None
        ) = None

        self.training_summary: (
            ResponseRewardTrainingSummary
            | None
        ) = None

    def fit(
        self,
        dataset: ResponseRewardDataset,
    ) -> ResponseRewardTrainingSummary:

        dataset = (
            validate_reward_dataset(
                dataset
            )
        )

        self.state_normalizer = (
            StateNormalizer.fit(
                dataset.states,
                dataset.next_states,
                eps=(
                    self.config
                    .state_std_eps
                ),
            )
        )

        self.reward_normalizer = (
            ScalarNormalizer.fit(
                dataset.rewards,
                eps=(
                    self.config
                    .reward_std_eps
                ),
            )
        )

        z_state = (
            self.state_normalizer
            .normalize_np(
                dataset.states
            )
        )

        z_next = (
            self.state_normalizer
            .normalize_np(
                dataset.next_states
            )
        )

        z_reward = (
            self.reward_normalizer
            .normalize_np(
                dataset.rewards
            )
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

        next_t = torch.as_tensor(
            z_next,
            dtype=torch.float32,
            device=self.device,
        )

        rewards_t = torch.as_tensor(
            z_reward,
            dtype=torch.float32,
            device=self.device,
        )

        rng = np.random.default_rng(
            int(
                self.config.model_seed
            )
            + 1_000_003
        )

        n = dataset.n_samples
        final_losses = []

        self.model.train()

        for _epoch in range(
            self.config.epochs
        ):
            order = rng.permutation(
                n
            )

            epoch_losses = []

            for start in range(
                0,
                n,
                self.config.batch_size,
            ):
                index = order[
                    start:
                    start
                    + self.config.batch_size
                ]

                index_t = (
                    torch.as_tensor(
                        index,
                        dtype=torch.long,
                        device=self.device,
                    )
                )

                predicted = (
                    self.model(
                        states_t[
                            index_t
                        ],
                        actions_t[
                            index_t
                        ],
                        next_t[
                            index_t
                        ],
                    )
                )

                target = rewards_t[
                    index_t
                ]

                loss = F.mse_loss(
                    predicted,
                    target,
                )

                if not torch.isfinite(
                    loss
                ):
                    raise RuntimeError(
                        "non-finite "
                        "reward-model loss"
                    )

                self.optimizer.zero_grad(
                    set_to_none=True
                )

                loss.backward()

                self.optimizer.step()

                epoch_losses.append(
                    float(
                        loss.detach()
                        .cpu()
                        .item()
                    )
                )

            final_losses = (
                epoch_losses
            )

        if not final_losses:
            raise RuntimeError(
                "no reward-model batches"
            )

        self.model.eval()

        summary = (
            ResponseRewardTrainingSummary(
                n_samples=n,

                final_loss=float(
                    np.mean(
                        final_losses
                    )
                ),

                reward_mean=float(
                    self.reward_normalizer
                    .mean
                ),

                reward_std=float(
                    self.reward_normalizer
                    .std
                ),
            )
        )

        self.training_summary = summary

        return summary

    def _require_fitted(
        self,
    ) -> tuple[
        StateNormalizer,
        ScalarNormalizer,
    ]:

        if (
            self.state_normalizer
            is None
            or self.reward_normalizer
            is None
        ):
            raise RuntimeError(
                "fit() or load_checkpoint() "
                "must be called first"
            )

        return (
            self.state_normalizer,
            self.reward_normalizer,
        )

    @torch.no_grad()
    def predict_tensor(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
    ) -> torch.Tensor:

        (
            state_norm,
            reward_norm,
        ) = self._require_fitted()

        states = states.to(
            device=self.device,
            dtype=torch.float32,
        )

        next_states = next_states.to(
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
            != FORMAL_STATE_DIM
            or next_states.shape
            != states.shape
        ):
            raise ValueError(
                "invalid prediction "
                "state shape"
            )

        if (
            actions.ndim != 1
            or actions.shape[0]
            != states.shape[0]
        ):
            raise ValueError(
                "invalid prediction "
                "action shape"
            )

        z_state = (
            state_norm
            .normalize_tensor(
                states
            )
        )

        z_next = (
            state_norm
            .normalize_tensor(
                next_states
            )
        )

        z_reward = (
            self.model(
                z_state,
                actions,
                z_next,
            )
        )

        return (
            reward_norm
            .denormalize_tensor(
                z_reward
            )
        )

    @torch.no_grad()
    def predict(
        self,
        states,
        actions,
        next_states,
    ) -> np.ndarray:

        states = _state_matrix(
            "states",
            states,
        )

        next_states = _state_matrix(
            "next_states",
            next_states,
        )

        actions = _action_array(
            actions
        )

        if (
            states.shape[0]
            != next_states.shape[0]
            or states.shape[0]
            != actions.shape[0]
        ):
            raise ValueError(
                "prediction length mismatch"
            )

        result = self.predict_tensor(
            torch.as_tensor(
                states,
                dtype=torch.float32,
                device=self.device,
            ),

            torch.as_tensor(
                actions,
                dtype=torch.long,
                device=self.device,
            ),

            torch.as_tensor(
                next_states,
                dtype=torch.float32,
                device=self.device,
            ),
        )

        return (
            result
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

    def save_checkpoint(
        self,
        path: str | Path,
    ) -> None:

        (
            state_norm,
            reward_norm,
        ) = self._require_fitted()

        path = Path(
            path
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            {
                "format_version":
                    self.FORMAT_VERSION,

                "config":
                    asdict(
                        self.config
                    ),

                "state_normalizer": {
                    "mean":
                        state_norm.mean,

                    "std":
                        state_norm.std,

                    "eps":
                        state_norm.eps,
                },

                "reward_normalizer": {
                    "mean":
                        reward_norm.mean,

                    "std":
                        reward_norm.std,

                    "eps":
                        reward_norm.eps,
                },

                "model_state_dict":
                    self.model
                    .state_dict(),
            },
            path,
        )

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str = "cpu",
    ) -> "ResponseRewardPredictor":

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
                "unsupported reward "
                "checkpoint version"
            )

        instance = cls(
            ResponseRewardPredictorConfig(
                **payload[
                    "config"
                ]
            ),
            device=device,
        )

        state_payload = (
            payload[
                "state_normalizer"
            ]
        )

        reward_payload = (
            payload[
                "reward_normalizer"
            ]
        )

        instance.state_normalizer = (
            StateNormalizer(
                mean=np.asarray(
                    state_payload[
                        "mean"
                    ],
                    dtype=np.float32,
                ),

                std=np.asarray(
                    state_payload[
                        "std"
                    ],
                    dtype=np.float32,
                ),

                eps=float(
                    state_payload[
                        "eps"
                    ]
                ),
            )
        )

        instance.reward_normalizer = (
            ScalarNormalizer(
                mean=float(
                    reward_payload[
                        "mean"
                    ]
                ),

                std=float(
                    reward_payload[
                        "std"
                    ]
                ),

                eps=float(
                    reward_payload[
                        "eps"
                    ]
                ),
            )
        )

        instance.model.load_state_dict(
            payload[
                "model_state_dict"
            ]
        )

        instance.model.eval()

        return instance