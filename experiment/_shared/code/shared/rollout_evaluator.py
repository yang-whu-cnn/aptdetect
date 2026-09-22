from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch

from shared.action_contract import (
    ACTION_CONTRACTS,
    N_ACTIONS,
)
from shared.formal_state import FORMAL_STATE_DIM
from shared.model_space_action import canonicalize_requested_tensor
from shared.d27_projection import D27ProjectionContext, project_d27_tensor


ACTION_DURATION = torch.tensor(
    [int(item.duration_ticks) for item in ACTION_CONTRACTS],
    dtype=torch.long,
)


@dataclass(frozen=True)
class SharedRolloutConfig:
    """Frozen Step-4 planning semantics."""

    horizon: int = 4
    gamma_tick: float = 0.99

    def __post_init__(self) -> None:
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, int):
            raise ValueError("horizon must be an integer")
        if self.horizon <= 0:
            raise ValueError("horizon must be > 0")

        gamma = float(self.gamma_tick)
        if not isfinite(gamma) or not (0.0 < gamma <= 1.0):
            raise ValueError("gamma_tick must be finite and in (0, 1]")


@dataclass(frozen=True)
class SharedRolloutResult:
    """
    Vectorized fixed-member rollout result.

    next_states:
        [H,N,M,D]

    member_returns:
        [N,M]

    expected_return:
        [N]
    """

    next_states: torch.Tensor
    member_returns: torch.Tensor
    expected_return: torch.Tensor


class SharedRolloutEvaluator:
    """
    Shared model-based rollout evaluator for LWM-RL / UG-CEM / CEM.

    This class deliberately contains no CEM, UG penalty, PPO, LLM,
    plan ranking, or planner-specific policy logic.

    Formal semantics:
      - state D=27;
      - requested categorical actions A=4;
      - requested action is canonicalized from the member's current
        planner-visible predicted state before each model step;
      - each ensemble member remains fixed for the whole horizon;
      - deterministic member mean rollout only;
      - no aleatoric resampling;
      - reward uses the shared response reward predictor;
      - discount is gamma_tick ** cumulative elapsed global ticks;
      - vectorization is over N plans, so model forward complexity is
        M * H rather than N * M * H.
    """

    def __init__(
        self,
        *,
        world_model,
        reward_predictor,
        config: SharedRolloutConfig | None = None,
    ):
        self.world_model = world_model
        self.reward_predictor = reward_predictor
        self.config = config if config is not None else SharedRolloutConfig()

        self._validate_components()

        self.device = torch.device(self.world_model.device)
        self.ensemble_size = int(self.world_model.config.ensemble_size)

        self._duration = ACTION_DURATION.to(device=self.device)

    def _validate_components(self) -> None:
        for name, component in (
            ("world_model", self.world_model),
            ("reward_predictor", self.reward_predictor),
        ):
            if not hasattr(component, "config"):
                raise TypeError(f"{name} must expose config")
            if not hasattr(component, "device"):
                raise TypeError(f"{name} must expose device")

        wm_cfg = self.world_model.config
        reward_cfg = self.reward_predictor.config

        if int(wm_cfg.state_dim) != FORMAL_STATE_DIM:
            raise ValueError("world model state_dim contract mismatch")
        if int(wm_cfg.n_actions) != N_ACTIONS:
            raise ValueError("world model action contract mismatch")
        if int(wm_cfg.ensemble_size) <= 0:
            raise ValueError("world model ensemble must be non-empty")
        if str(wm_cfg.target_mode) != "absolute":
            raise ValueError("formal Step-4 requires selected absolute world model")

        if int(reward_cfg.state_dim) != FORMAL_STATE_DIM:
            raise ValueError("reward model state_dim contract mismatch")
        if int(reward_cfg.n_actions) != N_ACTIONS:
            raise ValueError("reward model action contract mismatch")

        wm_device = torch.device(self.world_model.device)
        reward_device = torch.device(self.reward_predictor.device)
        if wm_device != reward_device:
            raise ValueError("world model and reward predictor must share device")

        if not hasattr(self.world_model, "predict_member_mean_tensor"):
            raise TypeError("world model lacks predict_member_mean_tensor")
        if not hasattr(self.reward_predictor, "predict_tensor"):
            raise TypeError("reward predictor lacks predict_tensor")

    def _prepare_state(self, state) -> torch.Tensor:
        state_t = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        )

        if state_t.shape != (FORMAL_STATE_DIM,):
            raise ValueError(
                f"state must have shape ({FORMAL_STATE_DIM},)"
            )

        if not torch.isfinite(state_t).all():
            raise ValueError("state must contain only finite values")

        return state_t

    def _prepare_plans(self, plans) -> torch.Tensor:
        raw = torch.as_tensor(plans, device=self.device)

        if raw.ndim != 2:
            raise ValueError("plans must have shape [N,H]")

        n_plans, horizon = raw.shape
        if n_plans <= 0:
            raise ValueError("N must be > 0")
        if horizon != self.config.horizon:
            raise ValueError(
                f"plan horizon mismatch: expected {self.config.horizon}, got {horizon}"
            )

        if raw.dtype.is_floating_point:
            if not torch.isfinite(raw).all():
                raise ValueError("plans must contain finite action IDs")
            if not torch.equal(raw, torch.round(raw)):
                raise ValueError("plans must contain integer categorical IDs")

        plans_t = raw.to(dtype=torch.long)

        if plans_t.numel():
            if int(plans_t.min()) < 0 or int(plans_t.max()) >= N_ACTIONS:
                raise ValueError("plans contain invalid action IDs")

        return plans_t

    @torch.no_grad()
    def evaluate(
        self,
        state,
        plans,
        *, projection_context: D27ProjectionContext | None = None,
    ) -> SharedRolloutResult:
        state_t = self._prepare_state(state)
        plans_t = self._prepare_plans(plans)

        n_plans = int(plans_t.shape[0])
        horizon = int(self.config.horizon)
        members = int(self.ensemble_size)

        current = (
            state_t
            .view(1, 1, FORMAL_STATE_DIM)
            .expand(n_plans, members, FORMAL_STATE_DIM)
            .clone()
        )

        next_states = torch.empty(
            (horizon, n_plans, members, FORMAL_STATE_DIM),
            dtype=torch.float32,
            device=self.device,
        )

        member_returns = torch.zeros(
            (n_plans, members),
            dtype=torch.float32,
            device=self.device,
        )

        elapsed_ticks = torch.zeros(
            (n_plans, members),
            dtype=torch.long,
            device=self.device,
        )

        gamma = torch.tensor(
            float(self.config.gamma_tick),
            dtype=torch.float32,
            device=self.device,
        )

        for step in range(horizon):
            requested = plans_t[:, step]

            for member_idx in range(members):
                current_member = current[:, member_idx, :]

                canonical = canonicalize_requested_tensor(
                    current_member,
                    requested,
                )

                predicted_next = self.world_model.predict_member_mean_tensor(
                    member_idx,
                    current_member,
                    canonical,
                )

                if predicted_next.shape != (n_plans, FORMAL_STATE_DIM):
                    raise RuntimeError("world model returned invalid next-state shape")
                if not torch.isfinite(predicted_next).all():
                    raise ValueError("world model returned non-finite next state")

                if projection_context is not None:
                    next_elapsed = elapsed_ticks[:, member_idx] + self._duration[canonical]
                    predicted_next = project_d27_tensor(predicted_next, context=projection_context,
                                                        elapsed_ticks=next_elapsed)

                predicted_reward = self.reward_predictor.predict_tensor(
                    current_member,
                    canonical,
                    predicted_next,
                )

                if predicted_reward.shape != (n_plans,):
                    raise RuntimeError("reward predictor returned invalid shape")
                if not torch.isfinite(predicted_reward).all():
                    raise ValueError("reward predictor returned non-finite reward")

                discount = torch.pow(
                    gamma,
                    elapsed_ticks[:, member_idx].to(dtype=torch.float32),
                )

                member_returns[:, member_idx] += discount * predicted_reward

                elapsed_ticks[:, member_idx] += self._duration[canonical]

                next_states[step, :, member_idx, :] = predicted_next
                current[:, member_idx, :] = predicted_next

        if not torch.isfinite(next_states).all():
            raise ValueError("rollout produced non-finite next_states")
        if not torch.isfinite(member_returns).all():
            raise ValueError("rollout produced non-finite member_returns")

        expected_return = member_returns.mean(dim=1)

        if expected_return.shape != (n_plans,):
            raise RuntimeError("expected_return shape mismatch")
        if not torch.isfinite(expected_return).all():
            raise ValueError("rollout produced non-finite expected_return")

        return SharedRolloutResult(
            next_states=next_states,
            member_returns=member_returns,
            expected_return=expected_return,
        )
