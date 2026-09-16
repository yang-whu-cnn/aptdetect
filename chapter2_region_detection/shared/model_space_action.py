from __future__ import annotations

import numpy as np
import torch

from shared.action_contract import N_ACTIONS
from shared.formal_state import (
    FORMAL_STATE_DIM,
    FORMAL_STATE_FEATURE_NAMES,
)


TARGET_THRESHOLD = 0.5

ANY_TARGET_INDEX = FORMAL_STATE_FEATURE_NAMES.index(
    "any_valid_observable_target"
)


def canonicalize_requested_action(
    state,
    requested_action_id: int,
    *,
    threshold: float = TARGET_THRESHOLD,
) -> int:
    """
    Convert planner-requested action to the canonical model-space action.

    Contract:
      - no_op (0) always remains Sleep/no_op;
      - targeted action + no valid observable target -> Sleep/no_op;
      - targeted action + valid observable target -> keep requested family.

    This function uses only the planner-visible 27D state.
    """
    state = np.asarray(state, dtype=np.float32)

    if state.shape != (FORMAL_STATE_DIM,):
        raise ValueError("state shape mismatch")

    if not np.all(np.isfinite(state)):
        raise ValueError("state contains non-finite values")

    requested_action_id = int(requested_action_id)

    if not 0 <= requested_action_id < N_ACTIONS:
        raise ValueError("invalid requested action")

    if requested_action_id == 0:
        return 0

    available = float(state[ANY_TARGET_INDEX]) >= float(threshold)
    return requested_action_id if available else 0


def canonicalize_requested_tensor(
    states: torch.Tensor,
    requested_actions: torch.Tensor,
    *,
    threshold: float = TARGET_THRESHOLD,
) -> torch.Tensor:
    """Vectorized version of canonicalize_requested_action."""
    if states.ndim != 2 or states.shape[1] != FORMAL_STATE_DIM:
        raise ValueError("states must have shape (N,D)")

    if (
        requested_actions.ndim != 1
        or requested_actions.shape[0] != states.shape[0]
    ):
        raise ValueError("requested action shape mismatch")

    requested_actions = requested_actions.to(dtype=torch.long)

    if requested_actions.numel():
        if (
            int(requested_actions.min()) < 0
            or int(requested_actions.max()) >= N_ACTIONS
        ):
            raise ValueError("invalid requested action")

    available = states[:, ANY_TARGET_INDEX] >= float(threshold)
    targeted = requested_actions != 0

    return torch.where(
        targeted & (~available),
        torch.zeros_like(requested_actions),
        requested_actions,
    )
