from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from formal_experiments.ours.llm_prior_v2 import (
    FORMAL_HORIZON,
    FORMAL_K_CANDIDATES,
)
from shared.action_contract import N_ACTIONS
from shared.formal_state import FORMAL_STATE_DIM


PLAN_ONE_HOT_DIM = FORMAL_HORIZON * N_ACTIONS
POSTERIOR_CANDIDATE_DIM = FORMAL_STATE_DIM + PLAN_ONE_HOT_DIM + 3


@dataclass(frozen=True)
class CandidateEvidence:
    expected_return: torch.Tensor
    predictive_uncertainty: torch.Tensor


def encode_plan_one_hot(plans, *, device=None) -> torch.Tensor:
    plans_t = torch.as_tensor(plans, dtype=torch.long, device=device)
    if tuple(plans_t.shape) != (FORMAL_K_CANDIDATES, FORMAL_HORIZON):
        raise ValueError("plans must have shape [6,4]")
    if int(plans_t.min()) < 0 or int(plans_t.max()) >= N_ACTIONS:
        raise ValueError("plans contain invalid action IDs")
    one_hot = torch.nn.functional.one_hot(
        plans_t,
        num_classes=N_ACTIONS,
    ).to(dtype=torch.float32)
    return one_hot.reshape(FORMAL_K_CANDIDATES, PLAN_ONE_HOT_DIM)


def evidence_from_rollout(rollout_result) -> CandidateEvidence:
    expected = torch.as_tensor(
        rollout_result.expected_return,
        dtype=torch.float32,
    )
    member_returns = torch.as_tensor(
        rollout_result.member_returns,
        dtype=torch.float32,
        device=expected.device,
    )

    if tuple(expected.shape) != (FORMAL_K_CANDIDATES,):
        raise ValueError("expected_return must have shape [6]")
    if member_returns.ndim != 2 or member_returns.shape[0] != FORMAL_K_CANDIDATES:
        raise ValueError("member_returns must have shape [6,M]")
    if member_returns.shape[1] <= 1:
        raise ValueError("member_returns requires ensemble M>1")
    if not torch.isfinite(expected).all() or not torch.isfinite(member_returns).all():
        raise ValueError("rollout evidence must be finite")

    member_mean = member_returns.mean(dim=1)
    if not torch.allclose(expected, member_mean, rtol=1e-4, atol=1e-5):
        raise ValueError("expected_return must equal member-return mean")

    uncertainty = member_returns.std(dim=1, unbiased=False)
    if not torch.isfinite(uncertainty).all():
        raise ValueError("predictive uncertainty must be finite")

    return CandidateEvidence(
        expected_return=expected,
        predictive_uncertainty=uncertainty,
    )


def build_posterior_candidate_features(
    state,
    plans,
    prior_preferences,
    rollout_result,
) -> torch.Tensor:
    evidence = evidence_from_rollout(rollout_result)
    device = evidence.expected_return.device

    state_t = torch.as_tensor(state, dtype=torch.float32, device=device)
    if tuple(state_t.shape) != (FORMAL_STATE_DIM,):
        raise ValueError("state must have shape [27]")
    if not torch.isfinite(state_t).all():
        raise ValueError("state must be finite")

    plan_oh = encode_plan_one_hot(plans, device=device)

    prior = torch.as_tensor(
        prior_preferences,
        dtype=torch.float32,
        device=device,
    )
    if tuple(prior.shape) != (FORMAL_K_CANDIDATES,):
        raise ValueError("prior_preferences must have shape [6]")
    if not torch.isfinite(prior).all():
        raise ValueError("prior_preferences must be finite")
    if torch.any(prior < 0.0):
        raise ValueError("prior_preferences must be non-negative")
    if abs(float(prior.sum().item()) - 1.0) > 1e-4:
        raise ValueError("prior_preferences must sum to 1")

    repeated_state = state_t.view(1, -1).expand(FORMAL_K_CANDIDATES, -1)
    features = torch.cat(
        [
            repeated_state,
            plan_oh,
            prior.view(-1, 1),
            evidence.expected_return.view(-1, 1),
            evidence.predictive_uncertainty.view(-1, 1),
        ],
        dim=1,
    )

    if tuple(features.shape) != (FORMAL_K_CANDIDATES, POSTERIOR_CANDIDATE_DIM):
        raise RuntimeError("posterior feature shape mismatch")
    if not torch.isfinite(features).all():
        raise ValueError("posterior features must be finite")

    return features
