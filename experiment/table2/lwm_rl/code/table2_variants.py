"""Development contract for independently-trained Table-2 variants."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from formal_experiments.ours.llm_prior_v2 import FORMAL_HORIZON, FORMAL_K_CANDIDATES
from formal_experiments.ours.posterior_features import build_posterior_candidate_features, encode_plan_one_hot
from formal_experiments.ours.ppo_core import CandidateActorCritic


class Table2Variant(str, Enum):
    RL_ONLY = "RL-Only"
    LLM_RL = "LLM-RL"
    WM_RL = "WM-RL"
    LWM_RL = "LWM-RL"


@dataclass(frozen=True)
class VariantSpec:
    variant: Table2Variant
    uses_llm_prior: bool
    uses_world_model: bool
    candidate_policy: bool


SPECS = {
    Table2Variant.RL_ONLY: VariantSpec(Table2Variant.RL_ONLY, False, False, False),
    Table2Variant.LLM_RL: VariantSpec(Table2Variant.LLM_RL, True, False, True),
    Table2Variant.WM_RL: VariantSpec(Table2Variant.WM_RL, False, True, True),
    Table2Variant.LWM_RL: VariantSpec(Table2Variant.LWM_RL, True, True, True),
}


@dataclass
class ModuleAudit:
    prior_calls: int = 0
    wm_calls: int = 0
    policy_calls: int = 0
    online_llm_calls: int = 0
    cache_misses: int = 0
    selected_first_actions: list[int] = field(default_factory=list)


class DirectA4ActorCritic(nn.Module):
    def __init__(self, *, init_seed: int):
        super().__init__(); torch.manual_seed(int(init_seed))
        self.body = nn.Sequential(nn.Linear(27, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.actor = nn.Linear(128, 4); self.critic = nn.Linear(128, 1)

    def forward(self, state):
        device = next(self.parameters()).device
        state = torch.as_tensor(state, dtype=torch.float32, device=device)
        if state.shape[-1] != 27 or not torch.isfinite(state).all(): raise ValueError("finite D27 required")
        hidden = self.body(state); return self.actor(hidden), self.critic(hidden).squeeze(-1)


def non_llm_candidate_plans(state, *, agent_name: str) -> np.ndarray:
    """Deterministic uniform A4 coverage, independent of hidden truth."""
    checksum = int(np.asarray(state, dtype=np.float32).view(np.uint8).sum()) + sum(map(ord, agent_name))
    c = checksum % 4
    plans = np.asarray([
        [0, c, (c + 1) % 4, (c + 2) % 4],
        [1, (c + 1) % 4, (c + 2) % 4, (c + 3) % 4],
        [2, (c + 2) % 4, (c + 3) % 4, c],
        [3, (c + 3) % 4, c, (c + 1) % 4],
        [0, (c + 2) % 4, c, (c + 3) % 4],
        [1, (c + 3) % 4, (c + 1) % 4, c],
    ], dtype=np.int64)
    if len({tuple(int(x) for x in row) for row in plans}) != 6:
        raise RuntimeError("non-LLM candidate generator uniqueness invariant failed")
    return plans


class Table2DecisionRuntime:
    def __init__(self, *, variant: Table2Variant, init_seed: int,
                 offline_prior=None, frozen_wm_evaluator=None, split: str = "train",
                 device: str = "cpu"):
        self.spec = SPECS[variant]; self.init_seed = int(init_seed); self.audit = ModuleAudit()
        if split not in ("train", "validation", "test"): raise ValueError("split isolation violation")
        self.split = split
        self.offline_prior = offline_prior; self.wm = frozen_wm_evaluator
        torch.manual_seed(self.init_seed)
        self.policy = (CandidateActorCritic() if self.spec.candidate_policy
                       else DirectA4ActorCritic(init_seed=init_seed)).to(torch.device(device))
        if self.spec.uses_llm_prior and offline_prior is None: raise ValueError("LLM variant requires offline prior")
        if self.spec.uses_world_model and frozen_wm_evaluator is None: raise ValueError("WM variant requires frozen evaluator")
        if offline_prior is not None and getattr(offline_prior, "online_allowed", False):
            raise ValueError("online LLM is forbidden")
        if frozen_wm_evaluator is not None and not getattr(frozen_wm_evaluator, "frozen", False):
            raise ValueError("world model and Full-Reward predictor must be frozen")

    def decide(self, state, *, agent_name: str, action_mask: list[bool], deterministic=True,
               projection_context=None) -> dict[str, Any]:
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (27,) or len(action_mask) != 4 or not action_mask[0]: raise ValueError("D27/A4 mask violation")
        self.audit.policy_calls += 1
        if not self.spec.candidate_policy:
            logits, value = self.policy(state)
            masked = logits.masked_fill(
                ~torch.tensor(action_mask, device=logits.device), float("-inf")
            )
            dist = Categorical(logits=masked)
            action_t = masked.argmax() if deterministic else dist.sample(); action = int(action_t)
            self.audit.selected_first_actions.append(action)
            return {"requested_action": action, "selected_plan": None,
                    "policy_input": torch.tensor(state).detach().clone(), "policy_action": action,
                    "policy_mask": torch.tensor(action_mask, dtype=torch.bool),
                    "log_prob": float(dist.log_prob(action_t).detach()),
                    "value": float(value.detach().cpu()), "audit": asdict(self.audit)}

        if self.spec.uses_llm_prior:
            self.audit.prior_calls += 1
            prior = self.offline_prior.load(state, agent_name=agent_name, split=self.split)
            if prior is None:
                self.audit.cache_misses += 1; raise RuntimeError("offline cache miss (fail closed)")
            plans = np.asarray(prior.plans, dtype=np.int64); preferences = np.asarray(prior.prior_preferences, dtype=np.float32)
        else:
            plans = non_llm_candidate_plans(state, agent_name=agent_name); preferences = np.full(6, 1/6, dtype=np.float32)

        if plans.shape != (FORMAL_K_CANDIDATES, FORMAL_HORIZON): raise ValueError("K6/H4 required")
        if np.any(plans < 0) or np.any(plans >= 4):
            raise ValueError("candidate plan contains an action outside A4")
        if len({tuple(int(x) for x in row) for row in plans}) != len(plans):
            raise ValueError("exact duplicate candidate plans are forbidden (fail closed)")
        if (preferences.shape != (FORMAL_K_CANDIDATES,)
                or not np.isfinite(preferences).all() or np.any(preferences < 0)
                or not np.isclose(float(preferences.sum()), 1.0, atol=1e-5)):
            raise ValueError("candidate prior preferences must be finite nonnegative K6 summing to one")
        plans_before = plans.copy()
        # The formal prior parser and deterministic non-LLM generator already
        # guarantee exact-plan uniqueness. Distinct plans sharing a first
        # action remain separate categorical choices by design.
        plans_after = plans.copy()
        if self.spec.uses_world_model:
            if projection_context is None:
                raise ValueError("model-based Table-2 decision requires D27 projection context")
            self.audit.wm_calls += 1; rollout = self.wm.evaluate(
                state, plans, projection_context=projection_context)
            features = build_posterior_candidate_features(state, plans, preferences, rollout)
        else:
            repeated = torch.tensor(state).view(1, -1).expand(6, -1)
            features = torch.cat((repeated, encode_plan_one_hot(plans),
                                  torch.tensor(preferences).view(-1, 1), torch.zeros(6, 2)), dim=1)
        policy_device = next(self.policy.parameters()).device
        features = features.to(policy_device)
        logits, value = self.policy(features)
        candidate_mask = torch.tensor(
            [action_mask[int(plan[0])] for plan in plans], device=logits.device
        )
        if not bool(candidate_mask.any()):
            raise RuntimeError("all candidate first actions are unavailable (fail closed)")
        logits = logits.masked_fill(~candidate_mask, float("-inf"))
        dist = Categorical(logits=logits); selected_t = logits.argmax() if deterministic else dist.sample()
        selected = int(selected_t)
        plan = tuple(int(x) for x in plans[selected]); action = plan[0]
        self.audit.selected_first_actions.append(action)
        return {"requested_action": action, "selected_plan": plan, "candidate_index": selected,
                "policy_input": features.detach().clone(), "policy_action": selected,
                "policy_mask": candidate_mask.detach().clone(),
                "candidate_plans_before_dedup": plans_before.tolist(),
                "candidate_plans_after_dedup": plans_after.tolist(),
                "candidate_action_mask": candidate_mask.detach().cpu().tolist(),
                "candidate_selection_probabilities": dist.probs.detach().cpu().tolist(),
                "candidate_probability_rule": (
                    "categorical_over_unique_full_plans;exact_duplicates_forbidden;"
                    "distinct_same_first_action_plans_retained;no_action_mass_aggregation"
                ),
                "log_prob": float(dist.log_prob(selected_t).detach()),
                "value": float(value.detach().cpu()), "audit": asdict(self.audit)}


def assert_independent_initializations(runtimes: list[Table2DecisionRuntime]) -> None:
    if len({runtime.init_seed for runtime in runtimes}) != len(runtimes):
        raise ValueError("Table-2 variants must use independent initialization seeds")
