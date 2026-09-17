from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping
import hashlib

import numpy as np
import torch

from formal_experiments.ours.llm_prior_v2 import PriorBatch
from formal_experiments.ours.model_registry import LLMModelRegistry, LLMModelSpec
from formal_experiments.ours.posterior_features import build_posterior_candidate_features
from formal_experiments.ours.ppo_core import CandidateActorCritic
from formal_experiments.ours.prior_cache import PriorCache, make_identity
from shared.formal_state import BLUE_AGENTS, FORMAL_STATE_DIM


LivePriorGenerator = Callable[[np.ndarray, str], tuple[PriorBatch, Mapping[str, object]]]


@dataclass(frozen=True)
class PreparedCandidates:
    agent_name: str
    state: np.ndarray
    prior: PriorBatch
    candidate_features: torch.Tensor
    critic_value: float
    cache_key: str
    cache_hit: bool
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class CandidateSelection:
    candidate_index: int
    requested_action_id: int
    behavior_log_prob: float
    entropy: float
    critic_value: float
    selected_plan: tuple[int, ...]


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class LWMDecisionRuntime:
    """Provider-neutral formal LWM-RL decision runtime.

    The runtime performs exactly one decision-epoch evidence path:

        visible D27 + public agent id
        -> model/split isolated prior cache
        -> K6/H4 candidate plans
        -> shared frozen WM/reward evaluator
        -> 46D candidate posterior features
        -> shared PPO policy over candidate indices
        -> selected plan[0] as requested A4 action

    It deliberately does not resolve a host target or touch the environment. Those
    operations remain in the shared resolver/adapter layer used by all methods.
    """

    def __init__(
        self,
        *,
        registry: LLMModelRegistry,
        model_spec: LLMModelSpec,
        registry_sha256: str,
        evaluator,
        prior_cache: PriorCache,
        policy: CandidateActorCritic,
        live_prior_generator: LivePriorGenerator,
        split: str,
    ) -> None:
        if bool(registry.primary_model_selected):
            raise ValueError("runtime cannot use a registry with premature primary selection")
        if model_spec.experiment_alias not in {
            item.experiment_alias for item in registry.models.values()
        }:
            raise ValueError("model_spec is not a member of the supplied registry")
        if split not in {"train", "validation", "test"}:
            raise ValueError("unsupported runtime split")
        if not isinstance(prior_cache, PriorCache):
            raise TypeError("prior_cache must be PriorCache")
        if not isinstance(policy, CandidateActorCritic):
            raise TypeError("policy must be CandidateActorCritic")
        if not callable(live_prior_generator):
            raise TypeError("live_prior_generator must be callable")
        if len(str(registry_sha256)) != 64:
            raise ValueError("registry_sha256 must be SHA256 hex")

        self.registry = registry
        self.model_spec = model_spec
        self.registry_sha256 = str(registry_sha256)
        self.evaluator = evaluator
        self.prior_cache = prior_cache
        self.policy = policy
        self.live_prior_generator = live_prior_generator
        self.split = str(split)

        self.cache_hits = 0
        self.cache_misses = 0
        self.prepare_calls = 0

    @property
    def generation_config(self) -> dict:
        return {
            "temperature": float(self.model_spec.actual_temperature),
            "structured_output": str(self.model_spec.structured_output_requested),
            "public_agent_identifier": True,
        }

    @property
    def live_api_calls(self) -> int:
        # Every miss is one logical provider transaction; bounded retries are tracked
        # in metadata and are not counted as distinct planner decisions here.
        return int(self.cache_misses)

    def _validated_state(self, state) -> np.ndarray:
        array = np.asarray(state, dtype=np.float32)
        if array.shape != (FORMAL_STATE_DIM,):
            raise ValueError("runtime state must have shape [27]")
        if not np.isfinite(array).all():
            raise ValueError("runtime state must be finite")
        return array.copy()

    def prepare(self, state, *, agent_name: str) -> PreparedCandidates:
        agent = str(agent_name)
        if agent not in BLUE_AGENTS:
            raise ValueError(f"unsupported Blue agent: {agent}")
        state_array = self._validated_state(state)

        identity = make_identity(
            split=self.split,
            model_alias=self.model_spec.experiment_alias,
            provider=self.model_spec.provider,
            exact_model_id=self.model_spec.exact_model_id,
            registry_version=self.registry.version,
            registry_sha256=self.registry_sha256,
            prompt_version=self.registry.prompt_version,
            agent_name=agent,
            generation_config=self.generation_config,
            state=state_array,
        )

        lookup = self.prior_cache.get_or_generate(
            identity,
            lambda: self.live_prior_generator(state_array.copy(), agent),
        )
        if lookup.cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

        cached = lookup.cached_prior
        prior = cached.prior
        rollout = self.evaluator.evaluate(state_array, prior.plans)
        features = build_posterior_candidate_features(
            state_array,
            prior.plans,
            prior.prior_preferences,
            rollout,
        )

        self.policy.eval()
        with torch.no_grad():
            _logits, value = self.policy(features)
        if value.ndim != 0 or not torch.isfinite(value):
            raise RuntimeError("PPO critic emitted invalid scalar value")
        if tuple(features.shape) != (6, 46) or not torch.isfinite(features).all():
            raise RuntimeError("runtime posterior feature contract failed")

        self.prepare_calls += 1
        return PreparedCandidates(
            agent_name=agent,
            state=state_array,
            prior=prior,
            candidate_features=features.detach().cpu().clone(),
            critic_value=float(value.detach().cpu()),
            cache_key=str(cached.cache_key),
            cache_hit=bool(lookup.cache_hit),
            metadata=dict(cached.metadata),
        )

    def select(
        self,
        prepared: PreparedCandidates,
        *,
        deterministic: bool = False,
    ) -> CandidateSelection:
        if not isinstance(prepared, PreparedCandidates):
            raise TypeError("prepared must be PreparedCandidates")
        if prepared.agent_name not in BLUE_AGENTS:
            raise ValueError("prepared candidate set has invalid agent")

        self.policy.eval()
        with torch.no_grad():
            action, log_prob, entropy, value = self.policy.act(
                prepared.candidate_features,
                deterministic=bool(deterministic),
            )

        candidate_index = int(action.item())
        if not 0 <= candidate_index < 6:
            raise RuntimeError("PPO selected candidate index outside K=6")
        plan = tuple(int(x) for x in prepared.prior.plans[candidate_index].tolist())
        if len(plan) != 4:
            raise RuntimeError("selected plan does not have H=4")
        requested_action_id = int(plan[0])
        if not 0 <= requested_action_id < 4:
            raise RuntimeError("selected plan[0] is outside A4")

        critic_value = float(value.detach().cpu())
        if abs(critic_value - float(prepared.critic_value)) > 1e-5:
            raise RuntimeError("critic value changed between prepare and selection")

        return CandidateSelection(
            candidate_index=candidate_index,
            requested_action_id=requested_action_id,
            behavior_log_prob=float(log_prob.detach().cpu()),
            entropy=float(entropy.detach().cpu()),
            critic_value=critic_value,
            selected_plan=plan,
        )
