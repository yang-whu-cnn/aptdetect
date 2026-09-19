from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol, Sequence

import numpy as np

from .node import ActionStatistics, SearchNode, UncertaintyTriple
from shared.d27_projection import D27ProjectionContext, project_d27_tensor


class WorldModel(Protocol):
    def predict_ensemble(self, state: np.ndarray, action: int) -> Sequence[Sequence[float]]: ...


class RewardModel(Protocol):
    def predict(self, state: np.ndarray, action: int, next_state: np.ndarray) -> float: ...


class ProgressModel(Protocol):
    def score(self, state: Sequence[float]) -> tuple[float, float]: ...


class PriorModel(Protocol):
    def probabilities(self, state: Sequence[float], actions: Sequence[int]) -> np.ndarray: ...


@dataclass(frozen=True)
class UAMCTSConfig:
    method_name: str = "UAMCTS-CC4 (adapted)"
    simulations: int = 64
    horizon: int = 4
    gamma: float = 0.99
    beta: float = 0.1
    c_puct: float = 1.0
    rho_max: float = 1.0
    c_u_max: float = 1.0
    phi: float = 1.0
    uncertainty_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        if self.method_name != "UAMCTS-CC4 (adapted)":
            raise ValueError("method must disclose the CC4 adaptation")
        if self.simulations <= 0 or self.horizon <= 0 or not 0 <= self.gamma <= 1:
            raise ValueError("invalid search budget, horizon, or discount")
        if self.beta < 0 or min(self.c_puct, self.rho_max, self.c_u_max) < 0:
            raise ValueError("search coefficients must be non-negative")
        if not 0 <= self.phi <= 1 or len(self.uncertainty_weights) != 3:
            raise ValueError("phi must be in [0,1] and three uncertainty weights are required")


def duration_aware_shaped_reward(base_reward: float, phi_state: float,
                                 phi_next: float, *, beta: float,
                                 gamma: float, duration: int) -> float:
    if duration <= 0:
        raise ValueError("action duration must be positive")
    return float(base_reward + beta * (gamma ** duration * phi_next - phi_state))


@dataclass(frozen=True)
class UAMCTSResult:
    action: int
    root_visits: tuple[int, int, int, int]
    root_q: tuple[float, float, float, float]
    simulations: int


class UAMCTSPlanner:
    """Fresh-root MCTS core. It returns only the most-visited root action."""

    def __init__(self, *, world_model: WorldModel, reward_model: RewardModel,
                 progress_model: ProgressModel, prior_model: PriorModel,
                 config: UAMCTSConfig | None = None, seed: int = 0,
                 uncertainty_normalizers=None):
        self.world_model = world_model
        self.reward_model = reward_model
        self.progress_model = progress_model
        self.prior_model = prior_model
        self.config = config or UAMCTSConfig()
        self.uncertainty_normalizers = uncertainty_normalizers
        self.projection_context = None
        self.rng = np.random.default_rng(int(seed))

    @staticmethod
    def _state(value: Sequence[float]) -> np.ndarray:
        state = np.asarray(value, dtype=np.float64)
        if state.shape != (27,) or not np.isfinite(state).all():
            raise ValueError("planner state must be a finite Blue-visible D27 vector")
        return state

    def _expand(self, node: SearchNode, available: Sequence[int]) -> None:
        actions = tuple(sorted(set(int(a) for a in available)))
        if not actions or any(a not in range(4) for a in actions):
            raise ValueError("available actions must be a non-empty A4 subset")
        priors = self.prior_model.probabilities(node.state, actions)
        prior_entropy = float(-sum(p * math.log(max(float(p), 1e-12)) for p in priors))
        _progress, progress_u = self.progress_model.score(node.state)
        if self.uncertainty_normalizers is not None:
            progress_u = self.uncertainty_normalizers.normalize("progress", progress_u)
            prior_entropy = self.uncertainty_normalizers.normalize("prior_entropy", prior_entropy)
        for action, prior in zip(actions, priors):
            node.actions[action] = ActionStatistics(
                prior=float(prior), uncertainty=UncertaintyTriple(0.0, progress_u, prior_entropy))

    def _transition(self, node: SearchNode, action: int) -> tuple[SearchNode, float, int]:
        state = np.asarray(node.state, dtype=np.float64)
        ensemble = np.asarray(self.world_model.predict_ensemble(state.copy(), action), dtype=np.float64)
        if ensemble.ndim != 2 or ensemble.shape[1:] != (27,) or len(ensemble) < 2:
            raise ValueError("world model must return at least two D27 ensemble predictions")
        if not np.isfinite(ensemble).all():
            raise ValueError("world model returned non-finite predictions")
        wm_u = float(np.var(ensemble, axis=0).mean())
        if self.uncertainty_normalizers is not None:
            wm_u = self.uncertainty_normalizers.normalize("world_model", wm_u)
        stats = node.actions[action]
        stats.uncertainty = UncertaintyTriple(wm_u, stats.uncertainty.progress,
                                              stats.uncertainty.prior)
        if action in node.children:
            child = node.children[action]
            next_state = np.asarray(child.state, dtype=np.float64)
        else:
            next_state = ensemble[int(self.rng.integers(len(ensemble)))]
            elapsed = node.elapsed_ticks + self._durations[action]
            if self.projection_context is not None:
                next_state = project_d27_tensor(next_state,context=self.projection_context,
                                                elapsed_ticks=elapsed).cpu().numpy()
            child = SearchNode(tuple(float(x) for x in next_state),elapsed_ticks=elapsed)
            node.children[action] = child
        return child, float(self.reward_model.predict(state, action, next_state)), 1

    def plan(self, state: Sequence[float], *, available_actions: Sequence[int],
             durations: Sequence[int] = (1, 1, 1, 1),
             projection_context: D27ProjectionContext | None = None) -> UAMCTSResult:
        if len(durations) != 4 or any(int(x) <= 0 for x in durations):
            raise ValueError("durations must contain four positive A4 values")
        initial = self._state(state)
        self.projection_context=projection_context
        self._durations=tuple(int(x) for x in durations)
        root = SearchNode(tuple(float(x) for x in initial),elapsed_ticks=0)
        self._expand(root, available_actions)
        cfg = self.config
        rho = cfg.rho_max * cfg.phi
        c_u = cfg.c_u_max * (1.0 - cfg.phi)

        for _ in range(cfg.simulations):
            node = root
            path: list[tuple[SearchNode, int, float, int, float, float]] = []
            actions_here = tuple(root.actions) if node is root else tuple(range(4))
            for _depth in range(cfg.horizon):
                if not node.actions:
                    self._expand(node, actions_here)
                action = node.select(c_puct=cfg.c_puct, rho=rho, c_u=c_u,
                                     uncertainty_weights=cfg.uncertainty_weights)
                phi_state, _ = self.progress_model.score(node.state)
                child, base_reward, _model_duration = self._transition(node, action)
                phi_next, _ = self.progress_model.score(child.state)
                duration = int(durations[action])
                shaped = duration_aware_shaped_reward(
                    base_reward, phi_state, phi_next, beta=cfg.beta,
                    gamma=cfg.gamma, duration=duration)
                path.append((node, action, shaped, duration, phi_state, phi_next))
                node = child
                actions_here = tuple(range(4))

            value = 0.0
            for visited, action, reward, duration, _p0, _p1 in reversed(path):
                value = reward + cfg.gamma ** duration * value
                visited.backup(action, value)

        root_action = max(root.actions, key=lambda a: (root.actions[a].n, root.actions[a].q, -a))
        visits = tuple(root.actions[a].n if a in root.actions else 0 for a in range(4))
        q_values = tuple(root.actions[a].q if a in root.actions else 0.0 for a in range(4))
        # No tree is retained on self: every decision starts from a fresh root.
        return UAMCTSResult(root_action, visits, q_values, cfg.simulations)
