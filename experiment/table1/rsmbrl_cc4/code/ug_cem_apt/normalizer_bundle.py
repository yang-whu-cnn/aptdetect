from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch

from baselines.ug_cem_apt.uncertainty import UGUncertainty
from shared.formal_state import BLUE_AGENTS, FORMAL_STATE_DIM


FORMAL_HORIZON = 4
EXPECTED_FORMAT_VERSION = 1
EXPECTED_CALIBRATION_SEEDS = tuple(range(3000, 3008))


def _validated_tensor(name: str, value, *, shape: tuple[int, ...], device) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device).detach().clone()
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{name} shape mismatch: expected {shape}, got {tuple(tensor.shape)}")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must be finite")
    return tensor


def load_ug_normalizer_bundle(path: str | Path, *, map_location: str = "cpu") -> dict:
    payload = torch.load(Path(path), map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError("normalizer bundle must be a dict")
    if int(payload.get("format_version", -1)) != EXPECTED_FORMAT_VERSION:
        raise ValueError("unsupported normalizer bundle format")
    if str(payload.get("source_split")) != "calibration":
        raise ValueError("formal normalizer bundle must come from calibration split")
    seeds = tuple(int(x) for x in payload.get("calibration_seeds", ()))
    if seeds != EXPECTED_CALIBRATION_SEEDS:
        raise ValueError("calibration seed contract mismatch")
    if int(payload.get("planner_calls_per_agent", -1)) != 100:
        raise ValueError("planner_calls_per_agent must be 100")
    if bool(payload.get("freeze_after_warmup", True)):
        raise ValueError("formal main bundle must not be frozen after warm-up")
    if not bool(payload.get("online_updates_after_warmup", False)):
        raise ValueError("formal main bundle must keep online EMA enabled")

    agents = payload.get("agents")
    if not isinstance(agents, Mapping):
        raise ValueError("bundle agents must be a mapping")
    if set(agents.keys()) != set(BLUE_AGENTS):
        raise ValueError("bundle must contain exactly all five Blue agents")

    # Validate every stored tensor now so corrupted bundles fail before smoke.
    for agent in BLUE_AGENTS:
        item = agents[agent]
        if not isinstance(item, Mapping):
            raise ValueError(f"{agent}: normalizer entry must be a mapping")
        obs_mean = _validated_tensor(
            f"{agent}.obs_mean", item.get("obs_mean"), shape=(FORMAL_STATE_DIM,), device=map_location
        )
        obs_std = _validated_tensor(
            f"{agent}.obs_std", item.get("obs_std"), shape=(FORMAL_STATE_DIM,), device=map_location
        )
        horizon_std = _validated_tensor(
            f"{agent}.horizon_std", item.get("horizon_std"), shape=(FORMAL_HORIZON,), device=map_location
        )
        if not torch.all(obs_std > 0):
            raise ValueError(f"{agent}.obs_std must be > 0")
        if not torch.all(horizon_std > 0):
            raise ValueError(f"{agent}.horizon_std must be > 0")

        # Replace with validated CPU/device copies for deterministic downstream use.
        item = dict(item)
        item["obs_mean"] = obs_mean
        item["obs_std"] = obs_std
        item["horizon_std"] = horizon_std
        agents[agent] = item

    payload = dict(payload)
    payload["agents"] = dict(agents)
    return payload


def restore_agent_normalizer(
    uncertainty: UGUncertainty,
    *,
    bundle: Mapping,
    agent_name: str,
) -> None:
    if not isinstance(uncertainty, UGUncertainty):
        raise TypeError("uncertainty must be UGUncertainty")
    if agent_name not in BLUE_AGENTS:
        raise ValueError(f"unsupported agent: {agent_name}")
    agents = bundle.get("agents") if isinstance(bundle, Mapping) else None
    if not isinstance(agents, Mapping) or agent_name not in agents:
        raise ValueError("bundle does not contain requested agent")

    item = agents[agent_name]
    device = uncertainty.device
    obs_mean = _validated_tensor(
        "obs_mean", item["obs_mean"], shape=(FORMAL_STATE_DIM,), device=device
    )
    obs_std = _validated_tensor(
        "obs_std", item["obs_std"], shape=(FORMAL_STATE_DIM,), device=device
    )
    horizon_std = _validated_tensor(
        "horizon_std", item["horizon_std"], shape=(FORMAL_HORIZON,), device=device
    )
    if not torch.all(obs_std > 0):
        raise ValueError("obs_std must be > 0")
    if not torch.all(horizon_std > 0):
        raise ValueError("horizon_std must be > 0")

    uncertainty.obs_mean = obs_mean
    uncertainty.obs_std = obs_std
    uncertainty.horizon_std = horizon_std
    uncertainty._state_dim = FORMAL_STATE_DIM
    uncertainty._horizon = FORMAL_HORIZON


def snapshot_matches_bundle(
    uncertainty: UGUncertainty,
    *,
    bundle: Mapping,
    agent_name: str,
) -> bool:
    if uncertainty.obs_mean is None or uncertainty.obs_std is None or uncertainty.horizon_std is None:
        return False
    item = bundle["agents"][agent_name]
    return bool(
        torch.equal(uncertainty.obs_mean.detach().cpu(), torch.as_tensor(item["obs_mean"]).detach().cpu())
        and torch.equal(uncertainty.obs_std.detach().cpu(), torch.as_tensor(item["obs_std"]).detach().cpu())
        and torch.equal(uncertainty.horizon_std.detach().cpu(), torch.as_tensor(item["horizon_std"]).detach().cpu())
    )
