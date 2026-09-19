"""Strict local artifact gate for RSMBRL-CC4.

This command never downloads artifacts and never runs an environment episode.
It validates the frozen world/reward models and the baseline-specific,
calibration-only uncertainty-normalizer bundle before planner construction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping

import numpy as np
import torch

from formal_experiments.training.bootstrap_world_model import BootstrapProbabilisticWorldModel
from formal_experiments.training.response_reward_predictor import ResponseRewardPredictor
from shared.formal_state import BLUE_AGENTS, FORMAL_STATE_DIM
from shared.d27_projection import PROJECTION_SHA256, PROJECTION_VERSION

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = "docs/FINAL_REWARD_MODEL_MANIFEST.json"
DEFAULT_WORLD_MODEL = "outputs/world_model_final_20260917/a4_5b/world_model_absolute.pt"
DEFAULT_REWARD_MODEL = "outputs/world_model_final_20260917/a4_5c/response_reward_predictor.pt"
DEFAULT_NORMALIZER = "outputs/rsmbrl_cc4/calibration/rsmbrl_normalizers_frozen.pt"
EXPECTED_REWARD_PROTOCOL = "final_paper_20260917_v1"
EXPECTED_CALIBRATION_SEEDS = tuple(range(3000, 3008))


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_tensor(errors: list[str], name: str, value, shape: tuple[int, ...]) -> None:
    try:
        tensor = torch.as_tensor(value, dtype=torch.float32)
    except Exception as exc:
        errors.append(f"{name}: cannot parse tensor ({exc})")
        return
    if tuple(tensor.shape) != shape:
        errors.append(f"{name}: expected shape {shape}, got {tuple(tensor.shape)}")
    elif not torch.isfinite(tensor).all():
        errors.append(f"{name}: contains NaN/Inf")
    elif name.endswith(("obs_std", "horizon_std")) and not torch.all(tensor > 0):
        errors.append(f"{name}: must be strictly positive")


def validate_frozen_normalizer(path: Path, *, world_sha256: str,
                               reward_sha256: str | None = None) -> tuple[dict | None, list[str]]:
    errors: list[str] = []
    if not path.is_file():
        return None, [
            f"missing frozen calibration normalizer: {path}",
            "generate it from CC4 calibration seeds 3000..3007 with a baseline-specific "
            "RSMBRL calibration job; save to outputs/rsmbrl_cc4/calibration/"
            "rsmbrl_normalizers_frozen.pt with freeze_after_calibration=true and no EMA updates",
        ]
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return None, [f"normalizer bundle cannot be loaded: {exc}"]
    if not isinstance(payload, Mapping):
        return None, ["normalizer bundle must be a mapping"]
    checks = {
        "format_version": 2,
        "method": "RSMBRL-CC4",
        "source_split": "calibration",
        "horizon": 4,
        "state_dim": 27,
        "n_actions": 4,
        "ensemble_size": 5,
        "freeze_after_calibration": True,
        "online_updates_after_calibration": False,
        "world_model_sha256": world_sha256,
        "projection_version": PROJECTION_VERSION,
        "projection_sha256": PROJECTION_SHA256,
        "beta": 0.1,
        "beta_source": "paper_fixed_no_validation_tuning",
    }
    if reward_sha256 is not None:
        checks["reward_model_sha256"] = reward_sha256
    for key, expected in checks.items():
        if payload.get(key) != expected:
            errors.append(f"normalizer.{key}: expected {expected!r}, got {payload.get(key)!r}")
    seeds = tuple(int(x) for x in payload.get("calibration_seeds", ()))
    if seeds != EXPECTED_CALIBRATION_SEEDS:
        errors.append(f"normalizer.calibration_seeds: expected {EXPECTED_CALIBRATION_SEEDS}, got {seeds}")
    agents = payload.get("agents")
    if not isinstance(agents, Mapping) or set(agents) != set(BLUE_AGENTS):
        errors.append("normalizer.agents must contain exactly all five Blue agents")
    else:
        for agent in BLUE_AGENTS:
            item = agents[agent]
            if not isinstance(item, Mapping):
                errors.append(f"normalizer.{agent}: must be a mapping")
                continue
            _check_tensor(errors, f"normalizer.{agent}.obs_mean", item.get("obs_mean"), (27,))
            _check_tensor(errors, f"normalizer.{agent}.obs_std", item.get("obs_std"), (27,))
            _check_tensor(errors, f"normalizer.{agent}.horizon_std", item.get("horizon_std"), (4,))
    profile = payload.get("planner_profile")
    expected_profile = {"population_size": 200, "num_iterations": 5,
                        "elite_ratio": 0.3, "alpha": 0.1, "beta": 0.1}
    if not isinstance(profile, Mapping) or any(profile.get(k) != v for k, v in expected_profile.items()):
        errors.append("normalizer.planner_profile does not match formal RSMBRL profile")
    coverage = payload.get("coverage")
    if not isinstance(coverage, Mapping) or set(coverage) != set(BLUE_AGENTS):
        errors.append("normalizer.coverage must contain all five Blue agents")
    else:
        for agent in BLUE_AGENTS:
            item = coverage[agent]
            if int(item.get("planner_calls", -1)) != 100:
                errors.append(f"normalizer.coverage.{agent}.planner_calls must be 100")
            if tuple(int(x) for x in item.get("seeds", ())) != EXPECTED_CALIBRATION_SEEDS:
                errors.append(f"normalizer.coverage.{agent}.seeds mismatch")
            if int(item.get("plans_per_call", -1)) != 200:
                errors.append(f"normalizer.coverage.{agent}.plans_per_call must be 200")
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        errors.append("normalizer.provenance must be a mapping")
    else:
        for key in ("states_sha256", "summary_sha256"):
            value = str(provenance.get(key, ""))
            if re.fullmatch(r"[0-9a-f]{64}", value) is None:
                errors.append(f"normalizer.provenance.{key} must be lowercase SHA256")
        count = provenance.get("record_count")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            errors.append("normalizer.provenance.record_count must be a positive integer")
    return dict(payload), errors


def run_preflight(
    *, manifest_path: str | Path = DEFAULT_MANIFEST,
    world_model_path: str | Path = DEFAULT_WORLD_MODEL,
    reward_model_path: str | Path = DEFAULT_REWARD_MODEL,
    normalizer_path: str | Path = DEFAULT_NORMALIZER,
    device: str = "cpu",
) -> dict:
    errors: list[str] = []
    manifest_file, world_file, reward_file, normalizer_file = map(
        resolve, (manifest_path, world_model_path, reward_model_path, normalizer_path)
    )
    for label, path in (("manifest", manifest_file), ("world model", world_file), ("reward model", reward_file)):
        if not path.is_file():
            errors.append(f"missing {label}: {path}")
    if errors:
        return {"status": "FAIL", "eligible": False, "errors": errors}
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "FAIL", "eligible": False, "errors": [f"invalid model manifest: {exc}"]}
    if manifest.get("reward_protocol") != EXPECTED_REWARD_PROTOCOL:
        errors.append("manifest reward_protocol is not final_paper_20260917_v1")
    wm_sha, reward_sha = sha256_file(world_file), sha256_file(reward_file)
    if wm_sha != manifest.get("world_model", {}).get("absolute_checkpoint_sha256"):
        errors.append("world model SHA256 does not match FINAL_REWARD_MODEL_MANIFEST")
    if reward_sha != manifest.get("response_reward_predictor", {}).get("checkpoint_sha256"):
        errors.append("reward predictor SHA256 does not match FINAL_REWARD_MODEL_MANIFEST")
    if manifest.get("world_model", {}).get("selected_target_mode") != "absolute":
        errors.append("manifest does not select the absolute world model")
    if manifest.get("world_model", {}).get("quality_gate_pass") is not True:
        errors.append("world model quality gate is not PASS")
    if manifest.get("response_reward_predictor", {}).get("quality_gate_pass") is not True:
        errors.append("Full-Reward predictor quality gate is not PASS")

    world = reward = None
    try:
        world = BootstrapProbabilisticWorldModel.load_checkpoint(world_file, device=device)
        cfg = world.config
        if (cfg.state_dim, cfg.n_actions, cfg.ensemble_size, cfg.target_mode) != (27, 4, 5, "absolute"):
            errors.append(
                "world model contract mismatch: expected D27/A4/ensemble5/absolute, got "
                f"D{cfg.state_dim}/A{cfg.n_actions}/M{cfg.ensemble_size}/{cfg.target_mode}"
            )
        if len(world.models) != 5:
            errors.append(f"world model member count is {len(world.models)}, expected 5")
        norm = world._require_fitted()
        if np.asarray(norm.mean).shape != (27,) or np.asarray(norm.std).shape != (27,):
            errors.append("world model train normalizer shape is not D27")
        if not np.isfinite(norm.mean).all() or not np.isfinite(norm.std).all() or np.any(norm.std <= 0):
            errors.append("world model train normalizer is non-finite or non-positive")
    except Exception as exc:
        errors.append(f"world model strict load failed: {exc}")
    try:
        reward = ResponseRewardPredictor.load_checkpoint(reward_file, device=device)
        cfg = reward.config
        if (cfg.state_dim, cfg.n_actions) != (27, 4):
            errors.append(f"reward predictor contract mismatch: D{cfg.state_dim}/A{cfg.n_actions}")
        state_norm, reward_norm = reward._require_fitted()
        if np.asarray(state_norm.mean).shape != (27,) or np.asarray(state_norm.std).shape != (27,):
            errors.append("reward predictor state normalizer shape is not D27")
        if not np.isfinite(reward_norm.mean) or not np.isfinite(reward_norm.std) or reward_norm.std <= 0:
            errors.append("reward predictor reward normalizer is invalid")
    except Exception as exc:
        errors.append(f"reward predictor strict load failed: {exc}")

    _, normalizer_errors = validate_frozen_normalizer(
        normalizer_file, world_sha256=wm_sha, reward_sha256=reward_sha
    )
    errors.extend(normalizer_errors)
    return {
        "status": "PASS" if not errors else "FAIL",
        "eligible": not errors,
        "device": str(device),
        "contracts": {"state_dim": 27, "n_actions": 4, "horizon": 4, "ensemble_size": 5,
                      "reward_protocol": EXPECTED_REWARD_PROTOCOL, "normalizer_frozen": True,
                      "projection_version": PROJECTION_VERSION,
                      "projection_sha256": PROJECTION_SHA256, "beta": 0.1},
        "artifacts": {
            "manifest": str(manifest_file), "world_model": str(world_file),
            "world_model_sha256": wm_sha, "reward_model": str(reward_file),
            "reward_model_sha256": reward_sha, "normalizer": str(normalizer_file),
            "normalizer_sha256": sha256_file(normalizer_file) if normalizer_file.is_file() else None,
        },
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--world-model", default=DEFAULT_WORLD_MODEL)
    parser.add_argument("--reward-model", default=DEFAULT_REWARD_MODEL)
    parser.add_argument("--normalizer", default=DEFAULT_NORMALIZER)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--report")
    args = parser.parse_args(argv)
    report = run_preflight(manifest_path=args.manifest, world_model_path=args.world_model,
                           reward_model_path=args.reward_model, normalizer_path=args.normalizer,
                           device=args.device)
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.report:
        path = resolve(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    return 0 if report["eligible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
