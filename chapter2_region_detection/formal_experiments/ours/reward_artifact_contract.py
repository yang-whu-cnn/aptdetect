"""Fail-closed validation for frozen Table-3 reward artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from formal_experiments.ours.reward_ablation import RewardMode


TABLE3_REWARD_MANIFEST_FORMAT = "table3_reward_ablation_v2"
DELAY_ONLY_MANIFEST_SHA256 = "fb9fdd64a6f576d0ffb444daacce07e891ba5505da66381b57acb52631c71eaf"
DELAY_ONLY_CHECKPOINT_SHA256 = "192efbe6512fbd83a746d86d759ebe5e68b1437ee78e570747a119d74b0e683f"
FAIL_ONLY_MANIFEST_FORMAT = TABLE3_REWARD_MANIFEST_FORMAT
FAIL_ONLY_MANIFEST_SHA256 = "8295b5177e900f032f7fed142b07cb7773607911d77d73be01b414aebf61f5cc"
FAIL_ONLY_CHECKPOINT_SHA256 = "8800a65d4cd4f540ca6d5fa5bcc0f7e87a913403c4cded5beed931e91bb0b3b2"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_equal(errors: list[str], label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        errors.append(f"{label}: expected {expected!r}, got {actual!r}")


def validate_frozen_reward_artifact(
    manifest_path: str | Path,
    *,
    mode: RewardMode,
    project_root: str | Path,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    """Load one frozen reward artifact and enforce its approved training contract."""
    if mode == RewardMode.FULL_REWARD:
        raise ValueError("Full-Reward uses the shared frozen model preflight")

    path = Path(manifest_path)
    if not path.is_file():
        raise RuntimeError(f"BLOCKED: missing {mode.value} frozen_manifest.json")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"BLOCKED: invalid {mode.value} frozen manifest: {exc}") from exc

    errors: list[str] = []
    _require_equal(errors, "mode", payload.get("mode"), mode.value)
    _require_equal(errors, "frozen", payload.get("frozen"), True)
    _require_equal(errors, "quality_gate.pass", payload.get("quality_gate", {}).get("pass"), True)
    _require_equal(
        errors,
        "reward_model_eligible_for_policy_training",
        payload.get("reward_model_eligible_for_policy_training"),
        True,
    )
    _require_equal(errors, "test_seeds_used", payload.get("test_seeds_used"), False)

    if mode == RewardMode.FAIL_ONLY:
        _require_equal(errors, "format", payload.get("format"), FAIL_ONLY_MANIFEST_FORMAT)
        _require_equal(
            errors,
            "formal_contract_architecture_match",
            payload.get("formal_contract_architecture_match"),
            True,
        )
        _require_equal(
            errors,
            "formal_contract_training_match",
            payload.get("formal_contract_training_match"),
            True,
        )
        _require_equal(
            errors,
            "diagnostic_hurdle_used_for_formal_artifact",
            payload.get("diagnostic_hurdle_used_for_formal_artifact"),
            False,
        )
        component = payload.get("architecture", {}).get("component_config", {})
        training_loss = payload.get("training_loss", {})
        _require_equal(errors, "component_config.loss", component.get("loss"), "smooth_l1")
        _require_equal(errors, "component_config.smooth_l1_beta", component.get("smooth_l1_beta"), 1.0)
        _require_equal(errors, "training_loss.name", training_loss.get("name"), "smooth_l1")
        _require_equal(errors, "training_loss.beta", training_loss.get("beta"), 1.0)
        _require_equal(
            errors,
            "training_loss.target_space",
            training_loss.get("target_space"),
            "standardized_reward_label",
        )
        _require_equal(errors, "training_loss.reduction", training_loss.get("reduction"), "mean")
        _require_equal(errors, "training_loss.sample_weighting", training_loss.get("sample_weighting"), "none")
        _require_equal(errors, "training_loss.resampling", training_loss.get("resampling"), "none")
        # The approved unweighted/non-resampled protocol is numerically
        # positive_weight=1 and oversample_factor=1.  Legacy weighted/resampled
        # manifests may expose those fields directly, so reject non-unit values.
        _require_equal(errors, "training_loss.positive_weight", training_loss.get("positive_weight", 1), 1)
        _require_equal(errors, "training_loss.oversample_factor", training_loss.get("oversample_factor", 1), 1)
        _require_equal(errors, "checkpoint_sha256", payload.get("checkpoint_sha256"), FAIL_ONLY_CHECKPOINT_SHA256)
    elif mode == RewardMode.DELAY_ONLY:
        _require_equal(errors, "format", payload.get("format"), TABLE3_REWARD_MANIFEST_FORMAT)
        _require_equal(errors, "formal_contract_architecture_match",
                       payload.get("formal_contract_architecture_match"), True)
        _require_equal(errors, "formal_contract_training_match",
                       payload.get("formal_contract_training_match"), True)
        _require_equal(errors, "diagnostic_hurdle_used_for_formal_artifact",
                       payload.get("diagnostic_hurdle_used_for_formal_artifact"), False)
        component = payload.get("architecture", {}).get("component_config", {})
        training_loss = payload.get("training_loss", {})
        _require_equal(errors, "component_config.loss", component.get("loss"), "mse")
        _require_equal(errors, "training_loss.name", training_loss.get("name"), "mse")
        _require_equal(errors, "training_loss.target_space", training_loss.get("target_space"),
                       "standardized_reward_label")
        _require_equal(errors, "training_loss.reduction", training_loss.get("reduction"), "mean")
        _require_equal(errors, "training_loss.sample_weighting", training_loss.get("sample_weighting"), "none")
        _require_equal(errors, "training_loss.resampling", training_loss.get("resampling"), "none")
        _require_equal(errors, "checkpoint_sha256", payload.get("checkpoint_sha256"), DELAY_ONLY_CHECKPOINT_SHA256)

    checkpoint = Path(str(payload.get("checkpoint", "")))
    if not checkpoint.is_absolute():
        checkpoint = Path(project_root) / checkpoint
    if not checkpoint.is_file():
        errors.append(f"checkpoint missing: {checkpoint}")
    else:
        actual_checkpoint_sha = sha256_file(checkpoint)
        if actual_checkpoint_sha != payload.get("checkpoint_sha256"):
            errors.append(
                "checkpoint integrity: expected "
                f"{payload.get('checkpoint_sha256')!r}, got {actual_checkpoint_sha!r}"
            )

    manifest_sha = sha256_file(path)
    if mode == RewardMode.FAIL_ONLY and manifest_sha != FAIL_ONLY_MANIFEST_SHA256:
        errors.append(
            f"manifest_sha256: expected {FAIL_ONLY_MANIFEST_SHA256!r}, got {manifest_sha!r}"
        )
    if mode == RewardMode.DELAY_ONLY and manifest_sha != DELAY_ONLY_MANIFEST_SHA256:
        errors.append(
            f"manifest_sha256: expected {DELAY_ONLY_MANIFEST_SHA256!r}, got {manifest_sha!r}"
        )
    if errors:
        raise RuntimeError(f"BLOCKED: {mode.value} frozen reward contract FAIL: " + "; ".join(errors))

    audit = {
        "pass": True,
        "mode": mode.value,
        "manifest_sha256": manifest_sha,
        "checkpoint_sha256": payload["checkpoint_sha256"],
        "approved_protocol": (
            "unweighted_smooth_l1_beta_1_no_resampling"
            if mode == RewardMode.FAIL_ONLY
            else "frozen_delay_only_v1"
        ),
    }
    return payload, checkpoint, audit
