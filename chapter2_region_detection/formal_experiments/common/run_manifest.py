"""Schema and immutable manifest validation for formal CC4 v3 results."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

from formal_experiments.evaluation.metrics_v3 import EPISODE_TICKS


PROTOCOL_VERSION = "cc4_v3_20260917"
FINAL_TEST_SEEDS = tuple(range(4000, 4100))
POLICY_SEEDS = (51001, 51002, 51003, 51004, 51005)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

MANIFEST_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "protocol_version",
        "method",
        "repeat_index",
        "training_seed",
        "test_episode_seeds",
        "episode_ticks",
        "code_commit",
        "config_sha256",
        "paper_sha256",
        "upstream_commit",
        "python_version",
        "dependencies",
        "hardware",
        "model_sha256",
        "run_mode",
        "formal_result_eligible",
        "method_slug",
        "git_dirty",
        "git_diff_sha256",
        "method_artifacts",
        "artifact_sha256",
    }
)


def validate_manifest(manifest: Mapping[str, Any], *, formal: bool = True) -> list[str]:
    errors: list[str] = []
    missing = sorted(MANIFEST_REQUIRED_FIELDS - set(manifest))
    if missing:
        errors.append(f"manifest missing fields: {missing}")
        return errors
    if manifest["schema_version"] != 1:
        errors.append("schema_version must be 1")
    if manifest["protocol_version"] != PROTOCOL_VERSION:
        errors.append(f"protocol_version must be {PROTOCOL_VERSION}")
    if not str(manifest["method"]).strip():
        errors.append("method must be non-empty")
    method_slug = manifest["method_slug"]
    formal_names = {"dca_cc4": "DCA-CC4 (adapted)", "rsmbrl_cc4": "RSMBRL-CC4",
                    "uamcts_cc4": "UAMCTS-CC4 (adapted)", "terla_a4": "TERLA-A4",
                    "carl_cc4": "CARL-CC4 (adapted)", "priorrl_ppo_cc4": "PriorRL-PPO-CC4"}
    if method_slug in formal_names and manifest["method"] != formal_names[method_slug]:
        errors.append("method does not match formal method_slug name")
    repeat_index = manifest["repeat_index"]
    if not isinstance(repeat_index, int) or isinstance(repeat_index, bool) or repeat_index not in range(1, 6):
        errors.append("repeat_index must be an integer in 1..5")
    episode_ticks = manifest["episode_ticks"]
    if isinstance(episode_ticks, bool) or not isinstance(episode_ticks, int) or episode_ticks <= 0:
        errors.append("episode_ticks must be a positive integer")
    elif formal and episode_ticks != EPISODE_TICKS:
        errors.append("formal episode_ticks must be 500")
    seeds = manifest["test_episode_seeds"]
    if not isinstance(seeds, list) or any(
        isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds
    ):
        errors.append("test_episode_seeds must be a list of integers")
        seeds_valid = False
    else:
        seeds_valid = True
    if formal and seeds_valid and tuple(seeds) != FINAL_TEST_SEEDS:
        errors.append("formal test_episode_seeds must be exactly 4000..4099 in order")
    run_mode = manifest["run_mode"]
    if run_mode not in ("dev", "formal"):
        errors.append("run_mode must be dev or formal")
    eligible = manifest["formal_result_eligible"]
    if not isinstance(eligible, bool):
        errors.append("formal_result_eligible must be boolean")
    elif formal and (run_mode != "formal" or not eligible):
        errors.append("formal validation requires run_mode=formal and formal_result_eligible=true")
    elif not formal and (run_mode != "dev" or eligible):
        errors.append("dev validation requires run_mode=dev and formal_result_eligible=false")
    training_seed = manifest["training_seed"]
    training_seed_valid = isinstance(training_seed, int) and not isinstance(training_seed, bool)
    if not training_seed_valid:
        errors.append("training_seed must be an integer")
    if formal and training_seed_valid and isinstance(repeat_index, int) and not isinstance(repeat_index, bool) and repeat_index in range(1, 6):
        expected_seed = POLICY_SEEDS[repeat_index - 1]
        if training_seed != expected_seed:
            errors.append(
                f"formal repeat {repeat_index} must use frozen training_seed {expected_seed}"
            )
    for field in ("config_sha256", "paper_sha256"):
        if not SHA256_RE.fullmatch(str(manifest[field])):
            errors.append(f"{field} must be a lowercase SHA256")
    model_hash = manifest["model_sha256"]
    if model_hash is not None and not SHA256_RE.fullmatch(str(model_hash)):
        errors.append("model_sha256 must be null or a lowercase SHA256")
    if not isinstance(manifest["dependencies"], Mapping):
        errors.append("dependencies must be an object")
    elif formal:
        for field in ("python", "torch", "numpy", "CybORG"):
            if field not in manifest["dependencies"]:
                errors.append(f"dependencies missing {field}")
        cyborg = manifest["dependencies"].get("CybORG")
        if not isinstance(cyborg, Mapping) or not SHA256_RE.fullmatch(
            str(cyborg.get("source_sha256", ""))
        ):
            errors.append("dependencies.CybORG.source_sha256 must be a lowercase SHA256")
    if not isinstance(manifest["hardware"], Mapping):
        errors.append("hardware must be an object")
    elif formal:
        for field in ("cpu", "gpu", "selected_device"):
            if field not in manifest["hardware"]:
                errors.append(f"hardware missing {field}")
    if not isinstance(manifest["git_dirty"], bool):
        errors.append("git_dirty must be boolean")
    elif formal and manifest["git_dirty"]:
        errors.append("formal manifest cannot use a dirty git snapshot")
    if not SHA256_RE.fullmatch(str(manifest["git_diff_sha256"])):
        errors.append("git_diff_sha256 must be a lowercase SHA256")
    artifacts = manifest["method_artifacts"]
    if not isinstance(artifacts, Mapping):
        errors.append("method_artifacts must be an object")
    elif method_slug == "dca_cc4":
        for field in ("policy_spec_sha256", "provenance_audit_sha256"):
            if not SHA256_RE.fullmatch(str(artifacts.get(field, ""))):
                errors.append(f"method_artifacts.{field} must be a lowercase SHA256")
    elif method_slug == "rsmbrl_cc4":
        if manifest["upstream_commit"] != "9f97859594f2b0547e01193a8758936090b0b2ec":
            errors.append("RSMBRL upstream_commit mismatch")
        for field in ("world_model_sha256", "reward_model_sha256", "normalizer_sha256",
                      "normalizer_sidecar_sha256"):
            if not SHA256_RE.fullmatch(str(artifacts.get(field, ""))):
                errors.append(f"method_artifacts.{field} must be a lowercase SHA256")
    elif method_slug == "uamcts_cc4":
        for field in ("policy_spec_sha256", "world_model_sha256", "reward_model_sha256",
                      "progress_model_sha256", "prototype_prior_sha256",
                      "prior_entropy_sha256", "normalizer_sha256"):
            if not SHA256_RE.fullmatch(str(artifacts.get(field, ""))):
                errors.append(f"method_artifacts.{field} must be a lowercase SHA256")
    elif method_slug == "terla_a4":
        for field in ("policy_spec_sha256", "checkpoint_sha256", "training_manifest_sha256"):
            if not SHA256_RE.fullmatch(str(artifacts.get(field, ""))):
                errors.append(f"method_artifacts.{field} must be a lowercase SHA256")
    elif method_slug == "carl_cc4":
        for field in ("policy_spec_sha256", "checkpoint_sha256", "validation_selection_sha256"):
            if not SHA256_RE.fullmatch(str(artifacts.get(field, ""))):
                errors.append(f"method_artifacts.{field} must be a lowercase SHA256")
    elif method_slug == "priorrl_ppo_cc4":
        for field in ("policy_spec_sha256", "checkpoint_sha256", "training_manifest_sha256",
                      "alpha_selection_sha256", "prototype_prior_sha256",
                      "prototype_coverage_sha256", "prototype_provenance_sha256"):
            if not SHA256_RE.fullmatch(str(artifacts.get(field, ""))):
                errors.append(f"method_artifacts.{field} must be a lowercase SHA256")
    bound_artifacts = manifest["artifact_sha256"]
    if not isinstance(bound_artifacts, Mapping) or not bound_artifacts:
        errors.append("artifact_sha256 must be a non-empty path-to-SHA256 object")
    else:
        for relative, digest in bound_artifacts.items():
            if not isinstance(relative, str) or not relative.strip():
                errors.append("artifact_sha256 keys must be non-empty relative paths")
            if not SHA256_RE.fullmatch(str(digest)):
                errors.append(f"artifact_sha256[{relative!r}] must be a lowercase SHA256")
        if formal:
            required = {"config.resolved.yaml", "episodes.jsonl", "metrics.json"}
            missing_artifacts = sorted(required - set(bound_artifacts))
            if missing_artifacts:
                errors.append(f"artifact_sha256 missing formal artifacts: {missing_artifacts}")
            if not ({"decisions.jsonl", "decisions.jsonl.zst"} & set(bound_artifacts)):
                errors.append("artifact_sha256 needs decisions.jsonl or decisions.jsonl.zst")
            if not ({"policy_spec.json", "checkpoint.pt"} & set(bound_artifacts)):
                errors.append("artifact_sha256 needs policy_spec.json or checkpoint.pt")
            if method_slug == "terla_a4" and "training_manifest.json" not in bound_artifacts:
                errors.append("artifact_sha256 needs TERLA training_manifest.json")
            if method_slug == "carl_cc4" and "validation_selection.json" not in bound_artifacts:
                errors.append("artifact_sha256 needs CARL validation_selection.json")
        config_digest = bound_artifacts.get("config.resolved.yaml")
        if config_digest is not None and config_digest != manifest["config_sha256"]:
            errors.append("config_sha256 must match artifact_sha256 for config.resolved.yaml")
    return errors


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    import json

    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL line {line_number} must be an object")
            rows.append(value)
    return rows


def validate_episode_schema(
    episode: Mapping[str, Any], *, expected_ticks: int = EPISODE_TICKS
) -> list[str]:
    errors: list[str] = []
    required = {
        "schema_version",
        "protocol_version",
        "episode_seed",
        "tick_count",
        "episode_end_tick",
        "tick_team_rewards",
        "operation_failure_events",
        "recovery_actions",
        "incidents",
    }
    missing = sorted(required - set(episode))
    if missing:
        return [f"episode missing fields: {missing}"]
    if episode["schema_version"] != 1:
        errors.append("episode schema_version must be 1")
    if episode["protocol_version"] != PROTOCOL_VERSION:
        errors.append(f"episode protocol_version must be {PROTOCOL_VERSION}")
    if isinstance(expected_ticks, bool) or not isinstance(expected_ticks, int) or expected_ticks <= 0:
        return ["expected_ticks must be a positive integer"]
    for field in ("tick_count", "episode_end_tick"):
        value = episode[field]
        if isinstance(value, bool) or not isinstance(value, int):
            errors.append(f"{field} must be an integer")
        elif value != expected_ticks:
            errors.append(f"{field} must be {expected_ticks}")
    rewards = episode["tick_team_rewards"]
    if not isinstance(rewards, list) or len(rewards) != expected_ticks:
        errors.append(f"tick_team_rewards must contain exactly {expected_ticks} records")
    for field in ("operation_failure_events", "recovery_actions", "incidents"):
        if not isinstance(episode[field], list):
            errors.append(f"{field} must be a list")
    if isinstance(episode["operation_failure_events"], list):
        for index, event in enumerate(episode["operation_failure_events"]):
            if not isinstance(event, Mapping):
                errors.append(f"operation_failure_events[{index}] must be an object")
                continue
            located = bool(str(event.get("agent_name", "")).strip()) and bool(
                str(event.get("host", "")).strip()
            )
            episode_scoped = event.get("scope") == "all_blue"
            if not (located or episode_scoped):
                errors.append(
                    f"operation_failure_events[{index}] needs agent_name+host or scope=all_blue"
                )
    return errors
