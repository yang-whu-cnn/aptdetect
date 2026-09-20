"""Resumable repeat orchestrator for Table-1 CC4 methods."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from typing import Any, Callable

from formal_experiments.common.run_manifest import FINAL_TEST_SEEDS, POLICY_SEEDS, PROTOCOL_VERSION
from formal_experiments.evaluation.metrics_v3 import aggregate_repeat
from formal_experiments.evaluation.run_method_episode import METHODS, run_method_episode
from formal_experiments.evaluation.validate_formal_run import validate_run_directory


REPO_ROOT = Path(__file__).resolve().parents[3]
METHOD_NAMES = {"dca_cc4": "DCA-CC4 (adapted)", "rsmbrl_cc4": "RSMBRL-CC4",
                "uamcts_cc4": "UAMCTS-CC4 (adapted)", "terla_a4": "TERLA-A4",
                "carl_cc4": "CARL-CC4 (adapted)", "priorrl_ppo_cc4": "PriorRL-PPO-CC4"}
PAPER_FILES = {"dca_cc4": "DCA.pdf", "rsmbrl_cc4": "RSMBRL.pdf",
               "uamcts_cc4": "UAMCTS.pdf", "terla_a4": "TERLA.pdf", "carl_cc4": "CARL.pdf",
               "priorrl_ppo_cc4": "PriorRL.pdf"}
EXPECTED_PAPER_HASHES = {
    "dca_cc4": "173fa04a1675e98ce636c2a8b2eda9eae5e3c9e2347f5f19cb4aafe6708e3cb1",
    "rsmbrl_cc4": "df594e5b3d2bd36e06d94d96e3cde0c549fa1894a9fe5d26ecd6c4552c91d4d7",
    "uamcts_cc4": "5dffdc568bf6e7fa04a430c44dcac9d293b588c4a2b5211cc5f14324f10af145",
    "terla_a4": "e2c53cc19c3647029870d9bd9438f9c20d50ab79e728374b8461852e729800ea",
    "carl_cc4": "42e8ef7d912007a2454002300852a54d1dfffad2d9b1099958f7c8d6a84cfae1",
    "priorrl_ppo_cc4": "40ab37985422db1a439a22757f30a368b04363b8a9a9b3d59dec3488085e1c71",
}
RSMBRL_UPSTREAM_COMMIT = "9f97859594f2b0547e01193a8758936090b0b2ec"


def git_state() -> dict[str, Any]:
    status = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                            capture_output=True, text=True, check=True, timeout=10).stdout
    diff = subprocess.run(["git", "diff", "--binary", "HEAD"], cwd=REPO_ROOT,
                          capture_output=True, check=True, timeout=20).stdout
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                            capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    return {"code_commit": commit, "git_dirty": bool(status.strip()),
            "git_status_porcelain": status.splitlines(),
            "git_diff_sha256": hashlib.sha256(diff).hexdigest()}


def dependency_and_hardware(device: str) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy
    import torch
    try:
        import CybORG
        cyborg_file = Path(CybORG.__file__).resolve()
        cyborg = {"version": getattr(CybORG, "__version__", None),
                  "source_file": str(cyborg_file), "source_sha256": sha256_file(cyborg_file)}
    except Exception as exc:
        cyborg = {"error": repr(exc), "source_sha256": None}
    dependencies = {"python": platform.python_version(), "torch": str(torch.__version__),
                    "torch_cuda": torch.version.cuda, "numpy": str(numpy.__version__), "CybORG": cyborg}
    gpu = None
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip().splitlines()
    except Exception:
        gpu = []
    hardware = {"cpu": platform.processor() or platform.machine(), "gpu": gpu,
                "selected_device": str(device), "torch_cuda_available": bool(torch.cuda.is_available())}
    return dependencies, hardware


def artifact_provenance(method: str, output: Path) -> tuple[dict[str, Any], str]:
    policy_sha = sha256_file(output / "policy_spec.json")
    if method == "dca_cc4":
        audit = REPO_ROOT / "chapter2_region_detection" / "baselines" / "dca_cc4" / "PROVENANCE_AUDIT.md"
        artifacts = {"policy_spec_sha256": policy_sha, "provenance_audit_sha256": sha256_file(audit)}
        return artifacts, policy_sha
    if method == "uamcts_cc4":
        from baselines.uamcts_cc4.preflight import run_preflight
        base = REPO_ROOT / "chapter2_region_detection" / "outputs"
        paths = {
            "world_model_sha256": base / "world_model_final_20260917/a4_5b/world_model_absolute.pt",
            "reward_model_sha256": base / "world_model_final_20260917/a4_5c/response_reward_predictor.pt",
            "progress_model_sha256": base / "uamcts_cc4/progress/progress_ensemble_train_only.pt",
            "progress_sidecar_sha256": base / "uamcts_cc4/progress/progress_ensemble_train_only.sidecar.json",
            "prototype_prior_sha256": base / "priorrl_cc4/prototypes/frozen_prototypes.json",
            "prototype_coverage_sha256": base / "priorrl_cc4/prototypes/frozen_prototype_coverage.json",
            "prototype_provenance_sha256": base / "priorrl_cc4/prototypes/frozen_prototype_provenance.json",
            "prior_entropy_sha256": base / "uamcts_cc4/calibration/validation_prior_entropy.json",
            "prior_entropy_sidecar_sha256": base / "uamcts_cc4/calibration/validation_prior_entropy.sidecar.json",
            "normalizer_sha256": base / "uamcts_cc4/calibration/uamcts_uncertainty_normalizers_frozen.pt",
            "normalizer_sidecar_sha256": base / "uamcts_cc4/calibration/uamcts_uncertainty_normalizers_frozen.sidecar.json",
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise RuntimeError(f"UAMCTS frozen artifact gate failed: missing {missing}")
        gate = run_preflight(require_sidecars=True)
        if not gate.get("eligible"):
            raise RuntimeError("UAMCTS frozen artifact gate failed: " + "; ".join(gate.get("errors", [])))
        artifacts = {"policy_spec_sha256": policy_sha,
                     **{key: sha256_file(path) for key, path in paths.items()}}
        return artifacts, artifacts["world_model_sha256"]
    if method == "priorrl_ppo_cc4":
        policy_seed = int(json.loads((output / "policy_spec.json").read_text(encoding="utf-8"))["policy_seed"])
        repeat_index = POLICY_SEEDS.index(policy_seed) + 1
        training = (REPO_ROOT / "chapter2_region_detection/outputs/formal_v3/training/priorrl_ppo_cc4"
                    / f"repeat_{repeat_index:02d}")
        paths = {"checkpoint_sha256": training / "checkpoint.pt",
                 "training_manifest_sha256": training / "training_manifest.json",
                 "alpha_selection_sha256": training / "alpha_selection.json",
                 "prototype_prior_sha256": REPO_ROOT / "chapter2_region_detection/outputs/priorrl_cc4/prototypes/frozen_prototypes.json",
                 "prototype_coverage_sha256": REPO_ROOT / "chapter2_region_detection/outputs/priorrl_cc4/prototypes/frozen_prototype_coverage.json",
                 "prototype_provenance_sha256": REPO_ROOT / "chapter2_region_detection/outputs/priorrl_cc4/prototypes/frozen_prototype_provenance.json"}
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise RuntimeError(f"PriorRL frozen artifact gate failed: missing {missing}")
        artifacts = {"policy_spec_sha256": policy_sha,
                     **{key: sha256_file(path) for key, path in paths.items()}}
        return artifacts, artifacts["checkpoint_sha256"]
    if method in ("terla_a4", "carl_cc4"):
        checkpoint = output / "checkpoint.pt"
        if not checkpoint.is_file():
            raise RuntimeError(f"missing copied frozen training checkpoint: {checkpoint}")
        checkpoint_sha = sha256_file(checkpoint)
        support_name = "training_manifest.json" if method == "terla_a4" else "validation_selection.json"
        support_key = ("training_manifest_sha256" if method == "terla_a4"
                       else "validation_selection_sha256")
        support = output / support_name
        if not support.is_file():
            raise RuntimeError(f"missing copied frozen training provenance: {support}")
        return {"policy_spec_sha256": policy_sha, "checkpoint_sha256": checkpoint_sha,
                support_key: sha256_file(support)}, checkpoint_sha
    from baselines.rsmbrl_cc4.artifact_preflight import (
        DEFAULT_NORMALIZER, DEFAULT_NORMALIZER_SIDECAR, DEFAULT_REWARD_MODEL,
        DEFAULT_WORLD_MODEL, resolve,
    )
    sidecar = resolve(DEFAULT_NORMALIZER_SIDECAR)
    if not sidecar.is_file():
        raise RuntimeError(f"RSMBRL frozen artifact gate failed: missing {sidecar}")
    artifacts = {"policy_spec_sha256": policy_sha,
                 "world_model_sha256": sha256_file(resolve(DEFAULT_WORLD_MODEL)),
                 "reward_model_sha256": sha256_file(resolve(DEFAULT_REWARD_MODEL)),
                 "normalizer_sha256": sha256_file(resolve(DEFAULT_NORMALIZER)),
                 "normalizer_sidecar_sha256": sha256_file(sidecar)}
    return artifacts, artifacts["world_model_sha256"]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_repeat_request(*, run_mode: str, repeat_index: int, policy_seed: int,
                            episode_seeds: list[int], ticks: int) -> None:
    if run_mode == "formal":
        if repeat_index not in range(1, 6) or policy_seed != POLICY_SEEDS[repeat_index - 1]:
            raise ValueError("formal repeat_index/policy_seed mismatch")
        if episode_seeds != list(FINAL_TEST_SEEDS) or ticks != 500:
            raise ValueError("formal repeat requires seeds 4000..4099 and 500 ticks")
    elif run_mode == "dev":
        if not episode_seeds or ticks <= 0:
            raise ValueError("dev repeat needs seeds and positive ticks")
    else:
        raise ValueError("run_mode must be dev or formal")
    if len(episode_seeds) != len(set(episode_seeds)):
        raise ValueError("episode seeds must be unique")


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _immutable_json(path: Path, value: Any, *, label: str = "artifact") -> str:
    """Create an immutable JSON artifact and return its content hash."""
    encoded = _canonical_json_bytes(value)
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"immutable {label} overwrite refused: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def _parameter_identity_hash(artifacts: dict[str, str]) -> str:
    return hashlib.sha256(_canonical_json_bytes(artifacts)).hexdigest()


def _uamcts_preflight() -> tuple[dict, dict[str, str]]:
    from baselines.uamcts_cc4.preflight import run_preflight

    report = run_preflight(require_sidecars=True)
    if not report.get("eligible"):
        raise RuntimeError("UAMCTS artifact preflight failed: " + "; ".join(report.get("errors", [])))
    frozen = report.get("artifacts", {}).get("frozen_artifact_sha256")
    if not isinstance(frozen, dict) or not frozen:
        raise RuntimeError("UAMCTS preflight did not return frozen_artifact_sha256")
    return report, {str(key): str(value) for key, value in frozen.items()
                    if key != "frozen_artifact_sha256"}


def _run_repeat_generic(*, method: str, run_mode: str, repeat_index: int,
                        policy_seed: int, episode_seeds: list[int], ticks: int,
                        output: Path, device: str, resume: bool,
                        episode_runner: Callable[..., tuple[dict, list[dict]]],
                        git: dict[str, Any]) -> dict:
    """Run the original generic repeat path without UAMCTS metadata."""
    output.mkdir(parents=True, exist_ok=True)
    parts = output / "episode_parts"; parts.mkdir(exist_ok=True)
    index_path = output / "resume_index.json"
    identity = {"method": method, "run_mode": run_mode, "repeat_index": repeat_index,
                "policy_seed": policy_seed, "episode_seeds": episode_seeds, "ticks": ticks}
    index = {"identity": identity, "episodes": {}}
    if resume:
        if not index_path.is_file():
            raise RuntimeError("resume requested without resume_index.json")
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if index.get("identity") != identity:
            raise RuntimeError("resume identity mismatch")
    elif index_path.exists():
        raise RuntimeError("output already contains resume state; use --resume or a fresh directory")

    all_episodes: list[dict] = []; all_decisions: list[dict] = []
    for episode_seed in episode_seeds:
        episode_path = parts / f"episode_{episode_seed}.json"
        decision_path = parts / f"decisions_{episode_seed}.jsonl"
        record = index["episodes"].get(str(episode_seed))
        if resume and record is not None:
            if (not episode_path.is_file() or not decision_path.is_file()
                    or sha256_file(episode_path) != record.get("episode_sha256")
                    or sha256_file(decision_path) != record.get("decisions_sha256")):
                raise RuntimeError(f"resume artifact hash mismatch for seed {episode_seed}")
            episode = json.loads(episode_path.read_text(encoding="utf-8"))
            decisions = [json.loads(line) for line in decision_path.read_text(encoding="utf-8").splitlines() if line]
        else:
            episode, decisions = episode_runner(
                method=method, seed=episode_seed, ticks=ticks, run_mode=run_mode,
                device=device, policy_seed=policy_seed,
            )
            _write_json(episode_path, episode)
            decision_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in decisions), encoding="utf-8")
            index["episodes"][str(episode_seed)] = {
                "episode_sha256": sha256_file(episode_path),
                "decisions_sha256": sha256_file(decision_path),
            }
            _write_json(index_path, index)
        if (episode.get("episode_seed") != episode_seed or episode.get("method") != method
                or episode.get("policy_seed") != policy_seed):
            raise RuntimeError(f"episode identity mismatch for seed {episode_seed}")
        all_episodes.append(episode); all_decisions.extend(decisions)

    episodes_path = output / "episodes.jsonl"
    decisions_path = output / "decisions.jsonl"
    episodes_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in all_episodes), encoding="utf-8")
    decisions_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in all_decisions), encoding="utf-8")
    metrics = aggregate_repeat(all_episodes, expected_ticks=ticks).to_dict()
    _write_json(output / "metrics.json", metrics)
    config_payload = json.dumps(identity, indent=2, sort_keys=True) + "\n"
    (output / "config.resolved.yaml").write_text(config_payload, encoding="utf-8")
    policy_spec = {"method": method, "policy_seed": policy_seed}
    if method in ("terla_a4", "carl_cc4", "priorrl_ppo_cc4"):
        if method == "priorrl_ppo_cc4":
            source_checkpoint = (REPO_ROOT / "chapter2_region_detection" / "outputs" /
                "formal_v3" / "training" / method / f"repeat_{repeat_index:02d}" / "checkpoint.pt")
        else:
            source_checkpoint = (REPO_ROOT / "chapter2_region_detection" / "outputs" /
                                 "formal_v3" / "training" / method / f"policy_{policy_seed}.pt")
        if not source_checkpoint.is_file():
            raise RuntimeError(f"missing frozen training checkpoint: {source_checkpoint}")
        checkpoint_bytes = source_checkpoint.read_bytes()
        (output / "checkpoint.pt").write_bytes(checkpoint_bytes)
        policy_spec["checkpoint_sha256"] = hashlib.sha256(checkpoint_bytes).hexdigest()
        if method == "terla_a4":
            source_support = source_checkpoint.with_name(f"policy_{policy_seed}.training.json")
            target_support = output / "training_manifest.json"
            support_label = "training_manifest_sha256"
        elif method == "carl_cc4":
            source_support = source_checkpoint.with_suffix(".selection.json")
            target_support = output / "validation_selection.json"
            support_label = "validation_selection_sha256"
        else:
            source_support = target_support = None
            support_label = None
        if source_support is not None:
            if not source_support.is_file():
                raise RuntimeError(f"missing frozen training provenance: {source_support}")
            support_bytes = source_support.read_bytes()
            target_support.write_bytes(support_bytes)
            policy_spec[support_label] = hashlib.sha256(support_bytes).hexdigest()
    _write_json(output / "policy_spec.json", policy_spec)
    (output / "stdout.log").touch()
    dependencies, hardware = dependency_and_hardware(device)
    method_artifacts, model_sha = artifact_provenance(method, output)
    paper_path = REPO_ROOT / PAPER_FILES[method]
    if not paper_path.is_file():
        raise RuntimeError(f"missing source paper: {paper_path}")
    paper_sha = sha256_file(paper_path)
    if paper_sha != EXPECTED_PAPER_HASHES[method]:
        raise RuntimeError(f"source paper SHA256 mismatch for {paper_path.name}")
    manifest = {
        "schema_version": 1, "protocol_version": PROTOCOL_VERSION,
        "method": METHOD_NAMES[method], "method_slug": method,
        "repeat_index": repeat_index, "training_seed": policy_seed,
        "test_episode_seeds": episode_seeds, "episode_ticks": ticks,
        "code_commit": git["code_commit"], "git_dirty": git["git_dirty"],
        "git_diff_sha256": git["git_diff_sha256"],
        "config_sha256": sha256_file(output / "config.resolved.yaml"),
        "paper_sha256": paper_sha,
        "upstream_commit": RSMBRL_UPSTREAM_COMMIT if method == "rsmbrl_cc4" else None,
        "python_version": platform.python_version(), "dependencies": dependencies, "hardware": hardware,
        "model_sha256": model_sha, "method_artifacts": method_artifacts,
        "run_mode": run_mode, "formal_result_eligible": run_mode == "formal",
        "table_id": "table1", "row_id": method,
        "component_variant": method,
        "reward_mode": ({"terla_a4": "terla_cyber_reward", "carl_cc4": "caics_reward"}.get(
            method, "full_reward" if method in ("rsmbrl_cc4", "uamcts_cc4", "priorrl_ppo_cc4") else "not_applicable")),
        "reward_semantics": ({"terla_a4": "negative_red_sessions_plus_service_unreliability_ot",
                              "carl_cc4": "standard_caics_cc4_mapping"}.get(
            method, "cc4_v3_full_reward" if method in ("rsmbrl_cc4", "uamcts_cc4", "priorrl_ppo_cc4")
            else "fixed_response_mapping_no_learning_reward")),
        # eligibility_report.json is deliberately excluded: it is the signed-by-
        # validation verdict over these inputs, and including itself would create
        # a recursive hash dependency.
        "artifact_sha256": {
            "config.resolved.yaml": sha256_file(output / "config.resolved.yaml"),
            "policy_spec.json": sha256_file(output / "policy_spec.json"),
            "episodes.jsonl": sha256_file(episodes_path),
            "decisions.jsonl": sha256_file(decisions_path),
            "metrics.json": sha256_file(output / "metrics.json"),
            **({"checkpoint.pt": sha256_file(output / "checkpoint.pt")}
               if method in ("terla_a4", "carl_cc4", "priorrl_ppo_cc4") else {}),
            **({"training_manifest.json": sha256_file(output / "training_manifest.json")}
               if method == "terla_a4" else {}),
            **({"validation_selection.json": sha256_file(output / "validation_selection.json")}
               if method == "carl_cc4" else {}),
        },
    }
    _write_json(output / "manifest.json", manifest)
    validation = validate_run_directory(output, formal=run_mode == "formal")
    _write_json(output / "validation.json", validation)
    if not validation["passed"]:
        raise RuntimeError(f"repeat validation failed: {validation['errors']}")
    return {"manifest": manifest, "metrics": metrics, "validation": validation}


def _run_repeat_uamcts(*, run_mode: str, repeat_index: int, policy_seed: int,
                       episode_seeds: list[int], ticks: int, output: Path,
                       device: str, resume: bool,
                       episode_runner: Callable[..., tuple[dict, list[dict]]],
                       git: dict[str, Any], uamcts_gate: dict,
                       preflight_artifacts: dict[str, str]) -> dict:
    """Run UAMCTS with immutable pre-episode trust-chain metadata."""
    episode_seeds = list(episode_seeds)
    if resume and not output.is_dir():
        raise RuntimeError("resume requested without an existing UAMCTS output directory")
    output.mkdir(parents=True, exist_ok=True)
    index_path = output / "resume_index.json"
    if not resume and index_path.exists():
        raise RuntimeError("output already contains resume state; use --resume or a fresh directory")
    metadata_paths = {
        "config": output / "config.resolved.yaml",
        "policy": output / "policy_spec.json",
        "snapshot": output / "preflight_snapshot.json",
    }
    if resume and any(not path.is_file() for path in metadata_paths.values()):
        raise RuntimeError("resume requested without immutable UAMCTS metadata")
    parts = output / "episode_parts"; parts.mkdir(exist_ok=True)

    policy_payload = {"method": "uamcts_cc4", "policy_seed": policy_seed}
    policy_sha = _immutable_json(metadata_paths["policy"], policy_payload, label="UAMCTS policy spec")
    frozen_artifacts = dict(preflight_artifacts)
    frozen_artifacts["policy_spec_sha256"] = policy_sha
    base_identity = {
        "method": "uamcts_cc4", "run_mode": run_mode, "repeat_index": repeat_index,
        "policy_seed": policy_seed, "episode_seeds": episode_seeds, "ticks": ticks,
        "code_commit": git["code_commit"], "git_diff_sha256": git["git_diff_sha256"],
        "canonical_output_dir": str(output.resolve()),
    }
    config_payload = {
        **base_identity,
        "policy_spec_sha256": policy_sha,
        "frozen_artifact_sha256": frozen_artifacts,
    }
    config_sha = _immutable_json(metadata_paths["config"], config_payload, label="UAMCTS config")
    snapshot_payload = {
        "schema": "uamcts_formal_preflight_snapshot_v1",
        **base_identity,
        "config_sha256": config_sha,
        "policy_spec_sha256": policy_sha,
        "frozen_artifact_sha256": frozen_artifacts,
        "provider_calls": 0, "test_seeds_used": False,
    }
    snapshot_sha = _immutable_json(
        metadata_paths["snapshot"], snapshot_payload, label="UAMCTS preflight snapshot"
    )
    identity = {
        **base_identity,
        "config_sha256": config_sha,
        "policy_spec_sha256": policy_sha,
        "preflight_snapshot_sha256": snapshot_sha,
        "frozen_artifact_sha256": frozen_artifacts,
    }
    if resume:
        if not index_path.is_file():
            raise RuntimeError("resume requested without resume_index.json")
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if index.get("identity") != identity:
            raise RuntimeError("UAMCTS resume identity drift (commit, snapshot, config, or artifact)")
    else:
        index = {"identity": identity, "episodes": {}}
        _write_json(index_path, index)

    all_episodes: list[dict] = []; all_decisions: list[dict] = []
    for episode_seed in episode_seeds:
        episode_path = parts / f"episode_{episode_seed}.json"
        decision_path = parts / f"decisions_{episode_seed}.jsonl"
        record = index["episodes"].get(str(episode_seed))
        if resume and record is not None:
            if (not episode_path.is_file() or not decision_path.is_file()
                    or sha256_file(episode_path) != record.get("episode_sha256")
                    or sha256_file(decision_path) != record.get("decisions_sha256")):
                raise RuntimeError(f"resume artifact hash mismatch for seed {episode_seed}")
            episode = json.loads(episode_path.read_text(encoding="utf-8"))
            decisions = [json.loads(line) for line in decision_path.read_text(encoding="utf-8").splitlines() if line]
        else:
            episode, decisions = episode_runner(
                method="uamcts_cc4", seed=episode_seed, ticks=ticks, run_mode=run_mode,
                device=device, policy_seed=policy_seed,
            )
            _write_json(episode_path, episode)
            decision_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in decisions), encoding="utf-8")
            index["episodes"][str(episode_seed)] = {
                "episode_sha256": sha256_file(episode_path),
                "decisions_sha256": sha256_file(decision_path),
            }
            _write_json(index_path, index)
        if (episode.get("episode_seed") != episode_seed or episode.get("method") != "uamcts_cc4"
                or episode.get("policy_seed") != policy_seed):
            raise RuntimeError(f"episode identity mismatch for seed {episode_seed}")
        all_episodes.append(episode); all_decisions.extend(decisions)

    episodes_path = output / "episodes.jsonl"
    decisions_path = output / "decisions.jsonl"
    episodes_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in all_episodes), encoding="utf-8")
    decisions_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in all_decisions), encoding="utf-8")
    metrics = aggregate_repeat(all_episodes, expected_ticks=ticks).to_dict()
    _write_json(output / "metrics.json", metrics)
    (output / "stdout.log").touch()
    dependencies, hardware = dependency_and_hardware(device)
    method_artifacts, model_sha = artifact_provenance("uamcts_cc4", output)
    paper_path = REPO_ROOT / PAPER_FILES["uamcts_cc4"]
    if not paper_path.is_file():
        raise RuntimeError(f"missing source paper: {paper_path}")
    paper_sha = sha256_file(paper_path)
    if paper_sha != EXPECTED_PAPER_HASHES["uamcts_cc4"]:
        raise RuntimeError(f"source paper SHA256 mismatch for {paper_path.name}")
    parameter_hash = _parameter_identity_hash(frozen_artifacts)
    manifest = {
        "schema_version": 1, "protocol_version": PROTOCOL_VERSION,
        "method": METHOD_NAMES["uamcts_cc4"], "method_slug": "uamcts_cc4",
        "repeat_index": repeat_index, "training_seed": policy_seed,
        "test_episode_seeds": episode_seeds, "episode_ticks": ticks,
        "code_commit": git["code_commit"], "git_dirty": git["git_dirty"],
        "git_diff_sha256": git["git_diff_sha256"], "config_sha256": config_sha,
        "paper_sha256": paper_sha, "upstream_commit": None,
        "python_version": platform.python_version(), "dependencies": dependencies,
        "hardware": hardware, "model_sha256": model_sha,
        "method_artifacts": method_artifacts,
        "frozen_artifact_sha256": frozen_artifacts,
        "preflight_snapshot_sha256": snapshot_sha,
        "provider_calls": 0, "test_seeds_used": False,
        "test_parameter_hash_before": parameter_hash,
        "test_parameter_hash_after": parameter_hash,
        "test_time_parameter_hash_before": parameter_hash,
        "test_time_parameter_hash_after": parameter_hash,
        "test_parameter_sha256_before": parameter_hash,
        "test_parameter_sha256_after": parameter_hash,
        "run_mode": run_mode, "formal_result_eligible": run_mode == "formal",
        "table_id": "table1", "row_id": "uamcts_cc4", "component_variant": "uamcts_cc4",
        "reward_mode": "full_reward", "reward_semantics": "cc4_v3_full_reward",
        "artifact_sha256": {
            "config.resolved.yaml": config_sha,
            "policy_spec.json": policy_sha,
            "preflight_snapshot.json": snapshot_sha,
            "episodes.jsonl": sha256_file(episodes_path),
            "decisions.jsonl": sha256_file(decisions_path),
            "metrics.json": sha256_file(output / "metrics.json"),
        },
        "uamcts_preflight": {
            "schema": uamcts_gate.get("schema"),
            "eligible": bool(uamcts_gate.get("eligible")),
            "formal_result_eligible": False,
            "frozen_artifact_sha256": frozen_artifacts,
        },
    }
    _write_json(output / "manifest.json", manifest)
    validation = validate_run_directory(output, formal=run_mode == "formal")
    _write_json(output / "validation.json", validation)
    if not validation["passed"]:
        raise RuntimeError(f"repeat validation failed: {validation['errors']}")
    return {"manifest": manifest, "metrics": metrics, "validation": validation}


def run_repeat(*, method: str, run_mode: str, repeat_index: int, policy_seed: int,
               episode_seeds: list[int], ticks: int, output: Path, device: str = "cpu",
               resume: bool = False,
               episode_runner: Callable[..., tuple[dict, list[dict]]] = run_method_episode) -> dict:
    if method not in METHODS:
        raise ValueError("unsupported method")
    validate_repeat_request(run_mode=run_mode, repeat_index=repeat_index,
                            policy_seed=policy_seed, episode_seeds=episode_seeds, ticks=ticks)
    git = git_state()
    if run_mode == "formal" and git["git_dirty"]:
        raise RuntimeError("formal repeat requires a clean git snapshot")
    if method == "uamcts_cc4":
        # This is intentionally before output.mkdir: a blocked UAMCTS gate
        # must not leave an apparently resumable output directory.
        uamcts_gate, preflight_artifacts = _uamcts_preflight()
        return _run_repeat_uamcts(
            run_mode=run_mode, repeat_index=repeat_index, policy_seed=policy_seed,
            episode_seeds=episode_seeds, ticks=ticks, output=output, device=device,
            resume=resume, episode_runner=episode_runner, git=git,
            uamcts_gate=uamcts_gate, preflight_artifacts=preflight_artifacts,
        )
    return _run_repeat_generic(
        method=method, run_mode=run_mode, repeat_index=repeat_index,
        policy_seed=policy_seed, episode_seeds=episode_seeds, ticks=ticks,
        output=output, device=device, resume=resume, episode_runner=episode_runner,
        git=git,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--run-mode", required=True, choices=("dev", "formal"))
    parser.add_argument("--repeat-index", type=int, required=True)
    parser.add_argument("--policy-seed", type=int, required=True)
    parser.add_argument("--episode-seeds", required=True)
    parser.add_argument("--ticks", type=int, required=True)
    parser.add_argument("--device", default="cpu"); parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    seeds = [int(value) for value in args.episode_seeds.split(",") if value.strip()]
    result = run_repeat(method=args.method, run_mode=args.run_mode, repeat_index=args.repeat_index,
                        policy_seed=args.policy_seed, episode_seeds=seeds, ticks=args.ticks,
                        output=args.out, device=args.device, resume=args.resume)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
