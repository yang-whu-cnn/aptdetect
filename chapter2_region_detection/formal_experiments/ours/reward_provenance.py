"""Immutable provenance sidecar for the approved Fail-Only reward artifact."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import numpy as np
import torch

from formal_experiments.ours.reward_ablation import RewardMode, transition_reward
from formal_experiments.ours.reward_artifact_contract import validate_frozen_reward_artifact


FORMAT = "table3_reward_provenance_v1"
DERIVATION_VERSION = "fail_only_incident_host_lwf_raw_penalty_v1"
LEGACY_CANDIDATE_FAIL_ONLY_PROVENANCE_SHA256 = "97fc9b908f7b8f6410adacb15ff7bea8793a2216d3c68d5446d73ede78e3dbdd"
CANDIDATE_FAIL_ONLY_PROVENANCE_SHA256 = "a1cb2162aff309f0f21fcaea7c1891c55c7d1e10fc8e4b86b42962214e2bc5d0"
CANDIDATE_TRAINING_SOURCE_AGGREGATE_SHA256 = "8d0dc9b20cc8014fc6367d578067fb6ce8d697d4abd9f3a7ef7d469e17e9ca2b"
CANDIDATE_PROVENANCE_PATH = Path(
    "outputs/formal_v3/table3_reward_models/fail_only/provenance_candidate.json"
)
# Intentionally unset until the root window commits the reviewed source tree
# and regenerates a clean=true final sidecar. Formal consumers must fail closed
# while this trust anchor is None.
APPROVED_FAIL_ONLY_PROVENANCE_SHA256: str | None = None

TRAINING_SOURCE_PATHS = (
    "formal_experiments/ours/reward_ablation.py",
    "formal_experiments/training/response_reward_predictor.py",
    "formal_experiments/training/bootstrap_world_model.py",
    "formal_experiments/evaluation/validate_bootstrap_world_model.py",
    "formal_experiments/evaluation/validate_response_reward_predictor.py",
    "formal_experiments/data_collection/decision_replay.py",
    "formal_experiments/data_collection/collect_cc4_formal_replay.py",
    "formal_experiments/data_collection/incident_response.py",
    "shared/action_contract.py",
    "shared/formal_state.py",
    "configs/compare_ug_cem_formal_v2_1.yaml",
)

SNAPSHOT_MATCH_SECTIONS = (
    "reward_artifact",
    "replay",
    "reward_label_derivation",
    "world_model",
    "training_source_snapshot",
    "runtime",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"BLOCKED: invalid {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"BLOCKED: {label} must be a JSON object: {path}")
    return value


def _resolve(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _relative(project_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        raise RuntimeError(f"BLOCKED: provenance path is outside project root: {path}")


def find_unique_upstream_manifest(
    *, project_root: Path, output_directory: Path, label: str,
) -> tuple[Path, dict[str, Any]]:
    """Find exactly one tracked-style docs manifest for an artifact directory."""
    docs = project_root / "docs"
    candidates: list[tuple[Path, dict[str, Any]]] = []
    if docs.is_dir():
        for path in sorted(docs.glob("*MANIFEST*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                declared = payload.get("output_directory") if isinstance(payload, dict) else None
                if declared and _resolve(project_root, declared).resolve() == output_directory.resolve():
                    candidates.append((path, payload))
            except Exception:
                continue
    if len(candidates) != 1:
        names = [item[0].as_posix() for item in candidates]
        raise RuntimeError(
            f"BLOCKED: {label} has no unique existing manifest for "
            f"{output_directory}; found {len(candidates)}: {names}"
        )
    return candidates[0]


def scan_replay(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256(); seeds: set[int] = set(); count = 0; size = 0
    with path.open("rb") as handle:
        for raw_line in handle:
            digest.update(raw_line); size += len(raw_line)
            if not raw_line.strip():
                continue
            try:
                item = json.loads(raw_line)
                seeds.add(int(item["episode_seed"]))
            except Exception as exc:
                raise RuntimeError(f"BLOCKED: invalid replay row in {path}: {exc}") from exc
            count += 1
    seed_list = sorted(seeds)
    return {
        "sha256": digest.hexdigest(), "bytes": size, "transition_count": count,
        "seeds": seed_list, "seed_list_sha256": canonical_sha256(seed_list),
    }


def _digest_normalizer(value: dict[str, Any]) -> str:
    digest = hashlib.sha256(b"reward-normalizer-digest-v1\0")
    for name in ("mean", "std"):
        array = np.ascontiguousarray(np.asarray(value[name]))
        digest.update(name.encode("ascii") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii") + b"\0")
        digest.update(array.tobytes(order="C"))
    digest.update(b"eps\0" + repr(float(value["eps"])).encode("ascii"))
    return digest.hexdigest()


def checkpoint_normalizer_digests(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    try:
        state_digest = _digest_normalizer(payload["state_normalizer"])
        reward_digest = _digest_normalizer(payload["reward_normalizer"])
        config = payload["config"]
    except Exception as exc:
        raise RuntimeError(f"BLOCKED: checkpoint normalizer metadata invalid: {exc}") from exc
    return {
        "algorithm": "sha256_reward_normalizer_digest_v1",
        "checkpoint_format_version": payload.get("format_version"),
        "state_normalizer_sha256": state_digest,
        "reward_normalizer_sha256": reward_digest,
        "combined_sha256": canonical_sha256({"state": state_digest, "reward": reward_digest}),
        "config_sha256": canonical_sha256(config),
    }


def _git_snapshot(project_root: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=project_root, capture_output=True, text=True,
            encoding="utf-8", errors="replace", check=True,
        )
        return result.stdout
    commit = git("rev-parse", "HEAD").strip()
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    diff = git("diff", "--binary", "HEAD")
    paths = sorted(line[3:] for line in status.splitlines() if len(line) >= 4)
    return {
        "commit": commit, "dirty": bool(status.strip()),
        "capture_scope": "sidecar_generation_snapshot",
        "status_porcelain_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "tracked_diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
        "dirty_path_count": len(paths), "dirty_paths_sha256": canonical_sha256(paths),
    }


def training_source_snapshot(project_root: Path) -> dict[str, Any]:
    files: dict[str, str] = {}
    for relative in TRAINING_SOURCE_PATHS:
        path = project_root / relative
        if not path.is_file():
            raise RuntimeError(f"BLOCKED: missing Fail-Only training source: {relative}")
        files[relative] = sha256_file(path)
    return {
        "algorithm": "sha256_file_bytes_v1",
        "files": files,
        "aggregate_sha256": canonical_sha256(files),
    }


def runtime_snapshot() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "python_executable_name": Path(sys.executable).name,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }


def _require_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise RuntimeError(f"BLOCKED: {label}: expected {expected!r}, got {actual!r}")


def _bind_training_snapshot_candidate(
    project_root: Path, payload: dict[str, Any],
) -> dict[str, Any]:
    candidate_path = project_root / CANDIDATE_PROVENANCE_PATH
    if not candidate_path.is_file():
        raise RuntimeError(
            f"BLOCKED: missing immutable Fail-Only training-snapshot candidate: {candidate_path}"
        )
    actual_sha = sha256_file(candidate_path)
    _require_equal(
        "training-snapshot candidate SHA256",
        actual_sha,
        CANDIDATE_FAIL_ONLY_PROVENANCE_SHA256,
    )
    candidate = _load_json(candidate_path, "Fail-Only training-snapshot candidate")
    _require_equal("training-snapshot candidate format", candidate.get("format"), FORMAT)
    _require_equal("training-snapshot candidate status", candidate.get("status"), "candidate")
    _require_equal("training-snapshot candidate mode", candidate.get("mode"), RewardMode.FAIL_ONLY.value)
    _require_equal("training-snapshot candidate current_clean", candidate.get("current_clean"), False)
    _require_equal(
        "training-snapshot candidate formal_result_eligible",
        candidate.get("formal_result_eligible"),
        False,
    )
    _require_equal(
        "training-snapshot candidate test_seeds_used",
        candidate.get("test_seeds_used"),
        False,
    )
    for section in SNAPSHOT_MATCH_SECTIONS:
        _require_equal(
            f"clean final matches training-snapshot candidate section {section}",
            payload.get(section),
            candidate.get(section),
        )
    aggregate = candidate.get("training_source_snapshot", {}).get("aggregate_sha256")
    _require_equal(
        "training-snapshot candidate source aggregate",
        aggregate,
        CANDIDATE_TRAINING_SOURCE_AGGREGATE_SHA256,
    )
    return {
        "candidate_path": CANDIDATE_PROVENANCE_PATH.as_posix(),
        "candidate_sha256": actual_sha,
        "training_source_aggregate_sha256": aggregate,
        "verified_sections": list(SNAPSHOT_MATCH_SECTIONS),
    }


def build_fail_only_provenance(project_root: Path, *, status: str = "candidate") -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    if status not in {"candidate", "final"}:
        raise ValueError("provenance status must be candidate or final")
    git_state = _git_snapshot(project_root)
    if status == "final" and git_state["dirty"]:
        raise RuntimeError("BLOCKED: final provenance requires a clean worktree")
    reward_manifest_path = project_root / "outputs/formal_v3/table3_reward_models/fail_only/frozen_manifest.json"
    reward_manifest, checkpoint_path, reward_audit = validate_frozen_reward_artifact(
        reward_manifest_path, mode=RewardMode.FAIL_ONLY, project_root=project_root,
    )

    replay_directory = project_root / "outputs/formal_replay_final_20260917"
    replay_manifest_path, replay_manifest = find_unique_upstream_manifest(
        project_root=project_root, output_directory=replay_directory, label="formal replay",
    )
    replay_splits: dict[str, Any] = {}
    all_seeds: set[int] = set()
    for split in ("train", "validation"):
        replay_path = replay_directory / f"{split}.jsonl"
        summary_path = replay_directory / f"{split}_summary.json"
        observed = scan_replay(replay_path); declared = replay_manifest.get("splits", {}).get(split, {})
        _require_equal(f"{split} replay sha256", observed["sha256"], declared.get("jsonl_sha256"))
        _require_equal(f"{split} replay bytes", observed["bytes"], declared.get("jsonl_bytes"))
        _require_equal(f"{split} transition count", observed["transition_count"], declared.get("transition_count"))
        _require_equal(f"{split} seed list", observed["seeds"], declared.get("seeds"))
        summary_sha = sha256_file(summary_path)
        _require_equal(f"{split} summary sha256", summary_sha, declared.get("summary_sha256"))
        overlap = all_seeds.intersection(observed["seeds"])
        if overlap:
            raise RuntimeError(f"BLOCKED: train/validation seed overlap: {sorted(overlap)}")
        all_seeds.update(observed["seeds"])
        replay_splits[split] = {
            "path": _relative(project_root, replay_path), **observed,
            "summary_path": _relative(project_root, summary_path),
            "summary_sha256": summary_sha,
        }
    test_overlap = sorted(all_seeds.intersection(range(4000, 4100)))
    if test_overlap:
        raise RuntimeError(f"BLOCKED: test seeds found in reward-model replay: {test_overlap}")

    world_path = project_root / "outputs/world_model_final_20260917/a4_5b/world_model_absolute.pt"
    world_output_directory = project_root / "outputs/world_model_final_20260917"
    model_manifest_path, model_manifest = find_unique_upstream_manifest(
        project_root=project_root, output_directory=world_output_directory, label="world model",
    )
    model_replay_manifest = _resolve(project_root, model_manifest.get("replay_manifest", ""))
    _require_equal("world-model replay manifest", model_replay_manifest.resolve(), replay_manifest_path.resolve())
    world_sha = sha256_file(world_path)
    _require_equal("world checkpoint sha256", world_sha,
                   model_manifest.get("world_model", {}).get("absolute_checkpoint_sha256"))
    _require_equal("reward manifest world sha256", reward_manifest.get("world_model_sha256"), world_sha)
    world_report_path = world_path.with_name("a4_5b_report.json")
    world_report_sha = sha256_file(world_report_path)
    _require_equal("world report sha256", world_report_sha,
                   model_manifest.get("world_model", {}).get("a4_5b_report_sha256"))
    _require_equal("world model quality gate", model_manifest.get("world_model", {}).get("quality_gate_pass"), True)

    derivation_module = Path(inspect.getsourcefile(transition_reward) or "").resolve()
    protocol_source = project_root / "formal_experiments/data_collection/incident_response.py"
    protocol_source_sha = sha256_file(protocol_source)
    _require_equal(
        "replay manifest reward source sha256", protocol_source_sha,
        replay_manifest.get("source_files", {}).get("formal_experiments/data_collection/incident_response.py"),
    )
    function_source = inspect.getsource(transition_reward)

    payload = {
        "format": FORMAT,
        "status": status,
        "provenance_review_commit": git_state["commit"] if status == "final" else None,
        "current_clean": not git_state["dirty"],
        "mode": RewardMode.FAIL_ONLY.value,
        "formal_result_eligible": False,
        "test_seeds_used": False,
        "reward_artifact": {
            "manifest_path": _relative(project_root, reward_manifest_path),
            "manifest_sha256": reward_audit["manifest_sha256"],
            "checkpoint_path": _relative(project_root, checkpoint_path),
            "checkpoint_sha256": reward_audit["checkpoint_sha256"],
            "normalizers": checkpoint_normalizer_digests(checkpoint_path),
        },
        "replay": {
            "manifest_path": _relative(project_root, replay_manifest_path),
            "manifest_sha256": sha256_file(replay_manifest_path),
            "reward_protocol": replay_manifest.get("reward_protocol"),
            "splits": replay_splits,
            "train_validation_seed_union_sha256": canonical_sha256(sorted(all_seeds)),
            "test_seed_overlap_count": len(test_overlap),
        },
        "reward_label_derivation": {
            "version": DERIVATION_VERSION,
            "mode": RewardMode.FAIL_ONLY.value,
            "label_field": "incident_host_lwf_raw_penalty",
            "module_path": _relative(project_root, derivation_module),
            "module_sha256": sha256_file(derivation_module),
            "function_source_sha256": hashlib.sha256(function_source.encode("utf-8")).hexdigest(),
            "upstream_reward_protocol_source_path": _relative(project_root, protocol_source),
            "upstream_reward_protocol_source_sha256": protocol_source_sha,
        },
        "world_model": {
            "checkpoint_path": _relative(project_root, world_path),
            "checkpoint_sha256": world_sha,
            "manifest_path": _relative(project_root, model_manifest_path),
            "manifest_sha256": sha256_file(model_manifest_path),
            "report_path": _relative(project_root, world_report_path),
            "report_sha256": world_report_sha,
            "selected_target_mode": model_manifest.get("world_model", {}).get("selected_target_mode"),
            "quality_gate_pass": True,
        },
        "training_source_snapshot": training_source_snapshot(project_root),
        "runtime": runtime_snapshot(),
        "git_state": git_state,
    }
    if status == "final":
        payload["training_snapshot_binding"] = _bind_training_snapshot_candidate(
            project_root, payload,
        )
    return payload


def write_immutable_sidecar(path: Path, payload: dict[str, Any]) -> str:
    path = Path(path); encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != encoded:
        raise RuntimeError(f"BLOCKED: refusing to overwrite immutable provenance sidecar: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(encoded, encoding="utf-8")
    return sha256_file(path)


def _validate_provenance(
    path: Path, *, project_root: Path, expected_sha256: str,
    require_final: bool,
) -> dict[str, Any]:
    path = Path(path); actual_sha = sha256_file(path)
    if actual_sha != expected_sha256:
        raise RuntimeError(
            f"BLOCKED: Fail-Only provenance SHA mismatch: expected {expected_sha256}, got {actual_sha}"
        )
    payload = _load_json(path, "Fail-Only provenance sidecar")
    _require_equal("provenance format", payload.get("format"), FORMAT)
    _require_equal("provenance mode", payload.get("mode"), RewardMode.FAIL_ONLY.value)
    _require_equal("formal_result_eligible", payload.get("formal_result_eligible"), False)
    _require_equal("test_seeds_used", payload.get("test_seeds_used"), False)
    if require_final:
        _require_equal("provenance status", payload.get("status"), "final")
        _require_equal("provenance current_clean", payload.get("current_clean"), True)
        current_git = _git_snapshot(Path(project_root))
        _require_equal("current git clean", current_git["dirty"], False)
        review_commit = payload.get("provenance_review_commit")
        if not isinstance(review_commit, str) or len(review_commit) != 40:
            raise RuntimeError("BLOCKED: invalid provenance review commit")
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", review_commit, current_git["commit"]],
            cwd=project_root, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )
        if ancestor.returncode != 0:
            raise RuntimeError(
                "BLOCKED: provenance review commit is not an ancestor of current HEAD"
            )
        expected_binding = {
            "candidate_path": CANDIDATE_PROVENANCE_PATH.as_posix(),
            "candidate_sha256": CANDIDATE_FAIL_ONLY_PROVENANCE_SHA256,
            "training_source_aggregate_sha256": CANDIDATE_TRAINING_SOURCE_AGGREGATE_SHA256,
            "verified_sections": list(SNAPSHOT_MATCH_SECTIONS),
        }
        _require_equal(
            "final provenance training-snapshot binding",
            payload.get("training_snapshot_binding"),
            expected_binding,
        )
    fresh = build_fail_only_provenance(Path(project_root), status="candidate")
    for section in SNAPSHOT_MATCH_SECTIONS:
        _require_equal(f"provenance section {section}", payload.get(section), fresh.get(section))
    return payload


def validate_fail_only_provenance(path: Path, *, project_root: Path) -> dict[str, Any]:
    """Formal validator; blocked until a clean-commit sidecar is approved."""
    if APPROVED_FAIL_ONLY_PROVENANCE_SHA256 is None:
        raise RuntimeError("BLOCKED: no clean-commit approved Fail-Only provenance sidecar")
    return _validate_provenance(
        path, project_root=project_root,
        expected_sha256=APPROVED_FAIL_ONLY_PROVENANCE_SHA256,
        require_final=True,
    )


def validate_candidate_fail_only_provenance(
    path: Path, *, project_root: Path, expected_sha256: str,
) -> dict[str, Any]:
    """Review-only validator; never authorizes a formal consumer."""
    return _validate_provenance(
        path, project_root=project_root, expected_sha256=expected_sha256,
        require_final=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--status", choices=("candidate", "final"), default="candidate")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--expected-sha256")
    args = parser.parse_args(); root = args.project_root.resolve()
    default_name = "provenance_candidate.json" if args.status == "candidate" else "provenance_sidecar.json"
    if args.out is None:
        args.out = Path("outputs/formal_v3/table3_reward_models/fail_only") / default_name
    out = args.out if args.out.is_absolute() else root / args.out
    if args.validate:
        if args.status == "candidate":
            if not args.expected_sha256:
                parser.error("candidate validation requires --expected-sha256")
            payload = validate_candidate_fail_only_provenance(
                out, project_root=root, expected_sha256=args.expected_sha256,
            )
        else:
            payload = validate_fail_only_provenance(out, project_root=root)
        digest = sha256_file(out)
    else:
        payload = build_fail_only_provenance(root, status=args.status)
        digest = write_immutable_sidecar(out, payload)
    print(json.dumps({"path": str(out), "sha256": digest, "payload": payload}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
