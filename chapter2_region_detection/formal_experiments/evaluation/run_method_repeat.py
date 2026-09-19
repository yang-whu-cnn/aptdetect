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
                "uamcts_cc4": "UAMCTS-CC4 (adapted)"}
PAPER_FILES = {"dca_cc4": "DCA.pdf", "rsmbrl_cc4": "RSMBRL.pdf",
               "uamcts_cc4": "UAMCTS.pdf"}
EXPECTED_PAPER_HASHES = {
    "dca_cc4": "173fa04a1675e98ce636c2a8b2eda9eae5e3c9e2347f5f19cb4aafe6708e3cb1",
    "rsmbrl_cc4": "df594e5b3d2bd36e06d94d96e3cde0c549fa1894a9fe5d26ecd6c4552c91d4d7",
    "uamcts_cc4": "5dffdc568bf6e7fa04a430c44dcac9d293b588c4a2b5211cc5f14324f10af145",
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
        base = REPO_ROOT / "chapter2_region_detection" / "outputs"
        paths = {
            "world_model_sha256": base / "world_model_final_20260917/a4_5b/world_model_absolute.pt",
            "reward_model_sha256": base / "world_model_final_20260917/a4_5c/response_reward_predictor.pt",
            "progress_model_sha256": base / "uamcts_cc4/progress/progress_ensemble_train_only.pt",
            "prototype_prior_sha256": base / "priorrl_cc4/prototypes/frozen_prototypes.json",
            "normalizer_sha256": base / "uamcts_cc4/calibration/uamcts_uncertainty_normalizers_frozen.pt",
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise RuntimeError(f"UAMCTS frozen artifact gate failed: missing {missing}")
        artifacts = {"policy_spec_sha256": policy_sha,
                     **{key: sha256_file(path) for key, path in paths.items()}}
        return artifacts, artifacts["world_model_sha256"]
    from baselines.rsmbrl_cc4.artifact_preflight import (
        DEFAULT_NORMALIZER, DEFAULT_REWARD_MODEL, DEFAULT_WORLD_MODEL, resolve,
    )
    artifacts = {"policy_spec_sha256": policy_sha,
                 "world_model_sha256": sha256_file(resolve(DEFAULT_WORLD_MODEL)),
                 "reward_model_sha256": sha256_file(resolve(DEFAULT_REWARD_MODEL)),
                 "normalizer_sha256": sha256_file(resolve(DEFAULT_NORMALIZER))}
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
    _write_json(output / "policy_spec.json", {"method": method, "policy_seed": policy_seed})
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
        "reward_mode": "full_reward" if method in ("rsmbrl_cc4", "uamcts_cc4") else "not_applicable",
        "reward_semantics": ("cc4_v3_full_reward" if method in ("rsmbrl_cc4", "uamcts_cc4")
                             else "fixed_response_mapping_no_learning_reward"),
        # eligibility_report.json is deliberately excluded: it is the signed-by-
        # validation verdict over these inputs, and including itself would create
        # a recursive hash dependency.
        "artifact_sha256": {
            "config.resolved.yaml": sha256_file(output / "config.resolved.yaml"),
            "policy_spec.json": sha256_file(output / "policy_spec.json"),
            "episodes.jsonl": sha256_file(episodes_path),
            "decisions.jsonl": sha256_file(decisions_path),
            "metrics.json": sha256_file(output / "metrics.json"),
        },
    }
    _write_json(output / "manifest.json", manifest)
    validation = validate_run_directory(output, formal=run_mode == "formal")
    _write_json(output / "validation.json", validation)
    if not validation["passed"]:
        raise RuntimeError(f"repeat validation failed: {validation['errors']}")
    return {"manifest": manifest, "metrics": metrics, "validation": validation}


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
