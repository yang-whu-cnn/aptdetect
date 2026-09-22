"""Resumable, offline-only PriorRL alpha selection and five-repeat training."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

import torch

from .cc4_callback import run_cc4_episode
from .policy import PriorRLActorCritic, PriorRLPPOConfig
from .prototype_retrieval import FrozenPrototypeRetriever
from .training import A4PPOConfig, A4PPOTrainer, checkpoint_sha256
from formal_experiments.common.prior_provenance import validate_frozen_prior_provenance

TRAIN_SEEDS = tuple(range(1000, 1032))
VALIDATION_SEEDS = tuple(range(2000, 2008))
POLICY_SEEDS = (51001, 51002, 51003, 51004, 51005)
ALPHA_GRID = (.01, .05, .1, .5)
TUNING_SEED = 50999
TICKS = 500
REPO_ROOT = Path(__file__).resolve().parents[3]
SHA256_RE = re.compile(r"[0-9a-f]{64}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def require_clean_git_snapshot() -> dict:
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT,
        capture_output=True, text=True, check=True, timeout=10,
    ).stdout
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"], cwd=REPO_ROOT,
        capture_output=True, check=True, timeout=20,
    ).stdout
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
        capture_output=True, text=True, check=True, timeout=10,
    ).stdout.strip()
    if status.strip():
        raise RuntimeError("PriorRL formal training requires a clean git snapshot")
    return {"code_commit": commit, "git_dirty": False,
            "git_diff_sha256": hashlib.sha256(diff).hexdigest()}


def validate_prototype_provenance(
    prototype: Path, *, coverage: Path | None = None, provenance: Path | None = None,
) -> dict:
    """Use the same frozen-prior validator as the Table 2/3 formal runner."""
    prototype = Path(prototype)
    expected_coverage = prototype.with_name("frozen_prototype_coverage.json")
    expected_provenance = prototype.with_name("frozen_prototype_provenance.json")
    if coverage is not None and Path(coverage).resolve() != expected_coverage.resolve():
        raise ValueError("PriorRL coverage must be the canonical sibling artifact")
    if provenance is not None and Path(provenance).resolve() != expected_provenance.resolve():
        raise ValueError("PriorRL provenance must be the canonical sibling artifact")
    try:
        audit = validate_frozen_prior_provenance(prototype)
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc
    return {
        **{key: audit[key] for key in (
            "prototype_sha256", "prototype_file_sha256",
            "prototype_coverage_sha256", "prototype_provenance_sha256")},
        "supplement_logical_manifest_sha256": audit["logical_manifest_sha256"],
        "supplement_entry_sha256_set_sha256": audit["entry_set_sha256"],
    }


def validate_alpha_selection(payload: dict, *, prototype_sha256: str,
                             prototype_file_sha256: str,
                             prototype_coverage_sha256: str,
                             prototype_provenance_sha256: str) -> float:
    if payload.get("schema") != "priorrl_alpha_selection_v1":
        raise ValueError("invalid PriorRL alpha-selection schema")
    if payload.get("train_seeds") != list(TRAIN_SEEDS) or payload.get("validation_seeds") != list(VALIDATION_SEEDS):
        raise ValueError("alpha selection did not use the frozen train/validation split")
    if payload.get("test_seeds_used") is not False or payload.get("online_llm_calls") != 0:
        raise ValueError("alpha selection violated isolation/offline requirements")
    for key, value in {
        "prototype_sha256": prototype_sha256,
        "prototype_file_sha256": prototype_file_sha256,
        "prototype_coverage_sha256": prototype_coverage_sha256,
        "prototype_provenance_sha256": prototype_provenance_sha256,
    }.items():
        if payload.get(key) != value:
            raise ValueError(f"alpha selection {key} mismatch")
    if (COMMIT_RE.fullmatch(str(payload.get("code_commit", ""))) is None
            or payload.get("git_dirty") is not False
            or SHA256_RE.fullmatch(str(payload.get("git_diff_sha256", ""))) is None):
        raise ValueError("alpha selection clean git provenance is invalid")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or [x.get("alpha_kl") for x in candidates] != list(ALPHA_GRID):
        raise ValueError("alpha selection grid is incomplete")
    selected = float(payload.get("selected_alpha_kl"))
    ranked = sorted(candidates, key=lambda x: (-float(x["validation_response_reward_mean"]), float(x["alpha_kl"])))
    if selected != float(ranked[0]["alpha_kl"]):
        raise ValueError("alpha selection does not follow frozen score/tie-break rule")
    return selected


def validate_training_manifest(
    payload: dict,
    *,
    repeat_index: int,
    prototype_sha256: str,
    prototype_file_sha256: str,
    checkpoint_file_sha256: str,
    selection_file_sha256: str,
    prototype_coverage_sha256: str,
    prototype_provenance_sha256: str,
) -> float:
    """Validate the frozen training provenance before any test episode runs."""
    if repeat_index not in range(1, 6):
        raise ValueError("repeat index must be 1..5")
    expected = {
        "schema": "priorrl_formal_training_manifest_v1",
        "method": "PriorRL-PPO-CC4",
        "repeat_index": repeat_index,
        "policy_seed": POLICY_SEEDS[repeat_index - 1],
        "train_seeds": list(TRAIN_SEEDS),
        "ticks_per_episode": TICKS,
        "prototype_sha256": prototype_sha256,
        "prototype_file_sha256": prototype_file_sha256,
        "prototype_coverage_sha256": prototype_coverage_sha256,
        "prototype_provenance_sha256": prototype_provenance_sha256,
        "checkpoint_file_sha256": checkpoint_file_sha256,
        "validation_selection_sha256": selection_file_sha256,
        "online_llm_calls": 0,
        "world_model_used": False,
        "test_seeds_used": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"PriorRL training manifest {key} mismatch")
    if (COMMIT_RE.fullmatch(str(payload.get("code_commit", ""))) is None
            or payload.get("git_dirty") is not False
            or SHA256_RE.fullmatch(str(payload.get("git_diff_sha256", ""))) is None):
        raise ValueError("PriorRL training manifest clean git provenance is invalid")
    alpha = float(payload.get("alpha_kl", float("nan")))
    if not math.isfinite(alpha) or alpha not in ALPHA_GRID:
        raise ValueError("PriorRL training manifest alpha_kl is invalid")
    rows = payload.get("train")
    if (not isinstance(rows, list)
            or [row.get("episode_seed") for row in rows] != list(TRAIN_SEEDS)
            or any(row.get("ticks") != TICKS for row in rows)):
        raise ValueError("PriorRL training manifest is incomplete")
    return alpha


def validate_checkpoint_metadata(
    metadata: dict, *, policy_seed: int, alpha_kl: float, code_commit: str,
) -> None:
    expected = {
        "schema": "priorrl_training_checkpoint_v1",
        "protocol": "formal_train",
        "policy_seed": policy_seed,
        "alpha_kl": alpha_kl,
        "ticks": TICKS,
        "code_commit": code_commit,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"PriorRL checkpoint metadata {key} mismatch")
    rows = metadata.get("train")
    if (not isinstance(rows, list)
            or [row.get("episode_seed") for row in rows] != list(TRAIN_SEEDS)
            or any(row.get("ticks") != TICKS for row in rows)):
        raise ValueError("PriorRL checkpoint training history is incomplete")


def _save_checkpoint(path: Path, *, policy, trainer, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({"policy_state_dict": policy.state_dict(), "optimizer_state_dict": trainer.optimizer.state_dict(),
                "metadata": metadata}, temporary)
    temporary.replace(path)


def _train_candidate(*, alpha: float, policy_seed: int, retriever, out: Path,
                     protocol: str, resume: bool, code_commit: str
                     ) -> tuple[PriorRLActorCritic, A4PPOTrainer, list[dict]]:
    torch.manual_seed(policy_seed)
    policy = PriorRLActorCritic(PriorRLPPOConfig(alpha_kl=alpha)).cpu()
    config = A4PPOConfig(alpha_kl=alpha)
    trainer = A4PPOTrainer(policy, config)
    checkpoint = out / "checkpoint.pt"
    rows: list[dict] = []
    start = 0
    if resume and checkpoint.is_file():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        meta = saved["metadata"]
        if (meta.get("alpha_kl") != alpha or meta.get("policy_seed") != policy_seed
                or meta.get("protocol") != protocol or meta.get("code_commit") != code_commit):
            raise RuntimeError("PriorRL training resume identity mismatch")
        policy.load_state_dict(saved["policy_state_dict"]); trainer.optimizer.load_state_dict(saved["optimizer_state_dict"])
        rows = list(meta.get("train", [])); start = len(rows)
    for episode_seed in TRAIN_SEEDS[start:]:
        row = run_cc4_episode(episode_seed=episode_seed, policy_seed=policy_seed, policy=policy,
            retriever=retriever, episode_ticks=TICKS, training=True, trainer=trainer,
            rollout_threshold=128, flush_final=True, protocol=protocol)
        rows.append({"episode_seed": episode_seed, "ticks": row["ticks"],
                     "transition_count": row["transition_count"], "ppo_updates": row["ppo_updates"],
                     "response_reward_sum": sum(x["response_reward"] for x in row["decisions"]),
                     "checkpoint_sha256": row["checkpoint_sha256"]})
        _save_checkpoint(checkpoint, policy=policy, trainer=trainer, metadata={
            "schema": "priorrl_training_checkpoint_v1", "protocol": protocol,
            "policy_seed": policy_seed, "alpha_kl": alpha, "ticks": TICKS,
            "code_commit": code_commit, "train": rows})
    return policy, trainer, rows


def select_alpha(*, prototype: Path, out: Path, resume: bool = False) -> dict:
    git = require_clean_git_snapshot()
    prior_provenance = validate_prototype_provenance(prototype)
    retriever = FrozenPrototypeRetriever.load(prototype)
    prototype_sha = retriever.payload["prototype_sha256"]
    candidates = []
    for alpha in ALPHA_GRID:
        candidate_dir = out / f"alpha_{alpha:g}"
        policy, _, train = _train_candidate(alpha=alpha, policy_seed=TUNING_SEED, retriever=retriever,
            out=candidate_dir, protocol="alpha_validation", resume=resume,
            code_commit=git["code_commit"])
        policy.eval(); validation = []
        with torch.no_grad():
            for seed in VALIDATION_SEEDS:
                row = run_cc4_episode(episode_seed=seed, policy_seed=TUNING_SEED, policy=policy,
                    retriever=retriever, episode_ticks=TICKS, training=False, protocol="alpha_validation")
                validation.append({"episode_seed": seed,
                    "response_reward_sum": sum(x["response_reward"] for x in row["decisions"])})
        candidates.append({"alpha_kl": alpha, "train_checkpoint_sha256": checkpoint_sha256(policy),
            "train": train, "validation": validation,
            "validation_response_reward_mean": sum(x["response_reward_sum"] for x in validation) / len(validation)})
    selected = sorted(candidates, key=lambda x: (-x["validation_response_reward_mean"], x["alpha_kl"]))[0]["alpha_kl"]
    payload = {"schema": "priorrl_alpha_selection_v1", "train_seeds": list(TRAIN_SEEDS),
        "validation_seeds": list(VALIDATION_SEEDS), "test_seeds_used": False, "ticks": TICKS,
        "tuning_seed": TUNING_SEED, "alpha_grid": list(ALPHA_GRID), "selected_alpha_kl": selected,
        "selection_metric": "validation_response_reward_mean", "tie_break": "smallest_alpha",
        **prior_provenance,
        "online_llm_calls": 0, "world_model_used": False, "candidates": candidates,
        **git}
    _write(out / "alpha_selection.json", payload)
    return payload


def train_repeat(*, repeat_index: int, prototype: Path, alpha_selection: Path,
                 out: Path, resume: bool = False) -> dict:
    if repeat_index not in range(1, 6):
        raise ValueError("repeat index must be 1..5")
    git = require_clean_git_snapshot()
    prior_provenance = validate_prototype_provenance(prototype)
    retriever = FrozenPrototypeRetriever.load(prototype)
    selection = json.loads(alpha_selection.read_text(encoding="utf-8"))
    alpha = validate_alpha_selection(selection, **{
        key: prior_provenance[key] for key in (
            "prototype_sha256", "prototype_file_sha256",
            "prototype_coverage_sha256", "prototype_provenance_sha256")})
    if selection["code_commit"] != git["code_commit"]:
        raise ValueError("alpha selection and formal training must use the same code commit")
    selection_snapshot = out / "alpha_selection.json"
    _write(selection_snapshot, selection)
    seed = POLICY_SEEDS[repeat_index - 1]
    policy, trainer, train = _train_candidate(alpha=alpha, policy_seed=seed, retriever=retriever,
        out=out, protocol="formal_train", resume=resume, code_commit=git["code_commit"])
    checkpoint = out / "checkpoint.pt"
    payload = {"schema": "priorrl_formal_training_manifest_v1", "method": "PriorRL-PPO-CC4",
        "repeat_index": repeat_index, "policy_seed": seed, "alpha_kl": alpha,
        "train_seeds": list(TRAIN_SEEDS), "validation_selection_sha256": _sha(selection_snapshot),
        **prior_provenance,
        "checkpoint_file_sha256": _sha(checkpoint), "checkpoint_parameter_sha256": checkpoint_sha256(policy),
        "ticks_per_episode": TICKS, "online_llm_calls": 0, "world_model_used": False,
        "test_seeds_used": False, "train": train, **git}
    _write(out / "training_manifest.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("select-alpha", "train-repeat"))
    parser.add_argument("--prototype", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True); parser.add_argument("--resume", action="store_true")
    parser.add_argument("--alpha-selection", type=Path); parser.add_argument("--repeat-index", type=int)
    args = parser.parse_args()
    if args.stage == "select-alpha": result = select_alpha(prototype=args.prototype, out=args.out, resume=args.resume)
    else:
        if args.alpha_selection is None or args.repeat_index is None:
            parser.error("train-repeat requires --alpha-selection and --repeat-index")
        result = train_repeat(repeat_index=args.repeat_index, prototype=args.prototype,
            alpha_selection=args.alpha_selection, out=args.out, resume=args.resume)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
