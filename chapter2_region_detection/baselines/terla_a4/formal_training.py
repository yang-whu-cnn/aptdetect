"""Resumable CPU-only formal TERLA-A4 training with validation-only selection."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Callable

import torch

from .model import TERLAPolicy
from .training import PPOConfig, TERLAPPOTrainer, policy_hash

TRAIN_SEEDS = tuple(range(1000, 1032))
VALIDATION_SEEDS = tuple(range(2000, 2008))
TEST_SEEDS = tuple(range(4000, 4100))
POLICY_SEEDS = (51001, 51002, 51003, 51004, 51005)
TICKS = 500
CHECKPOINT_INTERVAL = 8
EpisodeRunner = Callable[..., dict]
GitStateProvider = Callable[[], dict]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_torch_save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_git_state() -> dict:
    """Return auditable source identity; callers must reject a dirty tree."""
    root = Path(__file__).resolve().parents[3]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                            capture_output=True).stdout.decode("ascii").strip()
    status = subprocess.run(["git", "status", "--porcelain=v1", "--untracked-files=all"],
                            cwd=root, check=True, capture_output=True).stdout
    diff = subprocess.run(["git", "diff", "--binary", "HEAD", "--"], cwd=root,
                          check=True, capture_output=True).stdout
    return {"code_commit": commit, "git_dirty": bool(status.strip()),
            "git_diff_sha256": hashlib.sha256(diff).hexdigest()}


def _verified_git_state(provider: GitStateProvider) -> dict:
    state = dict(provider())
    required = {"code_commit", "git_dirty", "git_diff_sha256"}
    if set(state) != required or not state["code_commit"] or len(state["git_diff_sha256"]) != 64:
        raise RuntimeError("invalid TERLA git provenance")
    if state["git_dirty"] is not False:
        raise RuntimeError("formal TERLA training requires a clean git worktree")
    return state


def _run_default(*, policy, episode_seed: int, ticks: int, training: bool,
                 trainer, generator, decisions_handle) -> dict:
    # Keep the heavyweight CC4/Ray environment optional for protocol/unit tests.
    from .run_development import run_episode
    rows, report = run_episode(policy=policy, seed=episode_seed, ticks=ticks,
                               mode="train" if training else "eval",
                               decisions_handle=decisions_handle, generator=generator)
    updates = []
    if training:
        trainer.add(rows)
        updates = trainer.flush_episode()
    return {**report, "ppo_updates": updates}


def select_validation_checkpoint(candidates: list[dict]) -> dict:
    """Maximise original TERLA validation reward; earliest step breaks ties."""
    if not candidates:
        raise ValueError("no TERLA validation candidates")
    for row in candidates:
        if tuple(row.get("validation_seeds", ())) != VALIDATION_SEEDS:
            raise ValueError("TERLA checkpoint selection must use the frozen validation split")
        if any(seed in TEST_SEEDS for seed in row["validation_seeds"]):
            raise ValueError("TERLA checkpoint selection used a test seed")
    return min(candidates, key=lambda row: (-float(row["validation_reward_mean"]),
                                            int(row["training_episodes"])))


def train_policy(*, policy_seed: int, output_dir: Path, resume: bool = False,
                 episode_runner: EpisodeRunner = _run_default,
                 ticks: int = TICKS, checkpoint_interval: int = CHECKPOINT_INTERVAL,
                 git_state: GitStateProvider = read_git_state) -> dict:
    if policy_seed not in POLICY_SEEDS:
        raise ValueError("policy_seed must be one of the five frozen TERLA seeds")
    if ticks != TICKS:
        raise ValueError("formal TERLA training requires exactly 500 ticks per episode")
    if checkpoint_interval <= 0 or len(TRAIN_SEEDS) % checkpoint_interval:
        raise ValueError("checkpoint_interval must evenly divide the frozen training split")
    provenance = _verified_git_state(git_state)

    torch.manual_seed(policy_seed)
    policy = TERLAPolicy().cpu()
    trainer = TERLAPPOTrainer(policy, PPOConfig())
    generator = torch.Generator(device="cpu").manual_seed(policy_seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / f"policy_{policy_seed}.resume.pt"
    completed: list[int] = []
    candidates: list[dict] = []
    if resume and progress_path.is_file():
        saved = torch.load(progress_path, map_location="cpu", weights_only=False)
        identity = (saved.get("method"), saved.get("policy_seed"), saved.get("training_split"),
                    saved.get("ticks_per_episode"), tuple(saved.get("training_seed_plan", ())))
        if identity != ("terla_a4", policy_seed, "train", TICKS, TRAIN_SEEDS):
            raise RuntimeError("TERLA resume checkpoint identity mismatch")
        if any(saved.get(key) != value for key, value in provenance.items()):
            raise RuntimeError("TERLA resume checkpoint git provenance mismatch")
        policy.load_state_dict(saved["state_dict"], strict=True)
        trainer.optimizer.load_state_dict(saved["optimizer_state_dict"])
        generator.set_state(saved["generator_state"])
        completed = [int(seed) for seed in saved.get("completed_training_seeds", ())]
        candidates = list(saved.get("validation_candidates", ()))
        if completed != list(TRAIN_SEEDS[:len(completed)]):
            raise RuntimeError("TERLA resume checkpoint has non-prefix training provenance")

    decisions_path = output_dir / f"policy_{policy_seed}.decisions.jsonl"
    with decisions_path.open("a" if completed else "w", encoding="utf-8") as decisions:
        for episode_seed in TRAIN_SEEDS[len(completed):]:
            report = episode_runner(policy=policy, episode_seed=episode_seed, ticks=TICKS,
                                    training=True, trainer=trainer, generator=generator,
                                    decisions_handle=decisions)
            if int(report.get("controller_tick_end", TICKS)) != TICKS:
                raise RuntimeError("TERLA training episode did not complete 500 ticks")
            completed.append(episode_seed)
            if len(completed) % checkpoint_interval == 0:
                candidate_path = output_dir / f"policy_{policy_seed}.step_{len(completed):02d}.pt"
                candidate_payload = {"schema": "terla_a4_validation_candidate_v1",
                    "method": "terla_a4", "policy_seed": policy_seed,
                    "training_episodes": len(completed), **provenance,
                    "state_dict": policy.state_dict()}
                _atomic_torch_save(candidate_path, candidate_payload)
                validation = []
                before = policy_hash(policy); policy.eval()
                with torch.no_grad():
                    for validation_seed in VALIDATION_SEEDS:
                        row = episode_runner(policy=policy, episode_seed=validation_seed, ticks=TICKS,
                                             training=False, trainer=trainer, generator=generator,
                                             decisions_handle=decisions)
                        validation.append({"episode_seed": validation_seed,
                                           "reward_sum": float(row["reward_sum"])})
                if policy_hash(policy) != before:
                    raise RuntimeError("TERLA validation mutated the shared policy")
                policy.train()
                candidates.append({"training_episodes": len(completed), "training_seed": episode_seed,
                    "checkpoint": candidate_path.name, "checkpoint_sha256": _sha(candidate_path),
                    **provenance,
                    "validation_seeds": list(VALIDATION_SEEDS), "validation": validation,
                    "validation_reward_mean": sum(x["reward_sum"] for x in validation) / len(validation)})
            _atomic_torch_save(progress_path, {
                "schema": "terla_a4_resume_v1", "method": "terla_a4", "policy_seed": policy_seed,
                "training_split": "train", "ticks_per_episode": TICKS,
                "training_seed_plan": list(TRAIN_SEEDS), "completed_training_seeds": completed,
                "validation_candidates": candidates, "state_dict": policy.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "generator_state": generator.get_state(), **provenance})

    selected = select_validation_checkpoint(candidates)
    selected_path = output_dir / selected["checkpoint"]
    if _sha(selected_path) != selected["checkpoint_sha256"]:
        raise RuntimeError("selected TERLA validation checkpoint hash mismatch")
    selected_payload = torch.load(selected_path, map_location="cpu", weights_only=True)
    if (selected_payload.get("method") != "terla_a4"
            or selected_payload.get("policy_seed") != policy_seed
            or selected_payload.get("training_episodes") != selected["training_episodes"]
            or any(selected_payload.get(key) != value for key, value in provenance.items())
            or any(selected.get(key) != value for key, value in provenance.items())):
        raise RuntimeError("selected TERLA checkpoint git provenance mismatch")
    selected_state = selected_payload["state_dict"]
    final_path = output_dir / f"policy_{policy_seed}.pt"
    final_payload = {"schema": "terla_a4_formal_checkpoint_v1", "method": "terla_a4",
        "policy_seed": policy_seed, "formal_training_complete": True, "training_split": "train",
        "training_episode_seeds": list(TRAIN_SEEDS), "validation_episode_seeds": list(VALIDATION_SEEDS),
        "test_seeds_used": False, "ticks_per_episode": TICKS, "shared_policy_across_agents": True,
        "per_agent_history_isolated": True, "reward": "original_terla_cyber_reward",
        "hidden_truth_policy_input": False, "selected_training_episodes": selected["training_episodes"],
        "selected_validation_reward_mean": selected["validation_reward_mean"],
        "selected_checkpoint_sha256": selected["checkpoint_sha256"], **provenance,
        "state_dict": selected_state}
    _atomic_torch_save(final_path, final_payload)
    manifest = {key: value for key, value in final_payload.items() if key != "state_dict"}
    manifest.update({"checkpoint": final_path.name, "checkpoint_sha256": _sha(final_path),
                     "validation_candidates": candidates})
    _atomic_json(output_dir / f"policy_{policy_seed}.training.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-seed", type=int, required=True, choices=POLICY_SEEDS)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/formal_v3/training/terla_a4"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(train_policy(policy_seed=args.policy_seed, output_dir=args.output_dir,
                                  resume=args.resume), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
