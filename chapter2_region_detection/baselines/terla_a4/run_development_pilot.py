from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import torch

from baselines.terla_a4.model import TERLAPolicy
from baselines.terla_a4.run_development import run_episode
from baselines.terla_a4.training import PPOConfig, TERLAPPOTrainer, policy_hash


POLICY_SEEDS = (51001, 51002)
TRAIN_EPISODE_SEEDS = (1000, 1001, 1002, 1003, 1004)
DEV_EVAL_EPISODE_SEEDS = (3200, 3201, 3202, 3203, 3204)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _sum_counts(episodes: list[dict], key: str) -> dict[str, int]:
    total = Counter()
    for episode in episodes:
        total.update(episode[key])
    return dict(sorted(total.items()))


def _episode_gate(episode: dict, ticks: int) -> bool:
    return bool(
        episode["requested_ticks"] == ticks
        and episode["environment_steps"] == ticks
        and episode["controller_tick_end"] == ticks
        and episode["all_agents_done"]
        and episode["all_trajectories_episode_truncated"]
        and not episode["errors"]
    )


def _compact(episodes: list[dict]) -> dict:
    return {
        "episode_count": len(episodes),
        "episode_seeds": [row["seed"] for row in episodes],
        "requested_ticks": [row["requested_ticks"] for row in episodes],
        "executed_environment_steps": [row["environment_steps"] for row in episodes],
        "controller_tick_end": [row["controller_tick_end"] for row in episodes],
        "transition_count": sum(row["transitions"] for row in episodes),
        "reward_sum": sum(row["reward_sum"] for row in episodes),
        "requested_action_counts": _sum_counts(episodes, "requested_action_counts"),
        "executed_action_counts": _sum_counts(episodes, "executed_action_counts"),
        "duration_counts": _sum_counts(episodes, "duration_counts"),
        "fallback_reasons": _sum_counts(episodes, "fallback_reasons"),
        "error_count": sum(len(row["errors"]) for row in episodes),
        "wall_seconds": sum(row["wall_seconds"] for row in episodes),
        "episodes": episodes,
    }


def run_policy_seed(*, policy_seed: int, ticks: int, output_dir: Path,
                    device: torch.device) -> dict:
    torch.manual_seed(policy_seed)
    torch.cuda.manual_seed_all(policy_seed)
    policy = TERLAPolicy().to(device)
    trainer = TERLAPPOTrainer(policy, PPOConfig())
    generator = torch.Generator(device=device.type).manual_seed(policy_seed)
    initial_hash = policy_hash(policy)
    decisions_path = output_dir / f"decisions_policy_{policy_seed}.jsonl"
    train_episodes: list[dict] = []
    eval_episodes: list[dict] = []
    update_reports: list[dict] = []
    optimizer_id = id(trainer.optimizer)
    buffer_id = id(trainer.buffer)

    with decisions_path.open("w", encoding="utf-8", newline="\n") as handle:
        policy.train()
        for episode_seed in TRAIN_EPISODE_SEEDS:
            rows, report = run_episode(
                policy=policy, seed=episode_seed, ticks=ticks, mode="train",
                decisions_handle=handle, generator=generator,
            )
            train_episodes.append(report)
            trainer.add(rows)
            fragments = trainer.flush_episode()
            update_reports.append({
                "episode_seed": episode_seed,
                "fragment_sizes": [item["transitions"] for item in fragments],
                "optimizer_steps": sum(item["updates"] for item in fragments),
                "mean_fragment_loss": (
                    sum(item["mean_loss"] for item in fragments) / len(fragments)
                    if fragments else None
                ),
                "buffer_size_after_boundary_flush": len(trainer.buffer),
            })
        trained_hash = policy_hash(policy)
        checkpoint_path = output_dir / f"checkpoint_policy_{policy_seed}.pt"
        torch.save({
            "policy_seed": policy_seed,
            "train_episode_seeds": TRAIN_EPISODE_SEEDS,
            "policy_state_dict": {k: v.detach().cpu() for k, v in policy.state_dict().items()},
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "ppo_config": trainer.config.__dict__,
        }, checkpoint_path)
        checkpoint_file_sha = _file_sha256(checkpoint_path)

        policy.eval()
        before_eval_hash = policy_hash(policy)
        with torch.inference_mode():
            for episode_seed in DEV_EVAL_EPISODE_SEEDS:
                _rows, report = run_episode(
                    policy=policy, seed=episode_seed, ticks=ticks, mode="eval",
                    decisions_handle=handle, generator=generator,
                )
                eval_episodes.append(report)
        after_eval_hash = policy_hash(policy)

    train_gate = all(_episode_gate(row, ticks) for row in train_episodes)
    eval_gate = all(_episode_gate(row, ticks) for row in eval_episodes)
    fallback_count = sum(sum(row["fallback_reasons"].values())
                         for row in train_episodes + eval_episodes)
    error_count = sum(len(row["errors"]) for row in train_episodes + eval_episodes)
    fragment_gate = all(
        report["buffer_size_after_boundary_flush"] == 0
        and all(0 < size <= trainer.config.rollout for size in report["fragment_sizes"])
        for report in update_reports
    )
    passed = bool(
        train_gate and eval_gate and fragment_gate and error_count == 0
        and fallback_count == 0 and initial_hash != trained_hash
        and before_eval_hash == after_eval_hash == trained_hash
        and id(trainer.optimizer) == optimizer_id and id(trainer.buffer) == buffer_id
    )
    return {
        "policy_seed": policy_seed,
        "passed": passed,
        "device": str(device),
        "train": _compact(train_episodes),
        "development_eval": _compact(eval_episodes),
        "ppo_updates": update_reports,
        "persistent_optimizer": id(trainer.optimizer) == optimizer_id,
        "persistent_buffer": id(trainer.buffer) == buffer_id,
        "policy_hashes": {
            "initial": initial_hash,
            "after_training": trained_hash,
            "before_development_eval": before_eval_hash,
            "after_development_eval": after_eval_hash,
        },
        "checkpoint": {
            "path": checkpoint_path.name,
            "file_sha256": checkpoint_file_sha,
            "policy_hash": trained_hash,
        },
        "decisions": {
            "path": decisions_path.name,
            "file_sha256": _file_sha256(decisions_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ticks", type=int, default=100)
    args = parser.parse_args()
    if args.ticks != 100:
        raise SystemExit("development pilot is frozen at 100 ticks")
    if not torch.cuda.is_available():
        raise SystemExit("TERLA development pilot requires CUDA; fail closed")
    device = torch.device("cuda")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    args.out.mkdir(parents=True, exist_ok=True)
    config = {
        "schema": "terla_a4_development_pilot_2x5x100_v1",
        "policy_seeds": POLICY_SEEDS,
        "train_episode_seeds": TRAIN_EPISODE_SEEDS,
        "development_eval_episode_seeds": DEV_EVAL_EPISODE_SEEDS,
        "ticks_per_episode": args.ticks,
        "device_required": "cuda",
        "deterministic_eval": True,
        "ppo": PPOConfig().__dict__,
        "formal_result_eligible": False,
        "performance_claim": False,
    }
    started = time.perf_counter()
    reports = [run_policy_seed(
        policy_seed=seed, ticks=args.ticks, output_dir=args.out, device=device,
    ) for seed in POLICY_SEEDS]
    decisions_digest = hashlib.sha256()
    for report in reports:
        decisions_digest.update((args.out / report["decisions"]["path"]).read_bytes())
    passed = all(report["passed"] for report in reports)
    summary = {
        **config,
        "config_sha256": _canonical_sha256(config),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "policy_runs": reports,
        "aggregate": {
            "train_environment_steps": sum(sum(x["train"]["executed_environment_steps"])
                                           for x in reports),
            "development_eval_environment_steps": sum(
                sum(x["development_eval"]["executed_environment_steps"]) for x in reports),
            "train_transitions": sum(x["train"]["transition_count"] for x in reports),
            "development_eval_transitions": sum(
                x["development_eval"]["transition_count"] for x in reports),
            "fallback_count": sum(
                sum(x[split]["fallback_reasons"].values())
                for x in reports for split in ("train", "development_eval")
            ),
            "error_count": sum(
                x[split]["error_count"]
                for x in reports for split in ("train", "development_eval")
            ),
            "combined_decisions_sha256": decisions_digest.hexdigest(),
            "wall_seconds": time.perf_counter() - started,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
    }
    summary_path = args.out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8", newline="\n")
    print(json.dumps({
        "passed": passed,
        "summary": str(summary_path),
        "summary_sha256": _file_sha256(summary_path),
        "combined_decisions_sha256": decisions_digest.hexdigest(),
    }, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
