"""Create the frozen, baseline-specific RSMBRL uncertainty normalizers."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import torch

from baselines.rsmbrl_cc4.artifact_preflight import (
    DEFAULT_MANIFEST, DEFAULT_NORMALIZER, DEFAULT_NORMALIZER_SIDECAR,
    DEFAULT_REWARD_MODEL, DEFAULT_WORLD_MODEL, EXPECTED_CALIBRATION_SEEDS,
    EXPECTED_TRAIN_SEEDS, EXPECTED_VALIDATION_SEEDS, SIDECAR_SCHEMA,
    PROJECT_ROOT, resolve, run_preflight, sha256_file, validate_frozen_normalizer,
)
from baselines.ug_cem_apt.uncertainty import UGUncertainty, UGUncertaintyConfig
from formal_experiments.training.bootstrap_world_model import BootstrapProbabilisticWorldModel
from formal_experiments.training.response_reward_predictor import ResponseRewardPredictor
from shared.formal_state import BLUE_AGENTS, FORMAL_STATE_DIM
from shared.rollout_evaluator import SharedRolloutConfig, SharedRolloutEvaluator
from shared.d27_projection import D27ProjectionContext, PROJECTION_SHA256, PROJECTION_VERSION

DEFAULT_STATES = "outputs/ug_cem_v2/step6/calibration_states.jsonl"
DEFAULT_SUMMARY = "outputs/ug_cem_v2/step6/calibration_states_summary.json"
DEFAULT_REPLAY_MANIFEST = "docs/FINAL_REWARD_REPLAY_MANIFEST.json"
DEFAULT_TRAIN_REPLAY = "outputs/formal_replay_final_20260917/train.jsonl"
DEFAULT_TRAIN_SUMMARY = "outputs/formal_replay_final_20260917/train_summary.json"
DEFAULT_VALIDATION_REPLAY = "outputs/formal_replay_final_20260917/validation.jsonl"
DEFAULT_VALIDATION_SUMMARY = "outputs/formal_replay_final_20260917/validation_summary.json"
PLANNER_CALLS_PER_AGENT = 100
POPULATION_SIZE = 200
PLAN_SEED_BASE = 20260919
ALLOWED_KEYS = {"split", "episode_seed", "agent_name", "decision_index", "global_tick_start", "state"}
REPLAY_FIELDS = {
    "action_completed", "agent_name", "completed_action_success", "decision_dt",
    "decision_index", "done", "episode_seed", "executed_action_family",
    "executed_duration_ticks", "executed_index", "executed_label", "fallback",
    "fallback_reason", "global_tick_end", "global_tick_start", "incident_active_ticks",
    "incident_delay_penalty", "incident_event_ids", "incident_host_ids",
    "incident_host_lwf_count", "incident_host_lwf_raw_penalty", "next_state",
    "official_reward", "requested_action_id", "requested_action_name",
    "requested_cyborg_family", "requested_duration_ticks", "response_reward", "state",
    "target_host",
}


def inspect_clean_git() -> dict:
    status = subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
                            capture_output=True, text=True, check=True, timeout=10).stdout
    diff = subprocess.run(["git", "diff", "--binary", "HEAD"], cwd=PROJECT_ROOT,
                          capture_output=True, check=True, timeout=20).stdout
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                            capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    if status.strip() or diff:
        raise RuntimeError("RSMBRL normalizer rebuild requires a clean Git snapshot")
    return {"code_commit": commit, "git_dirty": False,
            "git_diff_sha256": hashlib.sha256(diff).hexdigest()}


def _audit_replay(split: str, replay_path: str | Path, summary_path: str | Path,
                  expected_seeds: tuple[int, ...]) -> dict:
    replay, summary_file = resolve(replay_path), resolve(summary_path)
    if not replay.is_file() or not summary_file.is_file():
        raise ValueError(f"{split} replay and summary must both exist")
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    if summary.get("split") != split or summary.get("seeds") != list(expected_seeds):
        raise ValueError(f"{split} summary split/seeds mismatch")
    if (summary.get("state_dim") != 27 or summary.get("n_actions") != 4
            or summary.get("episode_steps") != 500):
        raise ValueError(f"{split} summary schema mismatch")
    seeds, count = set(), 0
    with replay.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            if not isinstance(row, dict) or set(row) != REPLAY_FIELDS:
                raise ValueError(f"{split} replay line {line_no} schema mismatch")
            if (not isinstance(row["state"], list) or len(row["state"]) != 27
                    or not isinstance(row["next_state"], list) or len(row["next_state"]) != 27):
                raise ValueError(f"{split} replay line {line_no} state schema mismatch")
            seeds.add(int(row["episode_seed"]))
            count += 1
    if tuple(sorted(seeds)) != expected_seeds:
        raise ValueError(f"{split} replay seed coverage mismatch")
    if count != int(summary.get("transition_count", -1)):
        raise ValueError(f"{split} replay transition_count mismatch")
    return {"split": split, "seeds": list(expected_seeds),
            "replay_path": str(replay), "replay_sha256": sha256_file(replay),
            "summary_path": str(summary_file), "summary_sha256": sha256_file(summary_file),
            "transition_count": count}


def audit_rebuild_inputs(*, replay_manifest_path=DEFAULT_REPLAY_MANIFEST,
                         model_manifest_path=DEFAULT_MANIFEST,
                         train_replay=DEFAULT_TRAIN_REPLAY,
                         train_summary=DEFAULT_TRAIN_SUMMARY,
                         validation_replay=DEFAULT_VALIDATION_REPLAY,
                         validation_summary=DEFAULT_VALIDATION_SUMMARY,
                         states_path=DEFAULT_STATES, summary_path=DEFAULT_SUMMARY,
                         world_model_path=DEFAULT_WORLD_MODEL,
                         reward_model_path=DEFAULT_REWARD_MODEL) -> dict:
    git = inspect_clean_git()
    train = _audit_replay("train", train_replay, train_summary, EXPECTED_TRAIN_SEEDS)
    validation = _audit_replay("validation", validation_replay, validation_summary,
                               EXPECTED_VALIDATION_SEEDS)
    replay_manifest_file = resolve(replay_manifest_path)
    replay_manifest = json.loads(replay_manifest_file.read_text(encoding="utf-8"))
    if (replay_manifest.get("manifest_version") != 1
            or replay_manifest.get("reward_protocol") != "final_paper_20260917_v1"
            or replay_manifest.get("environment") != "cyborg_cc4"
            or replay_manifest.get("timesteps_per_episode") != 500):
        raise ValueError("replay manifest reward protocol mismatch")
    for split, audited in (("train", train), ("validation", validation)):
        declared = replay_manifest.get("splits", {}).get(split, {})
        if (declared.get("seeds") != audited["seeds"]
                or declared.get("transition_count") != audited["transition_count"]
                or declared.get("jsonl_sha256") != audited["replay_sha256"]
                or declared.get("summary_sha256") != audited["summary_sha256"]):
            raise ValueError(f"replay manifest {split} provenance mismatch")
    _, calibration = audit_calibration_states(states_path, summary_path)
    model_manifest_file = resolve(model_manifest_path)
    model_manifest = json.loads(model_manifest_file.read_text(encoding="utf-8"))
    world_file, reward_file = resolve(world_model_path), resolve(reward_model_path)
    if (model_manifest.get("manifest_version") != 1
            or model_manifest.get("reward_protocol") != "final_paper_20260917_v1"):
        raise ValueError("model manifest schema/protocol mismatch")
    if model_manifest.get("replay_manifest") != DEFAULT_REPLAY_MANIFEST:
        raise ValueError("model manifest replay_manifest mismatch")
    if (model_manifest.get("world_model", {}).get("selected_target_mode") != "absolute"
            or model_manifest.get("world_model", {}).get("quality_gate_pass") is not True
            or model_manifest.get("response_reward_predictor", {}).get("quality_gate_pass") is not True):
        raise ValueError("model manifest quality gate mismatch")
    if model_manifest.get("world_model", {}).get("absolute_checkpoint_sha256") != sha256_file(world_file):
        raise ValueError("model manifest world model SHA256 mismatch")
    if model_manifest.get("response_reward_predictor", {}).get("checkpoint_sha256") != sha256_file(reward_file):
        raise ValueError("model manifest reward model SHA256 mismatch")
    return {"git": git, "train": train, "validation": validation,
            "calibration": calibration,
            "replay_manifest_sha256": sha256_file(replay_manifest_file),
            "model_manifest_sha256": sha256_file(model_manifest_file)}


def _canonical_file_hash(path: Path) -> str:
    return sha256_file(path)


def audit_calibration_states(states_path: str | Path, summary_path: str | Path) -> tuple[list[dict], dict]:
    states_file, summary_file = resolve(states_path), resolve(summary_path)
    if not states_file.is_file() or not summary_file.is_file():
        raise ValueError("calibration states and provenance summary must both exist")
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    expected_summary = {
        "split": "calibration", "seeds": list(EXPECTED_CALIBRATION_SEEDS),
        "episode_steps": 100,
        "state_dim": FORMAL_STATE_DIM, "contains_hidden_truth": False,
        "contains_reward_labels": False,
    }
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            raise ValueError(f"calibration summary {key} mismatch: {summary.get(key)!r}")
    if set(summary.get("record_fields", ())) != ALLOWED_KEYS:
        raise ValueError("calibration summary record_fields violates state-only schema")
    records, counts_agent, counts_seed = [], Counter(), Counter()
    with states_file.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            item = json.loads(raw)
            if set(item) != ALLOWED_KEYS:
                raise ValueError(f"line {line_no}: hidden/reward/extra fields forbidden")
            if item["split"] != "calibration" or int(item["episode_seed"]) not in EXPECTED_CALIBRATION_SEEDS:
                raise ValueError(f"line {line_no}: non-calibration provenance")
            if item["agent_name"] not in BLUE_AGENTS:
                raise ValueError(f"line {line_no}: invalid Blue agent")
            state = np.asarray(item["state"], dtype=np.float32)
            if state.shape != (27,) or not np.isfinite(state).all():
                raise ValueError(f"line {line_no}: state is not finite D27")
            row = dict(item)
            row["state"] = state
            records.append(row)
            counts_agent[item["agent_name"]] += 1
            counts_seed[int(item["episode_seed"])] += 1
    if len(records) != int(summary.get("record_count", -1)):
        raise ValueError("calibration record_count does not match summary")
    if dict(counts_agent) != summary.get("per_agent_counts"):
        raise ValueError("calibration per-agent coverage does not match summary")
    expected_seed_counts = {int(k): int(v) for k, v in summary.get("per_seed_counts", {}).items()}
    if dict(counts_seed) != expected_seed_counts:
        raise ValueError("calibration per-seed coverage does not match summary")
    if set(counts_agent) != set(BLUE_AGENTS) or set(counts_seed) != set(EXPECTED_CALIBRATION_SEEDS):
        raise ValueError("calibration states do not cover all agents and frozen seeds")
    return records, {
        "states_path": str(states_file), "states_sha256": _canonical_file_hash(states_file),
        "summary_path": str(summary_file), "summary_sha256": _canonical_file_hash(summary_file),
        "record_count": len(records), "per_agent": dict(counts_agent),
        "per_seed": {str(k): v for k, v in sorted(counts_seed.items())},
    }


def _select(records: list[dict], agent: str) -> list[dict]:
    cells = defaultdict(list)
    for row in records:
        if row["agent_name"] == agent:
            cells[int(row["episode_seed"])].append(row)
    for rows in cells.values():
        rows.sort(key=lambda x: (int(x["decision_index"]), int(x["global_tick_start"])))
    selected, offset = [], 0
    while len(selected) < PLANNER_CALLS_PER_AGENT:
        for seed in EXPECTED_CALIBRATION_SEEDS:
            if offset < len(cells[seed]):
                selected.append(cells[seed][offset])
                if len(selected) == PLANNER_CALLS_PER_AGENT:
                    break
        offset += 1
    if len(selected) != PLANNER_CALLS_PER_AGENT or {int(x["episode_seed"]) for x in selected} != set(EXPECTED_CALIBRATION_SEEDS):
        raise ValueError(f"{agent}: insufficient balanced calibration coverage")
    return selected


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".normalizer.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _calibrate_to_path(*, states_path=DEFAULT_STATES, summary_path=DEFAULT_SUMMARY,
                       world_model_path=DEFAULT_WORLD_MODEL,
                       reward_model_path=DEFAULT_REWARD_MODEL,
                       output_path=DEFAULT_NORMALIZER, device="cpu") -> dict:
    records, provenance = audit_calibration_states(states_path, summary_path)
    world_path, reward_path, output = resolve(world_model_path), resolve(reward_model_path), resolve(output_path)
    world_sha, reward_sha = sha256_file(world_path), sha256_file(reward_path)
    world = BootstrapProbabilisticWorldModel.load_checkpoint(world_path, device=device)
    reward = ResponseRewardPredictor.load_checkpoint(reward_path, device=device)
    if (world.config.state_dim, world.config.n_actions, world.config.ensemble_size, world.config.target_mode) != (27, 4, 5, "absolute"):
        raise ValueError("world model violates final D27/A4/M5/absolute contract")
    if (reward.config.state_dim, reward.config.n_actions) != (27, 4):
        raise ValueError("reward predictor violates final D27/A4 contract")
    evaluator = SharedRolloutEvaluator(world_model=world, reward_predictor=reward,
                                       config=SharedRolloutConfig(horizon=4, gamma_tick=0.99))
    agents, coverage = {}, {}
    for agent_idx, agent in enumerate(BLUE_AGENTS):
        uncertainty = UGUncertainty(UGUncertaintyConfig(alpha=0.01, eps=1e-8, device=device))
        selected = _select(records, agent)
        generator = torch.Generator(device=device).manual_seed(PLAN_SEED_BASE + agent_idx)
        with torch.inference_mode():
            for row in selected:
                plans = torch.randint(0, 4, (POPULATION_SIZE, 4), generator=generator, device=device)
                context = D27ProjectionContext.from_root(
                    row["state"], root_tick=int(row["global_tick_start"]), episode_steps=100
                )
                rollout = evaluator.evaluate(row["state"], plans, projection_context=context)
                uncertainty.compute(rollout.next_states, update_stats=True)
        if uncertainty.obs_mean is None or uncertainty.obs_std is None or uncertainty.horizon_std is None:
            raise RuntimeError(f"{agent}: normalizer was not initialized")
        agents[agent] = {
            "obs_mean": uncertainty.obs_mean.detach().cpu(),
            "obs_std": uncertainty.obs_std.detach().cpu(),
            "horizon_std": uncertainty.horizon_std.detach().cpu(),
        }
        coverage[agent] = {
            "planner_calls": len(selected),
            "seeds": sorted({int(x["episode_seed"]) for x in selected}),
            "plan_seed": PLAN_SEED_BASE + agent_idx,
            "plans_per_call": POPULATION_SIZE,
        }
    bundle = {
        "format_version": 2, "method": "RSMBRL-CC4", "source_split": "calibration",
        "calibration_seeds": list(EXPECTED_CALIBRATION_SEEDS), "horizon": 4,
        "state_dim": 27, "n_actions": 4, "ensemble_size": 5,
        "freeze_after_calibration": True, "online_updates_after_calibration": False,
        "world_model_sha256": world_sha, "reward_model_sha256": reward_sha,
        "projection_version": PROJECTION_VERSION, "projection_sha256": PROJECTION_SHA256,
        "beta": 0.1, "beta_source": "paper_fixed_no_validation_tuning",
        "planner_profile": {"population_size": 200, "num_iterations": 5,
                            "elite_ratio": 0.3, "alpha": 0.1, "beta": 0.1},
        "calibration_sampling": "deterministic_uniform_a4_plan_batches_v1",
        "provenance": provenance, "coverage": coverage, "agents": agents,
    }
    _, errors = validate_frozen_normalizer_payload(bundle, world_sha)
    if errors:
        raise RuntimeError("pre-save frozen bundle validation failed: " + "; ".join(errors))
    _atomic_torch_save(bundle, output)
    _, errors = validate_frozen_normalizer(
        output, world_sha256=world_sha, reward_sha256=reward_sha
    )
    if errors:
        raise RuntimeError("post-save frozen bundle validation failed: " + "; ".join(errors))
    return {"status": "PASS", "output": str(output), "output_sha256": sha256_file(output),
            "world_model_sha256": world_sha, "reward_model_sha256": reward_sha,
            "provenance": provenance, "coverage": coverage}


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _promote_pair(temp_normalizer: Path, temp_sidecar: Path,
                  normalizer: Path, sidecar: Path) -> None:
    """Promote an artifact pair with rollback if either replace fails."""
    normalizer.parent.mkdir(parents=True, exist_ok=True)
    if sidecar.parent != normalizer.parent:
        raise ValueError("normalizer and sidecar must share a directory")
    backups: dict[Path, Path] = {}
    promoted: list[Path] = []
    try:
        for destination in (normalizer, sidecar):
            if destination.exists():
                handle = tempfile.NamedTemporaryFile(
                    dir=destination.parent, prefix=f".{destination.name}.",
                    suffix=".backup", delete=False,
                )
                backup = Path(handle.name)
                handle.close()
                backup.unlink()
                os.replace(destination, backup)
                backups[destination] = backup
        for source, destination in ((temp_normalizer, normalizer), (temp_sidecar, sidecar)):
            os.replace(source, destination)
            promoted.append(destination)
    except Exception:
        for destination in promoted:
            if destination.exists():
                destination.unlink()
        for destination, backup in backups.items():
            if backup.exists():
                os.replace(backup, destination)
        raise
    else:
        for backup in backups.values():
            if backup.exists():
                backup.unlink()


def calibrate(*, states_path=DEFAULT_STATES, summary_path=DEFAULT_SUMMARY,
              world_model_path=DEFAULT_WORLD_MODEL, reward_model_path=DEFAULT_REWARD_MODEL,
              output_path=DEFAULT_NORMALIZER,
              sidecar_path=DEFAULT_NORMALIZER_SIDECAR,
              replay_manifest_path=DEFAULT_REPLAY_MANIFEST,
              model_manifest_path=DEFAULT_MANIFEST,
              train_replay=DEFAULT_TRAIN_REPLAY, train_summary=DEFAULT_TRAIN_SUMMARY,
              validation_replay=DEFAULT_VALIDATION_REPLAY,
              validation_summary=DEFAULT_VALIDATION_SUMMARY,
              device="cpu") -> dict:
    """Rebuild and promote the normalizer/sidecar pair transactionally."""
    audited = audit_rebuild_inputs(
        replay_manifest_path=replay_manifest_path,
        model_manifest_path=model_manifest_path,
        train_replay=train_replay, train_summary=train_summary,
        validation_replay=validation_replay, validation_summary=validation_summary,
        states_path=states_path, summary_path=summary_path,
        world_model_path=world_model_path, reward_model_path=reward_model_path,
    )
    output, sidecar = resolve(output_path), resolve(sidecar_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent != sidecar.parent:
        raise ValueError("normalizer and sidecar must share a directory")
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".rsmbrl-rebuild-") as temp:
        temp_root = Path(temp)
        temp_normalizer = temp_root / output.name
        temp_sidecar = temp_root / sidecar.name
        report = _calibrate_to_path(
            states_path=states_path, summary_path=summary_path,
            world_model_path=world_model_path, reward_model_path=reward_model_path,
            output_path=temp_normalizer, device=device,
        )
        sidecar_payload = {
            "schema": SIDECAR_SCHEMA, "method": "RSMBRL-CC4",
            "normalizer_sha256": report["output_sha256"],
            "world_model_sha256": report["world_model_sha256"],
            "reward_model_sha256": report["reward_model_sha256"],
            "formal_result_eligible": False, "test_seeds_used": False,
            "normalizer_schema": {"format_version": 2, "source_split": "calibration",
                                  "state_dim": 27, "horizon": 4},
            "train": audited["train"], "validation": audited["validation"],
            "calibration": audited["calibration"], "git": audited["git"],
            "replay_manifest_sha256": audited["replay_manifest_sha256"],
            "model_manifest_sha256": audited["model_manifest_sha256"],
        }
        _write_json(temp_sidecar, sidecar_payload)
        gate = run_preflight(
            manifest_path=model_manifest_path, world_model_path=world_model_path,
            reward_model_path=reward_model_path, normalizer_path=temp_normalizer,
            normalizer_sidecar_path=temp_sidecar, device=device,
        )
        if not gate["eligible"]:
            raise RuntimeError("temporary RSMBRL artifact pair failed preflight: "
                               + "; ".join(gate["errors"]))
        _promote_pair(temp_normalizer, temp_sidecar, output, sidecar)
    return {**report, "output": str(output), "sidecar": str(sidecar),
            "sidecar_sha256": sha256_file(sidecar), "preflight": gate}


def validate_frozen_normalizer_payload(bundle: dict, world_sha: str):
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "bundle.pt"
        torch.save(bundle, path)
        return validate_frozen_normalizer(path, world_sha256=world_sha)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", default=DEFAULT_STATES)
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    parser.add_argument("--world-model", default=DEFAULT_WORLD_MODEL)
    parser.add_argument("--reward-model", default=DEFAULT_REWARD_MODEL)
    parser.add_argument("--output", default=DEFAULT_NORMALIZER)
    parser.add_argument("--sidecar", default=DEFAULT_NORMALIZER_SIDECAR)
    parser.add_argument("--replay-manifest", default=DEFAULT_REPLAY_MANIFEST)
    parser.add_argument("--model-manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--train-replay", default=DEFAULT_TRAIN_REPLAY)
    parser.add_argument("--train-summary", default=DEFAULT_TRAIN_SUMMARY)
    parser.add_argument("--validation-replay", default=DEFAULT_VALIDATION_REPLAY)
    parser.add_argument("--validation-summary", default=DEFAULT_VALIDATION_SUMMARY)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args(argv)
    if args.audit_only:
        _, report = audit_calibration_states(args.states, args.summary)
        print(json.dumps({"status": "PASS", **report}, indent=2, sort_keys=True))
        return 0
    report = calibrate(states_path=args.states, summary_path=args.summary,
                       world_model_path=args.world_model, reward_model_path=args.reward_model,
                       output_path=args.output, sidecar_path=args.sidecar,
                       replay_manifest_path=args.replay_manifest,
                       model_manifest_path=args.model_manifest,
                       train_replay=args.train_replay, train_summary=args.train_summary,
                       validation_replay=args.validation_replay,
                       validation_summary=args.validation_summary,
                       device=args.device)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
