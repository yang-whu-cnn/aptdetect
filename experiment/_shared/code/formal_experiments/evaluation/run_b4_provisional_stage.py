from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from formal_experiments.data_collection.incident_response import (
    RESPONSE_REWARD_PROTOCOL,
)
from formal_experiments.evaluation.run_b4_provisional_ppo import (
    DEFAULT_PPO_SEED,
    report_path_for_display,
    resolve_project_path,
    run_provisional_stage,
)
from formal_experiments.ours.ppo_core import PPOCoreConfig
from formal_experiments.ours.ppo_training import PPOTrainingConfig
from formal_experiments.ours.provisional_protocol import (
    DEFAULT_PROVISIONAL_SCENARIO_STEPS,
    DEFAULT_PROVISIONAL_STAGE_TARGET,
    FORMAL_ROLLOUT_TARGET,
    PROVISIONAL_STAGE_TARGETS,
    validate_probe_record,
    validate_stage_target,
)
from shared.formal_state import BLUE_AGENTS


PROTOCOL_MANIFEST_FORMAT_VERSION = 3
DEFAULT_FINAL_WORLD_MODEL = "outputs/world_model_final_20260917/a4_5b/world_model_absolute.pt"
DEFAULT_FINAL_REWARD_MODEL = "outputs/world_model_final_20260917/a4_5c/response_reward_predictor.pt"
DEFAULT_FINAL_CACHE_ROOT = "outputs/lwm_rl_final_20260917/prior_cache"
DEFAULT_FINAL_OUT_ROOT = "outputs/lwm_rl_final_20260917/b4/provisional"


def _stable_sha256(payload: dict) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _file_sha256(path: str | Path) -> str:
    return hashlib.sha256(resolve_project_path(path).read_bytes()).hexdigest()


def frozen_protocol_manifest(
    *,
    model_alias: str,
    world_model_path: str | Path = DEFAULT_FINAL_WORLD_MODEL,
    reward_model_path: str | Path = DEFAULT_FINAL_REWARD_MODEL,
    cache_root: str | Path = DEFAULT_FINAL_CACHE_ROOT,
) -> dict:
    return {
        "format_version": PROTOCOL_MANIFEST_FORMAT_VERSION,
        "phase": "gate_b4_provisional_ppo",
        "development_only": True,
        "formal_result_eligible": False,
        "model_alias": str(model_alias),
        "response_reward_protocol": RESPONSE_REWARD_PROTOCOL,
        "world_model_sha256": _file_sha256(world_model_path),
        "reward_model_sha256": _file_sha256(reward_model_path),
        "cache_namespace": str(cache_root),
        "scenario_steps": int(DEFAULT_PROVISIONAL_SCENARIO_STEPS),
        "ppo_seed": int(DEFAULT_PPO_SEED),
        "stage_targets": [int(x) for x in PROVISIONAL_STAGE_TARGETS],
        "rollout_target": int(FORMAL_ROLLOUT_TARGET),
        "ppo_core_config": asdict(PPOCoreConfig()),
        "ppo_training_config": asdict(PPOTrainingConfig(seed=int(DEFAULT_PPO_SEED))),
        "train_seed_schedule": "1000..1031 round-robin",
        "update_rule": "batch>=128 and all PPO/replay open decisions closed; final stage flush allowed",
        "formal_reuse_forbidden": True,
    }


def _manifest_path(*, out_root: str | Path, model_alias: str) -> Path:
    return resolve_project_path(out_root) / str(model_alias) / "protocol_manifest.json"


def _validate_manifest_payload(*, path: Path, expected: dict, expected_sha: str) -> dict:
    actual = json.loads(path.read_text(encoding="utf-8"))
    if actual != expected:
        raise RuntimeError("provisional protocol manifest mismatch")
    if _stable_sha256(actual) != expected_sha:
        raise RuntimeError("provisional protocol manifest hash mismatch")
    return actual


def write_or_validate_manifest(
    *,
    out_root: str | Path,
    model_alias: str,
    resume: bool,
    world_model_path: str | Path = DEFAULT_FINAL_WORLD_MODEL,
    reward_model_path: str | Path = DEFAULT_FINAL_REWARD_MODEL,
    cache_root: str | Path = DEFAULT_FINAL_CACHE_ROOT,
) -> tuple[Path, dict, str]:
    path = _manifest_path(out_root=out_root, model_alias=model_alias)
    expected = frozen_protocol_manifest(
        model_alias=model_alias,
        world_model_path=world_model_path,
        reward_model_path=reward_model_path,
        cache_root=cache_root,
    )
    expected_sha = _stable_sha256(expected)

    if resume:
        if not path.exists():
            raise FileNotFoundError(f"resume requested but protocol manifest missing: {path}")
        actual = _validate_manifest_payload(path=path, expected=expected, expected_sha=expected_sha)
        return path, actual, expected_sha

    if path.exists():
        # A crash can occur after the strict entrypoint writes its immutable manifest
        # but before the collector reaches the first completed episode/checkpoint.
        # Such a pre-checkpoint aborted attempt is safe to restart fresh: the policy
        # state has never been persisted and any already-generated priors live only in
        # the exact train cache, so they may be reused without another API charge.
        checkpoint_path = path.parent / "resume.pt"
        probe_path = path.parent / "probe_transitions.jsonl"
        if checkpoint_path.exists() or probe_path.exists():
            raise RuntimeError("provisional artifacts already exist; use --resume")
        actual = _validate_manifest_payload(path=path, expected=expected, expected_sha=expected_sha)
        return path, actual, expected_sha

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(expected, ensure_ascii=False, indent=2), encoding="utf-8")
    return path, expected, expected_sha


def cumulative_probe_summary(path: str | Path) -> dict:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"probe JSONL missing: {source}")

    requested = Counter()
    canonical = Counter()
    executed = Counter()
    per_agent = Counter()
    per_seed = Counter()
    fallback_count = 0
    action_completed_count = 0
    plan0_match_count = 0
    response_reward_total = 0.0
    official_reward_total = 0.0
    records = 0

    with source.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
                validate_probe_record(item)
            except Exception as exc:
                raise ValueError(f"invalid probe record at line {line_no}") from exc

            records += 1
            requested[int(item["requested_action_id"])] += 1
            canonical[int(item["canonical_action_id"])] += 1
            executed[str(item["executed_action_family"])] += 1
            per_agent[str(item["agent_name"])] += 1
            per_seed[int(item["episode_seed"])] += 1
            fallback_count += int(bool(item["fallback"]))
            action_completed_count += int(bool(item["action_completed"]))
            plan = tuple(int(x) for x in item["selected_plan"])
            plan0_match_count += int(int(item["requested_action_id"]) == plan[0])
            response_reward_total += float(item["response_reward"])
            official_reward_total += float(item.get("official_reward", 0.0))

    if records <= 0:
        raise ValueError("probe JSONL contains no records")

    return {
        "record_count": int(records),
        "requested_action_counts": {str(i): int(requested.get(i, 0)) for i in range(4)},
        "canonical_action_counts": {str(i): int(canonical.get(i, 0)) for i in range(4)},
        "executed_action_family_counts": dict(sorted(executed.items())),
        "per_agent_counts": {agent: int(per_agent.get(agent, 0)) for agent in BLUE_AGENTS},
        "per_seed_counts": {str(seed): int(count) for seed, count in sorted(per_seed.items())},
        "fallback_count": int(fallback_count),
        "action_completed_count": int(action_completed_count),
        "plan0_match_count": int(plan0_match_count),
        "response_reward_total": float(response_reward_total),
        "official_reward_total": float(official_reward_total),
    }


def run_strict_stage(
    *,
    model_alias: str,
    stage_target: int = DEFAULT_PROVISIONAL_STAGE_TARGET,
    device: str = "cpu",
    resume: bool = False,
    world_model_path: str | Path = DEFAULT_FINAL_WORLD_MODEL,
    reward_model_path: str | Path = DEFAULT_FINAL_REWARD_MODEL,
    cache_root: str | Path = DEFAULT_FINAL_CACHE_ROOT,
    out_root: str | Path = DEFAULT_FINAL_OUT_ROOT,
) -> dict:
    target = validate_stage_target(stage_target)
    if target != DEFAULT_PROVISIONAL_STAGE_TARGET and not resume:
        raise ValueError("stages above 2000 must continue from the frozen provisional checkpoint using --resume")

    manifest_path, manifest, manifest_sha = write_or_validate_manifest(
        out_root=out_root,
        model_alias=model_alias,
        resume=resume,
        world_model_path=world_model_path,
        reward_model_path=reward_model_path,
        cache_root=cache_root,
    )

    report = run_provisional_stage(
        model_alias=model_alias,
        stage_target=target,
        scenario_steps=DEFAULT_PROVISIONAL_SCENARIO_STEPS,
        ppo_seed=DEFAULT_PPO_SEED,
        world_model_path=world_model_path,
        reward_model_path=reward_model_path,
        cache_root=cache_root,
        out_root=out_root,
        device=device,
        resume=resume,
    )

    cumulative = cumulative_probe_summary(report["probe_path"])
    if int(cumulative["record_count"]) != int(report["probe_record_count"]):
        raise RuntimeError("cumulative probe summary count differs from collector report")
    if int(cumulative["record_count"]) != int(report["transition_count"]):
        raise RuntimeError("cumulative probe summary count differs from transition count")
    if int(cumulative["plan0_match_count"]) != int(report["transition_count"]):
        raise RuntimeError("cumulative probe contains a plan[0] mismatch")

    report["strict_protocol_manifest_path"] = str(manifest_path)
    report["strict_protocol_manifest_sha256"] = manifest_sha
    report["strict_protocol_manifest"] = manifest
    report["cumulative_probe_summary"] = cumulative
    report["strict_entrypoint"] = True
    report["pass"] = bool(report.get("pass", False) and cumulative["record_count"] > 0)

    report_path = report_path_for_display(report, out_root)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strict frozen Gate B4 provisional PPO stage entrypoint"
    )
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--stage-target", type=int, default=DEFAULT_PROVISIONAL_STAGE_TARGET)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--world-model", default=DEFAULT_FINAL_WORLD_MODEL)
    parser.add_argument("--reward-model", default=DEFAULT_FINAL_REWARD_MODEL)
    parser.add_argument("--cache-root", default=DEFAULT_FINAL_CACHE_ROOT)
    parser.add_argument("--out-root", default=DEFAULT_FINAL_OUT_ROOT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    report = run_strict_stage(
        model_alias=args.model_alias,
        stage_target=args.stage_target,
        device=args.device,
        resume=args.resume,
        world_model_path=args.world_model,
        reward_model_path=args.reward_model,
        cache_root=args.cache_root,
        out_root=args.out_root,
    )

    cumulative = report["cumulative_probe_summary"]
    print("=" * 80)
    print("[GATE B4 STRICT PROVISIONAL PPO STAGE]")
    print("model:", report["model_alias"], report["exact_model_id"])
    print("stage target/completed:", report["stage_target"], report["transition_count"])
    print("manifest sha256:", report["strict_protocol_manifest_sha256"])
    print("episodes/updates/policy_version:", report["episode_count"], report["update_count"], report["policy_version"])
    print("canonical actions cumulative:", cumulative["canonical_action_counts"])
    print("fallback cumulative:", cumulative["fallback_count"])
    print("response reward cumulative:", cumulative["response_reward_total"])
    print("cache hits/misses cumulative:", report["cache_hits_cumulative"], report["cache_misses_cumulative"])
    print("live API calls cumulative:", report["live_api_calls_cumulative"])
    print("estimated cost USD cumulative:", report["estimated_cost_usd_cumulative"])
    print("safe update barriers:", report["safe_update_barriers"])
    print("primary_model_selected:", not report["primary_model_selected_is_false"])
    print("pass:", report["pass"])
    print("[OK] report:", report_path_for_display(report, args.out_root))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
