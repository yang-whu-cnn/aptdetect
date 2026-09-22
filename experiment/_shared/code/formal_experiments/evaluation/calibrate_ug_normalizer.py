from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from baselines.ug_cem_apt.categorical_cem import (
    CategoricalCEMConfig,
    CategoricalCEMOptimizer,
)
from baselines.ug_cem_apt.normalizer_warmup import (
    CALIBRATION_SEEDS,
    UGNormalizerWarmup,
    UGNormalizerWarmupConfig,
    WarmupState,
)
from baselines.ug_cem_apt.planner import (
    UGCEMPlanner,
    UGCEMPlannerConfig,
)
from baselines.ug_cem_apt.uncertainty import (
    UGUncertainty,
    UGUncertaintyConfig,
)
from formal_experiments.training.bootstrap_world_model import (
    BootstrapProbabilisticWorldModel,
)
from formal_experiments.training.response_reward_predictor import (
    ResponseRewardPredictor,
)
from shared.formal_state import BLUE_AGENTS, FORMAL_STATE_DIM
from shared.rollout_evaluator import (
    SharedRolloutConfig,
    SharedRolloutEvaluator,
)


THIS = Path(__file__).resolve()
PROJ = THIS.parents[2]

DEFAULT_STATES = (
    "outputs/ug_cem_v2/step6/calibration_states.jsonl"
)
DEFAULT_WORLD_MODEL = (
    "outputs/world_model_v2/a4_5b/world_model_absolute.pt"
)
DEFAULT_REWARD_MODEL = (
    "outputs/world_model_v2/a4_5c/response_reward_predictor.pt"
)
DEFAULT_OUT_DIR = "outputs/ug_cem_v2/step6"

PLANNER_CALLS_PER_AGENT = 100
CALIBRATION_BETA = 0.10
CEM_POPULATION_SIZE = 64
CEM_NUM_ITERATIONS = 4
CEM_ELITE_RATIO = 0.30
CEM_ALPHA = 0.10
CEM_PROB_FLOOR = 0.01
CEM_SEED_BASE = 20260916
UG_ALPHA = 0.01
UG_EPS = 1e-8
CALIBRATION_SEED_ORDER = tuple(sorted(CALIBRATION_SEEDS))

ALLOWED_STATE_KEYS = frozenset({
    "split",
    "episode_seed",
    "agent_name",
    "decision_index",
    "global_tick_start",
    "state",
})


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (PROJ / path).resolve()
    return path


def load_calibration_records(path: str | Path) -> list[dict]:
    path = resolve_project_path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            item = json.loads(raw)
            if not isinstance(item, dict):
                raise ValueError(f"line {line_number}: record must be object")
            if frozenset(item.keys()) != ALLOWED_STATE_KEYS:
                raise ValueError(
                    f"line {line_number}: state-only schema mismatch; "
                    "hidden/reward/extra fields are forbidden"
                )

            warmup = WarmupState(
                state=np.asarray(item["state"], dtype=np.float32),
                split=str(item["split"]),
                episode_seed=item["episode_seed"],
                agent_name=str(item["agent_name"]),
            )
            if warmup.split != "calibration":
                raise ValueError("formal Step-6 runner requires calibration-only states")
            if warmup.agent_name not in BLUE_AGENTS:
                raise ValueError(f"unsupported agent: {warmup.agent_name}")

            decision_index = item["decision_index"]
            global_tick_start = item["global_tick_start"]
            if (
                isinstance(decision_index, bool)
                or not isinstance(decision_index, int)
                or decision_index < 0
            ):
                raise ValueError("decision_index must be non-negative int")
            if (
                isinstance(global_tick_start, bool)
                or not isinstance(global_tick_start, int)
                or global_tick_start < 0
            ):
                raise ValueError("global_tick_start must be non-negative int")

            records.append({
                "warmup": warmup,
                "decision_index": int(decision_index),
                "global_tick_start": int(global_tick_start),
            })

    if not records:
        raise ValueError("calibration state file is empty")
    return records


def select_balanced_warmup_states(
    records: list[dict],
    *,
    agent_name: str,
    planner_calls: int = PLANNER_CALLS_PER_AGENT,
) -> list[WarmupState]:
    if agent_name not in BLUE_AGENTS:
        raise ValueError(f"unsupported agent: {agent_name}")
    if (
        isinstance(planner_calls, bool)
        or not isinstance(planner_calls, int)
        or planner_calls <= 0
    ):
        raise ValueError("planner_calls must be positive int")

    by_seed: dict[int, list[dict]] = {seed: [] for seed in CALIBRATION_SEED_ORDER}
    for item in records:
        warmup = item["warmup"]
        if warmup.agent_name != agent_name:
            continue
        by_seed[int(warmup.episode_seed)].append(item)

    missing = [seed for seed, rows in by_seed.items() if not rows]
    if missing:
        raise ValueError(f"{agent_name}: missing calibration seeds {missing}")

    for seed in CALIBRATION_SEED_ORDER:
        by_seed[seed].sort(
            key=lambda item: (
                item["decision_index"],
                item["global_tick_start"],
            )
        )

    cursors = {seed: 0 for seed in CALIBRATION_SEED_ORDER}
    selected: list[WarmupState] = []

    while len(selected) < planner_calls:
        progressed = False
        for seed in CALIBRATION_SEED_ORDER:
            cursor = cursors[seed]
            rows = by_seed[seed]
            if cursor >= len(rows):
                continue
            selected.append(rows[cursor]["warmup"])
            cursors[seed] = cursor + 1
            progressed = True
            if len(selected) >= planner_calls:
                break
        if not progressed:
            raise ValueError(
                f"{agent_name}: only {len(selected)} calibration states "
                f"available, need {planner_calls}"
            )

    return selected


def calibration_cem_config(*, agent_index: int, device: str) -> CategoricalCEMConfig:
    return CategoricalCEMConfig(
        horizon=4,
        n_actions=4,
        population_size=CEM_POPULATION_SIZE,
        num_iterations=CEM_NUM_ITERATIONS,
        elite_ratio=CEM_ELITE_RATIO,
        alpha=CEM_ALPHA,
        prob_floor=CEM_PROB_FLOOR,
        seed=CEM_SEED_BASE + int(agent_index),
        device=device,
    )


def build_planner(
    *,
    world_model: BootstrapProbabilisticWorldModel,
    reward_predictor: ResponseRewardPredictor,
    agent_index: int,
    device: str,
) -> UGCEMPlanner:
    cem = CategoricalCEMOptimizer(
        calibration_cem_config(agent_index=agent_index, device=device)
    )
    rollout = SharedRolloutEvaluator(
        world_model=world_model,
        reward_predictor=reward_predictor,
        config=SharedRolloutConfig(horizon=4, gamma_tick=0.99),
    )
    uncertainty = UGUncertainty(
        UGUncertaintyConfig(
            alpha=UG_ALPHA,
            eps=UG_EPS,
            device=device,
        )
    )
    return UGCEMPlanner(
        cem_optimizer=cem,
        rollout_evaluator=rollout,
        uncertainty=uncertainty,
        config=UGCEMPlannerConfig(
            beta=CALIBRATION_BETA,
            update_uncertainty_stats=True,
        ),
    )


def tensor_summary(tensor: torch.Tensor) -> dict:
    value = tensor.detach().cpu().to(dtype=torch.float32)
    if not torch.isfinite(value).all():
        raise ValueError("normalizer tensor contains NaN/Inf")
    return {
        "shape": list(value.shape),
        "min": float(value.min().item()),
        "max": float(value.max().item()),
        "mean": float(value.mean().item()),
        "values": value.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run formal Step-6 per-agent UG normalizer calibration."
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--states", default=DEFAULT_STATES)
    parser.add_argument("--world-model", default=DEFAULT_WORLD_MODEL)
    parser.add_argument("--reward-model", default=DEFAULT_REWARD_MODEL)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--freeze-after-warmup",
        action="store_true",
        help="Sensitivity mode only; formal main mode leaves this unset.",
    )
    args = parser.parse_args()

    device = str(args.device)
    records = load_calibration_records(args.states)

    world_model_path = resolve_project_path(args.world_model)
    reward_model_path = resolve_project_path(args.reward_model)

    world_model = BootstrapProbabilisticWorldModel.load_checkpoint(
        world_model_path,
        device=device,
    )
    reward_predictor = ResponseRewardPredictor.load_checkpoint(
        reward_model_path,
        device=device,
    )

    if int(world_model.config.state_dim) != FORMAL_STATE_DIM:
        raise RuntimeError("world model state dimension is not formal D=27")
    if int(world_model.config.n_actions) != 4:
        raise RuntimeError("world model action dimension is not A=4")
    if int(world_model.config.ensemble_size) != 5:
        raise RuntimeError("world model ensemble is not M=5")
    if str(world_model.config.target_mode) != "absolute":
        raise RuntimeError("selected formal world model must be absolute")

    out_dir = resolve_project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    agent_payload = {}
    agent_report = {}

    for agent_index, agent_name in enumerate(BLUE_AGENTS):
        print("=" * 80)
        print("[STEP6 WARMUP]", agent_name)

        selected = select_balanced_warmup_states(
            records,
            agent_name=agent_name,
            planner_calls=PLANNER_CALLS_PER_AGENT,
        )

        planner = build_planner(
            world_model=world_model,
            reward_predictor=reward_predictor,
            agent_index=agent_index,
            device=device,
        )

        runner = UGNormalizerWarmup(
            planner=planner,
            config=UGNormalizerWarmupConfig(
                planner_calls=PLANNER_CALLS_PER_AGENT,
                freeze_after_warmup=bool(args.freeze_after_warmup),
            ),
        )
        report = runner.run(selected)

        used_seeds = tuple(sorted(set(report.unique_episode_seeds)))
        if used_seeds != CALIBRATION_SEED_ORDER:
            raise RuntimeError(
                f"{agent_name}: warm-up did not cover all calibration seeds: "
                f"{used_seeds}"
            )
        if report.train_calls != 0:
            raise RuntimeError("formal main calibration must not consume train records")
        if report.calibration_calls != PLANNER_CALLS_PER_AGENT:
            raise RuntimeError("calibration call count mismatch")
        if not report.all_warm_starts_disabled:
            raise RuntimeError("warm-up used previous solution")
        if not report.finite:
            raise RuntimeError("normalizer report is non-finite")

        snapshot = report.snapshot
        agent_payload[agent_name] = {
            "obs_mean": snapshot.obs_mean.detach().cpu(),
            "obs_std": snapshot.obs_std.detach().cpu(),
            "horizon_std": snapshot.horizon_std.detach().cpu(),
        }

        agent_report[agent_name] = {
            "planner_calls": int(report.planner_calls),
            "calibration_calls": int(report.calibration_calls),
            "unique_episode_seeds": list(used_seeds),
            "all_warm_starts_disabled": bool(report.all_warm_starts_disabled),
            "finite": bool(report.finite),
            "freeze_after_warmup": bool(report.freeze_after_warmup),
            "online_updates_after_warmup": bool(report.online_updates_after_warmup),
            "cem_seed": CEM_SEED_BASE + agent_index,
            "obs_mean": tensor_summary(snapshot.obs_mean),
            "obs_std": tensor_summary(snapshot.obs_std),
            "horizon_std": tensor_summary(snapshot.horizon_std),
        }

        print("  calls:", report.planner_calls)
        print("  seeds:", used_seeds)
        print("  finite:", report.finite)
        print("  online_updates_after_warmup:", report.online_updates_after_warmup)

    bundle = {
        "format_version": 1,
        "source_split": "calibration",
        "calibration_seeds": CALIBRATION_SEED_ORDER,
        "planner_calls_per_agent": PLANNER_CALLS_PER_AGENT,
        "freeze_after_warmup": bool(args.freeze_after_warmup),
        "online_updates_after_warmup": not bool(args.freeze_after_warmup),
        "world_model_checkpoint": str(world_model_path),
        "reward_model_checkpoint": str(reward_model_path),
        "planner_profile": {
            "horizon": 4,
            "n_actions": 4,
            "population_size": CEM_POPULATION_SIZE,
            "num_iterations": CEM_NUM_ITERATIONS,
            "elite_ratio": CEM_ELITE_RATIO,
            "alpha": CEM_ALPHA,
            "prob_floor": CEM_PROB_FLOOR,
            "beta": CALIBRATION_BETA,
            "ug_alpha": UG_ALPHA,
            "ug_eps": UG_EPS,
        },
        "agents": agent_payload,
    }

    overall_pass = (
        set(agent_payload.keys()) == set(BLUE_AGENTS)
        and all(item["finite"] for item in agent_report.values())
        and all(item["all_warm_starts_disabled"] for item in agent_report.values())
    )

    json_report = {
        "status": "PASS" if overall_pass else "FAIL",
        "source_split": "calibration",
        "calibration_seeds": list(CALIBRATION_SEED_ORDER),
        "planner_calls_per_agent": PLANNER_CALLS_PER_AGENT,
        "total_planner_calls": PLANNER_CALLS_PER_AGENT * len(BLUE_AGENTS),
        "freeze_after_warmup": bool(args.freeze_after_warmup),
        "online_updates_after_warmup": not bool(args.freeze_after_warmup),
        "world_model_checkpoint": str(world_model_path),
        "reward_model_checkpoint": str(reward_model_path),
        "planner_profile": bundle["planner_profile"],
        "agents": agent_report,
        "pass": bool(overall_pass),
    }

    suffix = "frozen" if args.freeze_after_warmup else "online"
    bundle_path = out_dir / f"ug_normalizers_{suffix}.pt"
    report_path = out_dir / f"ug_normalizers_{suffix}_report.json"

    torch.save(bundle, bundle_path)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(json_report, handle, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("[STEP6 SUMMARY]")
    print("agents:", len(agent_payload))
    print("calls_per_agent:", PLANNER_CALLS_PER_AGENT)
    print("total_calls:", json_report["total_planner_calls"])
    print("freeze_after_warmup:", json_report["freeze_after_warmup"])
    print("online_updates_after_warmup:", json_report["online_updates_after_warmup"])
    print("pass:", overall_pass)
    print("[OK] bundle:", bundle_path)
    print("[OK] report:", report_path)

    if not overall_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
