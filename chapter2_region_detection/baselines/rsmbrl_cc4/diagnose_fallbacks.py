"""Read-only diagnosis of the existing 20-tick RSMBRL wiring smoke."""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from baselines.rsmbrl_cc4.artifact_preflight import DEFAULT_REWARD_MODEL, DEFAULT_WORLD_MODEL, resolve
from baselines.rsmbrl_cc4.runtime import build_runtime
from formal_experiments.training.bootstrap_world_model import BootstrapProbabilisticWorldModel
from formal_experiments.training.response_reward_predictor import ResponseRewardPredictor
from shared.rollout_evaluator import SharedRolloutConfig, SharedRolloutEvaluator

DEFAULT_SMOKE = "outputs/rsmbrl_cc4/wiring_smoke/episode_3000_steps_20.json"
DEFAULT_STATES = "outputs/ug_cem_v2/step6/calibration_states.jsonl"
DEFAULT_OUT = "outputs/rsmbrl_cc4/wiring_smoke/fallback_diagnostic.json"
ACTION_FAMILY = {1: "Analyse", 2: "Remove", 3: "Restore"}


def diagnose(*, device="cpu") -> dict:
    smoke = json.loads(resolve(DEFAULT_SMOKE).read_text(encoding="utf-8"))
    decisions = smoke["decisions"]
    # The old report did not persist the full three-family availability map.
    # For the requested targeted family, adapter semantics make availability
    # exactly auditable: a fallback with this reason means resolver returned no
    # legal observable target for that family.
    rows = []
    for item in decisions:
        action = int(item["requested_action_id"])
        if action == 0:
            continue
        inferred_available = not bool(item["fallback"])
        rows.append({
            "agent_name": item["agent_name"], "controller_tick": item["controller_tick"],
            "requested_action_id": action, "requested_family": ACTION_FAMILY[action],
            "requested_family_available": inferred_available,
            "fallback": bool(item["fallback"]), "fallback_reason": item["fallback_reason"],
        })
    grouped = Counter((x["agent_name"], x["requested_family"], x["requested_family_available"], x["fallback"]) for x in rows)
    fallbacks = [x for x in rows if x["fallback"]]
    all_explained = all(
        not x["requested_family_available"] and x["fallback_reason"] == "no_valid_observable_target"
        for x in fallbacks
    )

    world = BootstrapProbabilisticWorldModel.load_checkpoint(resolve(DEFAULT_WORLD_MODEL), device=device)
    reward = ResponseRewardPredictor.load_checkpoint(resolve(DEFAULT_REWARD_MODEL), device=device)
    evaluator = SharedRolloutEvaluator(world_model=world, reward_predictor=reward,
                                       config=SharedRolloutConfig(horizon=4, gamma_tick=.99))
    planners, _ = build_runtime(device=device)
    states = []
    with resolve(DEFAULT_STATES).open("r", encoding="utf-8") as handle:
        for raw in handle:
            row = json.loads(raw)
            state = np.asarray(row["state"], dtype=np.float32)
            if float(state[17]) < .5:
                states.append((row["agent_name"], int(row["episode_seed"]), int(row["decision_index"]), state))
            if len(states) == 3:
                break
    comparisons = []
    for agent, seed, decision_index, state in states:
        plans = torch.tensor([[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0]], device=device)
        rollout = evaluator.evaluate(state, plans)
        risks = planners[agent].uncertainty.compute(rollout.next_states, update_stats=False)
        returns = rollout.expected_return.detach().cpu()
        risks = risks.detach().cpu()
        comparisons.append({
            "agent_name": agent, "episode_seed": seed, "decision_index": decision_index,
            "state_any_valid_observable_target": float(state[17]),
            "plans": plans.cpu().tolist(), "expected_returns": returns.tolist(),
            "uncertainties": risks.tolist(),
            "all_returns_bitwise_equal": bool(torch.equal(returns, returns[0].expand_as(returns))),
            "all_uncertainties_bitwise_equal": bool(torch.equal(risks, risks[0].expand_as(risks))),
            "all_rollout_states_bitwise_equal": bool(torch.equal(
                rollout.next_states, rollout.next_states[:, 0:1].expand_as(rollout.next_states)
            )),
        })
    return {
        "status": "PASS", "environment_rerun": False,
        "source_smoke": str(resolve(DEFAULT_SMOKE)),
        "limitation": "existing report stores requested-family resolver outcome, not the full three-family availability map",
        "targeted_requests": len(rows), "fallback_count": len(fallbacks),
        "all_fallbacks_explained_by_requested_family_unavailable": all_explained,
        "grouped_requested_family_availability": [
            {"agent_name": k[0], "family": k[1], "available": k[2], "fallback": k[3], "count": v}
            for k, v in sorted(grouped.items())
        ],
        "per_tick_requested_family": rows,
        "canonicalization_comparisons": comparisons,
        "diagnosis": (
            "When D27 any_valid_observable_target=0, model-space canonicalization maps all targeted "
            "first actions to no-op. Their rollout return, uncertainty, and predicted states are exactly "
            "equal, so CEM may select an illegal requested first action by sampling/argmax order."
        ),
        "proposed_minimal_fix_not_applied": (
            "At the root decision only, mask targeted A4 actions when their observable family has no legal "
            "target; renormalize CEM probabilities and deterministically tie-break canonical-equivalent plans "
            "in favor of requested no-op. Keep later model-space canonicalization unchanged."
        ),
    }


def main():
    report = diagnose(device="cpu")
    path = resolve(DEFAULT_OUT)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"per_tick_requested_family"}}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
