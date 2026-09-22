"""Development-only fake/readiness smoke for Table-2 PPO variants."""

from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np

from formal_experiments.ours.ppo_core import compute_duration_aware_gae
from formal_experiments.ours.table2_variants import Table2DecisionRuntime, Table2Variant


def run_fake_smoke(runtime: Table2DecisionRuntime, *, steps: int = 20) -> dict:
    if steps != 20: raise ValueError("development readiness smoke is frozen to 20 transitions")
    actions = []; durations = []
    duration_map = (1, 2, 3, 5)
    for tick in range(steps):
        state = np.zeros(27, dtype=np.float32); state[tick % 27] = tick / steps
        decision = runtime.decide(state, agent_name="blue_agent_0", action_mask=[True, True, tick % 3 != 0, True])
        action = int(decision["requested_action"]); actions.append(action); durations.append(duration_map[action])
    gae = compute_duration_aware_gae(
        np.zeros(steps), np.zeros(steps), np.zeros(steps), [False] * (steps - 1) + [True], durations
    )
    return {"development_only": True, "formal_result_eligible": False, "fake_environment": True,
            "variant": runtime.spec.variant.value, "transitions": steps, "actions": actions,
            "durations": durations, "discounts": gae.discounts.tolist(), "module_audit": vars(runtime.audit),
            "blocked": False}


def readiness(variant: Table2Variant, *, cache_ready: bool, wm_ready: bool) -> dict:
    spec = __import__("formal_experiments.ours.table2_variants", fromlist=["SPECS"]).SPECS[variant]
    blockers = []
    if spec.uses_llm_prior and not cache_ready: blockers.append("offline_prior_cache_not_ready")
    if spec.uses_world_model and not wm_ready: blockers.append("frozen_wm_or_full_reward_not_ready")
    return {"variant": variant.value, "status": "BLOCKED" if blockers else "READY",
            "blockers": blockers, "formal_result_eligible": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=[x.value for x in Table2Variant], required=True)
    parser.add_argument("--cache-ready", action="store_true"); parser.add_argument("--wm-ready", action="store_true")
    parser.add_argument("--out", type=Path, required=True); args = parser.parse_args()
    report = readiness(Table2Variant(args.variant), cache_ready=args.cache_ready, wm_ready=args.wm_ready)
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
