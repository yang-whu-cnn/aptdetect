"""Development-only fail-closed probe for on-policy prototype coverage."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .cc4_callback import run_cc4_episode
from .policy import PriorRLActorCritic
from .prototype_retrieval import FrozenPrototypeRetriever, PrototypeLookupMiss
from .training import A4PPOConfig


def probe(*, artifact: str, output: str, episode_seed: int = 1000,
          policy_seed: int = 61001, ticks: int = 100, device: str | None = None) -> dict:
    run_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(policy_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(policy_seed)
    policy = PriorRLActorCritic().to(run_device)
    retriever = FrozenPrototypeRetriever.load(artifact)
    try:
        run_cc4_episode(
            episode_seed=episode_seed, policy_seed=policy_seed, policy=policy,
            retriever=retriever, episode_ticks=ticks, training=True,
            rollout_threshold=128, ppo_config=A4PPOConfig(),
        )
    except PrototypeLookupMiss as exc:
        report = {
            "schema": "priorrl_online_prototype_coverage_probe_v1",
            "status": "BLOCKED",
            "formal_result_eligible": False,
            "provider_calls": 0,
            "episode_seed": episode_seed,
            "policy_seed": policy_seed,
            "ticks_requested": ticks,
            "agent_name": exc.agent_name,
            "state": exc.state.tolist(),
            "state_sha256": exc.state_sha256,
            "nearest_distance": exc.distance,
            "frozen_radius": exc.radius,
            "prototype_sha256": retriever.payload["prototype_sha256"],
        }
    else:
        report = {
            "schema": "priorrl_online_prototype_coverage_probe_v1",
            "status": "PASS",
            "formal_result_eligible": False,
            "provider_calls": 0,
            "episode_seed": episode_seed,
            "policy_seed": policy_seed,
            "ticks_requested": ticks,
            "prototype_sha256": retriever.payload["prototype_sha256"],
        }
    report["report_sha256"] = hashlib.sha256(json.dumps(
        report, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    path = Path(output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", default="outputs/priorrl_cc4/prototypes/frozen_prototypes.json")
    parser.add_argument("--output", default="outputs/priorrl_cc4/prototypes/online_coverage_probe.json")
    parser.add_argument("--episode-seed", type=int, default=1000)
    parser.add_argument("--policy-seed", type=int, default=61001)
    parser.add_argument("--ticks", type=int, default=100)
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    report = probe(**vars(args))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
