# experiments/run_deploy_baseline.py
# Baselines for local-online CC4 deploy:
#   - noop   : always action_id=0
#   - random : random action_id in [0, n_actions)
#
# IMPORTANT:
#   Always run as module from project root:
#     python -m experiments.run_deploy_baseline ...
#   or set PowerShell env:
#     $env:PYTHONPATH="."

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# --- Make sure "src" is importable no matter how you run it ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import load_yaml, set_seed
from src.env_region import build_region_objects
from src.action_space import ACTION_SPACE


def _extract_risk_proxy(obs: Dict[str, Any]) -> float:
    """Try best to extract risk proxy from different obs formats."""
    if obs is None:
        return 0.0
    if "risk_proxy" in obs and obs["risk_proxy"] is not None:
        try:
            return float(obs["risk_proxy"])
        except Exception:
            return 0.0
    # Some obs may store stats dict
    stats = obs.get("stats") or obs.get("raw_stats") or {}
    if isinstance(stats, dict) and "risk_proxy" in stats:
        try:
            return float(stats["risk_proxy"])
        except Exception:
            return 0.0
    return 0.0


def _summarize_observation(summarizer: Any, obs: Dict[str, Any]) -> Any:
    """
    Your SlidingWindowSummarizer API is:
      - reset()
      - update(events)
      - build(risk_proxy) -> StateSummary
    """
    events = obs.get("events", [])
    if hasattr(summarizer, "update"):
        summarizer.update(events)
    risk_proxy = _extract_risk_proxy(obs)
    if hasattr(summarizer, "build"):
        return summarizer.build(risk_proxy=risk_proxy)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, required=True)
    ap.add_argument("--region_id", type=int, required=True)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy", type=str, choices=["noop", "random"], required=True)
    ap.add_argument("--out_jsonl", type=str, required=True)
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--cc4_eval", action="store_true", help="按CC4风格评测：100个episode、每个500步")
    ap.add_argument("--cc4_n_episodes", type=int, default=100)
    ap.add_argument("--cc4_episode_steps", type=int, default=500)
    args = ap.parse_args()

    if args.cc4_eval:
        args.episodes = int(args.cc4_n_episodes)
        args.steps = int(args.cc4_episode_steps)

    cfg = load_yaml(args.cfg)

    # seed
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # Build region objects (client + summarizer etc.)
    objs = build_region_objects(cfg, region_id=args.region_id, device=args.device)
    client = objs.client
    summarizer = objs.summarizer

    if hasattr(client, "max_steps"):
        try:
            client.max_steps = int(args.steps)
        except Exception:
            pass

    n_actions = len(ACTION_SPACE)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fmode = "a" if args.append else "w"
    total_lines = 0
    episode_returns = []

    with open(out_path, fmode, encoding="utf-8") as f:
        for ep in range(args.episodes):
            # reset
            if hasattr(summarizer, "reset"):
                summarizer.reset()

            try:
                obs = client.reset(seed=args.seed + ep)
            except TypeError:
                obs = client.reset()
            _ = _summarize_observation(summarizer, obs)

            done = False
            ep_return = 0.0
            action_hist = []

            for t in range(int(args.steps)):
                if args.policy == "noop":
                    action_id = 0
                else:
                    action_id = int(rng.integers(0, n_actions))

                obs2, reward, done, info = client.step(action_id)
                ep_return += float(reward)
                action_hist.append(action_id)

                record = {
                    "episode": ep,
                    "t": t,
                    "policy": args.policy,
                    "choose_plan": -1,          # keep compatible with your deploy jsonl
                    "action_id": int(action_id),
                    "reward": float(reward),
                    "done": bool(done),
                    "info": info if isinstance(info, dict) else {"info": str(info)},
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_lines += 1

                obs = obs2
                _ = _summarize_observation(summarizer, obs)

                if done:
                    break

            episode_returns.append(float(ep_return))

            if not args.quiet:
                unique_actions = len(set(action_hist))
                top_actions = sorted([(a, action_hist.count(a)) for a in set(action_hist)],
                                     key=lambda x: -x[1])[:5]
                print(f"[EP={ep:02d}] policy={args.policy} steps={len(action_hist)} "
                      f"return={ep_return:.3f} unique_actions={unique_actions} top5={top_actions}")

    if episode_returns:
        arr = np.asarray(episode_returns, dtype=np.float64)
        print(f"[CC4-STYLE] episodes={len(arr)} steps_per_episode={int(args.steps)} mean_reward={arr.mean():.6f} std_reward={arr.std(ddof=0):.6f}")

    print(f"[DONE] wrote jsonl: {out_path} lines={total_lines}")


if __name__ == "__main__":
    main()
