# experiments/compare_deploys.py
import os
import sys
import json
import argparse
from collections import defaultdict, Counter

import numpy as np


def load_jsonl_group_by_episode(path: str):
    eps = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            eps[int(obj["episode"])].append(obj)
    # sort by t
    for k in list(eps.keys()):
        eps[k] = sorted(eps[k], key=lambda x: int(x.get("t", 0)))
    return eps


def eval_one(jsonl_path: str, early_k: int = 20):
    eps = load_jsonl_group_by_episode(jsonl_path)

    ep_metrics = []
    all_actions = Counter()
    all_plans = Counter()

    for ep, rows in eps.items():
        rewards = np.array([float(r["reward"]) for r in rows], dtype=np.float32)
        actions = np.array([int(r["action_id"]) for r in rows], dtype=np.int64)
        plans = [int(r.get("choose_plan", -999)) for r in rows]
        steps = len(rows)
        ret = float(rewards.sum()) if steps > 0 else 0.0

        nz = np.where(actions != 0)[0]
        t_first_nz = int(nz[0]) if len(nz) > 0 else None
        nonzero_ratio = float(np.mean(actions != 0)) if steps > 0 else 0.0
        switches = int(np.sum(actions[1:] != actions[:-1])) if steps >= 2 else 0

        k = min(int(early_k), steps)
        if k > 0:
            rewards_k = rewards[:k]
            rmean_k = float(rewards_k.mean())
            rret_k = float(rewards_k.sum())
        else:
            rmean_k, rret_k = 0.0, 0.0

        all_actions.update(list(actions))
        all_plans.update(plans)

        ep_metrics.append({
            "steps": steps,
            "return": ret,
            "reward_mean": float(rewards.mean()) if steps > 0 else 0.0,
            "reward_min": float(rewards.min()) if steps > 0 else 0.0,
            "reward_max": float(rewards.max()) if steps > 0 else 0.0,
            "t_first_nonzero_action": t_first_nz,
            "nonzero_action_ratio": nonzero_ratio,
            "action_switches": switches,
            "reward_mean_first_k": rmean_k,
            "return_first_k": rret_k,
        })

    # aggregate
    if len(ep_metrics) == 0:
        return {"episodes": 0}

    def mean_ignore_none(vals):
        v = [x for x in vals if x is not None]
        if len(v) == 0:
            return None
        return float(np.mean(v))

    out = {
        "episodes": len(ep_metrics),
        "steps_mean": float(np.mean([m["steps"] for m in ep_metrics])),
        "return_mean": float(np.mean([m["return"] for m in ep_metrics])),
        "reward_mean": float(np.mean([m["reward_mean"] for m in ep_metrics])),
        "reward_mean_first_k": float(np.mean([m["reward_mean_first_k"] for m in ep_metrics])),
        "t_first_nonzero_action_mean": mean_ignore_none([m["t_first_nonzero_action"] for m in ep_metrics]),
        "nonzero_action_ratio_mean": float(np.mean([m["nonzero_action_ratio"] for m in ep_metrics])),
        "action_switches_mean": float(np.mean([m["action_switches"] for m in ep_metrics])),
        "top_actions": all_actions.most_common(5),
        "top_choose_plan": all_plans.most_common(6),
        "jsonl": jsonl_path
    }
    return out


def fmt(x, w=10):
    if x is None:
        return ("-" * (w-1) + " ").rjust(w)
    if isinstance(x, float):
        return f"{x:.4f}".rjust(w)
    return str(x).rjust(w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonls", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=False)
    ap.add_argument("--early_k", type=int, default=20)
    args = ap.parse_args()

    labels = args.labels
    if labels is None or len(labels) != len(args.jsonls):
        labels = [os.path.basename(p).replace(".jsonl", "") for p in args.jsonls]

    results = []
    for lab, p in zip(labels, args.jsonls):
        r = eval_one(p, early_k=args.early_k)
        r["label"] = lab
        results.append(r)

    # print table
    print("\n=== DEPLOY COMPARISON (higher is better for return/reward; lower is 'earlier' for t_first_nonzero_action) ===")
    header = [
        "label", "episodes", "return_mean", "reward_mean",
        f"reward_mean@first{args.early_k}",
        "t_first_nz", "nz_ratio", "switches"
    ]
    print(" | ".join([h.ljust(18) for h in header]))
    print("-" * (18 * len(header)))

    for r in results:
        print(" | ".join([
            str(r["label"]).ljust(18),
            fmt(r.get("episodes", 0), 18),
            fmt(r.get("return_mean", None), 18),
            fmt(r.get("reward_mean", None), 18),
            fmt(r.get("reward_mean_first_k", None), 18),
            fmt(r.get("t_first_nonzero_action_mean", None), 18),
            fmt(r.get("nonzero_action_ratio_mean", None), 18),
            fmt(r.get("action_switches_mean", None), 18),
        ]))

    print("\n[TOP ACTIONS]")
    for r in results:
        print(f"- {r['label']}: {r.get('top_actions', [])}")

    print("\n[TOP CHOOSE_PLAN]")
    for r in results:
        print(f"- {r['label']}: {r.get('top_choose_plan', [])}")


if __name__ == "__main__":
    main()
