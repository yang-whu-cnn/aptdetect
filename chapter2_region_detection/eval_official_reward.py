import json, statistics
from collections import defaultdict

PATH = r"outputs\local_online_deploy\deploy_region0.jsonl"
CAND_KEYS = ["official_reward", "blue_reward", "env_reward", "reward"]

eps = defaultdict(list)
with open(PATH, "r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        ep = int(r.get("episode", 0))
        rew = None
        for k in CAND_KEYS:
            if k in r:
                rew = r[k]
                break
        if rew is None:
            raise KeyError(f"No reward key found. Tried {CAND_KEYS}. keys={list(r.keys())[:30]}")
        eps[ep].append(float(rew))

ep_returns = [sum(v) for _, v in sorted(eps.items())]
mean_ret = statistics.mean(ep_returns)
std_ret = statistics.stdev(ep_returns) if len(ep_returns) > 1 else 0.0
steps_avg = sum(len(v) for v in eps.values()) / max(1, len(ep_returns))

print("[OFFICIAL-STYLE EVAL]")
print("  episodes:", len(ep_returns))
print("  steps_per_episode(avg):", steps_avg)
print("  mean_reward:", mean_ret)
print("  std_reward:", std_ret)
