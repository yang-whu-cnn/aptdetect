import json, sys, math
from collections import Counter, defaultdict

p = sys.argv[1]
ACTION_KEY_CANDS = ["action", "action_id"]
EP_KEY_CANDS = ["ep", "episode"]
T_KEY_CANDS = ["t", "step"]

def pick(d, cands, default=None):
    for k in cands:
        if k in d:
            return d[k]
    return default

actions = Counter()
plans = Counter()
ep_ret = defaultdict(float)
ep_steps = Counter()
nan_cnt = 0
missing_cnt = 0
neg_big = []  # reward < -1

with open(p, "r", encoding="utf-8") as f:
    for line in f:
        line=line.strip()
        if not line:
            continue
        j = json.loads(line)
        ep = pick(j, EP_KEY_CANDS, None)
        t  = pick(j, T_KEY_CANDS, None)
        r  = j.get("reward", None)
        cp = j.get("choose_plan", None)
        a  = pick(j, ACTION_KEY_CANDS, None)
        info = j.get("info", {}) if isinstance(j.get("info", {}), dict) else {}

        if ep is None or t is None or r is None or a is None or cp is None:
            missing_cnt += 1
            continue

        if isinstance(r, float) and (math.isnan(r) or math.isinf(r)):
            nan_cnt += 1

        actions[int(a)] += 1
        plans[int(cp)] += 1
        ep_ret[int(ep)] += float(r)
        ep_steps[int(ep)] += 1

        if float(r) < -1:
            neg_big.append({
                "ep": int(ep), "t": int(t), "r": float(r),
                "plan": int(cp), "a": int(a),
                "fp_risk": info.get("fp_risk", None),
                "act_cost": info.get("act_cost", None),
                "early_gain": info.get("early_gain", None),
                "risk_for_reward": info.get("risk_for_reward", None),
                "src_file": info.get("src_file", None),
                "orig_step": info.get("orig_step", None),
                "zone": info.get("zone", None),
            })

eps = list(ep_ret.keys())
mean_ret = sum(ep_ret.values()) / max(1, len(ep_ret))

print("file:", p)
print("episodes:", len(ep_ret), "lines:", sum(ep_steps.values()))
print("return mean:", mean_ret, "min:", (min(ep_ret.values()) if ep_ret else None), "max:", (max(ep_ret.values()) if ep_ret else None))
print("top actions:", actions.most_common(8))
print("top plans:", plans.most_common(8))
print("missing_cnt:", missing_cnt, "nan_cnt:", nan_cnt)
print("neg_big_cnt:", len(neg_big))
print("neg_big_head:", neg_big[:5])
