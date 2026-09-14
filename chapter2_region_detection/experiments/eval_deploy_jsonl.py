# experiments/eval_deploy_jsonl.py
# 读取 deploy_region*.jsonl，汇总本地 online 的“检测效果”指标（无需真标签）

import argparse
import json
from collections import Counter, defaultdict
from statistics import mean


def load_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def group_by_episode(rows):
    eps = defaultdict(list)
    for r in rows:
        eps[int(r.get("episode", 0))].append(r)
    # 按 t 排序
    for k in eps:
        eps[k].sort(key=lambda x: int(x.get("t", 0)))
    return eps


def first_nonzero_action_t(ep_rows):
    for r in ep_rows:
        a = int(r.get("action_id", 0))
        if a != 0:
            return int(r.get("t", 0))
    return None


def window_stats(ep_rows, k=20):
    rs = [float(r.get("reward", 0.0)) for r in ep_rows[:k]]
    if not rs:
        return {"k": 0}
    return {
        "k": len(rs),
        "reward_mean_first_k": mean(rs),
        "reward_min_first_k": min(rs),
        "reward_max_first_k": max(rs),
        "return_first_k": sum(rs),
    }


def eval_episode(ep_rows, early_k=20):
    rewards = [float(r.get("reward", 0.0)) for r in ep_rows]
    actions = [int(r.get("action_id", 0)) for r in ep_rows]
    plans = [int(r.get("choose_plan", -1)) for r in ep_rows if "choose_plan" in r]

    total_return = sum(rewards) if rewards else 0.0
    mean_reward = mean(rewards) if rewards else 0.0
    min_reward = min(rewards) if rewards else 0.0
    max_reward = max(rewards) if rewards else 0.0
    steps = len(ep_rows)

    act_cnt = Counter(actions)
    plan_cnt = Counter(plans) if plans else Counter()

    # 反应时延：首次采取非 noop（action_id != 0）的时刻
    t_first_act = first_nonzero_action_t(ep_rows)

    # 动作切换次数（从一个动作变到另一个动作），越频繁可能越“抖动/不稳定”
    switches = 0
    for i in range(1, len(actions)):
        if actions[i] != actions[i - 1]:
            switches += 1

    # 处置动作占比（非0占比）
    nonzero_ratio = (steps - act_cnt.get(0, 0)) / steps if steps > 0 else 0.0

    # 早期窗口统计：看前K步是否更“早/更稳”
    w = window_stats(ep_rows, k=early_k)

    out = {
        "steps": steps,
        "return": total_return,
        "reward_mean": mean_reward,
        "reward_min": min_reward,
        "reward_max": max_reward,
        "t_first_nonzero_action": t_first_act,  # None 表示全程 noop
        "nonzero_action_ratio": nonzero_ratio,
        "action_switches": switches,
        "top_actions": act_cnt.most_common(5),
        "unique_actions": len(act_cnt),
        "unique_choose_plan": len(plan_cnt) if plan_cnt else 0,
        "top_choose_plan": plan_cnt.most_common(5) if plan_cnt else [],
        **w
    }
    return out


def summarize(all_ep_stats):
    if not all_ep_stats:
        return {}

    returns = [s["return"] for s in all_ep_stats]
    means = [s["reward_mean"] for s in all_ep_stats]
    firsts = [s["t_first_nonzero_action"] for s in all_ep_stats if s["t_first_nonzero_action"] is not None]
    nonzero = [s["nonzero_action_ratio"] for s in all_ep_stats]
    switches = [s["action_switches"] for s in all_ep_stats]
    early_r = [s.get("reward_mean_first_k", 0.0) for s in all_ep_stats]

    # overall 动作统计聚合
    act_cnt_all = Counter()
    plan_cnt_all = Counter()
    for s in all_ep_stats:
        for a, c in s["top_actions"]:
            act_cnt_all[a] += c
        for p, c in s.get("top_choose_plan", []):
            plan_cnt_all[p] += c

    return {
        "episodes": len(all_ep_stats),
        "return_mean": mean(returns),
        "return_min": min(returns),
        "return_max": max(returns),
        "reward_mean": mean(means),
        "reward_mean_first_k": mean(early_r),
        "t_first_nonzero_action_mean": (mean(firsts) if firsts else None),
        "nonzero_action_ratio_mean": mean(nonzero),
        "action_switches_mean": mean(switches),
        "top_actions_overall": act_cnt_all.most_common(8),
        "top_choose_plan_overall": plan_cnt_all.most_common(8),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True,
                    help="deploy jsonl path, e.g. outputs/local_online_deploy/deploy_region0.jsonl")
    ap.add_argument("--early_k", type=int, default=20, help="early window steps for early-effect proxy")
    ap.add_argument("--print_each", action="store_true", help="print each episode stats")
    args = ap.parse_args()

    rows = load_jsonl(args.jsonl)
    eps = group_by_episode(rows)

    all_stats = []
    for ep_id in sorted(eps.keys()):
        st = eval_episode(eps[ep_id], early_k=args.early_k)
        all_stats.append(st)
        if args.print_each:
            print(f"[EP={ep_id:02d}] {st}")

    overall = summarize(all_stats)
    # CC4-style（按episode return统计 mean/std）
    ep_returns = [float(st.get("return", 0.0)) for st in all_stats]
    if ep_returns:
        import numpy as _np
        _arr = _np.asarray(ep_returns, dtype=_np.float64)
        _steps = [len(ep_rows) for ep_rows in eps.values()]
        _steps_unique = sorted(set(_steps))
        print("\n[CC4-STYLE EVAL]")
        print("  episodes:", int(len(_arr)))
        print("  steps_per_episode:", _steps_unique[0] if len(_steps_unique)==1 else _steps_unique)
        print("  mean_reward:", float(_arr.mean()))
        print("  std_reward:", float(_arr.std(ddof=0)))

    print("\n[OVERALL]")
    for k, v in overall.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
