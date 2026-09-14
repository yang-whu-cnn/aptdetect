# experiments/run_deploy_region.py
# 作用：部署推理（单区域）
# - 每步：用 LLM 生成候选计划（prior），世界模型评估得到证据 E 与 prior_logits，再由 PPO（posterior）选择并执行动作
# - 支持：多episode评估、UTF-8日志输出、JSONL逐步记录、末尾统计汇总
import sys
import argparse
import json
from pathlib import Path
from collections import Counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# 让 `import src.*` 在任意工作目录下都可用（优先插入）
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import load_yaml, set_seed  # noqa: E402
from src.env_region import build_region_objects, build_evidence  # noqa: E402


def load_world_model(objs, path: Path):
    import torch
    sd_list = torch.load(path, map_location=objs.world_model.device)
    for m, sd in zip(objs.world_model.models, sd_list):
        m.load_state_dict(sd)
        m.eval()


def _safe_open_text(path: str, mode: str):
    if not path:
        return None
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # 强制 UTF-8，避免 powershell / Tee-Object 编码差异带来的解析问题
    return open(str(p), mode, encoding="utf-8", newline="\n")


def _write_line(f, s: str):
    if f is None:
        return
    f.write(s.rstrip("\n") + "\n")
    f.flush()


def _write_jsonl(f, obj: dict):
    if f is None:
        return
    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    f.flush()


def _summarize_episode(ep_rewards, ep_actions):
    out = {}
    if ep_rewards:
        out["reward_mean"] = sum(ep_rewards) / len(ep_rewards)
        out["reward_min"] = min(ep_rewards)
        out["reward_max"] = max(ep_rewards)
    else:
        out["reward_mean"] = None
        out["reward_min"] = None
        out["reward_max"] = None
    out["steps"] = len(ep_rewards)
    out["unique_actions"] = len(set(ep_actions))
    out["top5_actions"] = Counter(ep_actions).most_common(5)
    return out




def _z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd < 1e-6:
        return x - mu
    return (x - mu) / sd


def _heuristic_plan_score(prior_logits: np.ndarray, E: np.ndarray) -> np.ndarray:
    # E = [G, U, C, Cost]
    G = _z(E[:, 0])
    U = _z(E[:, 1])
    C = _z(E[:, 2])
    Cost = _z(E[:, 3])
    P = _z(prior_logits)
    return 0.55 * P + 1.20 * G - 0.45 * U + 0.50 * C - 0.22 * Cost


def _hybrid_choose_plan(objs, prior_logits: np.ndarray, E: np.ndarray, deterministic: bool = True):
    import torch
    pl = torch.tensor(prior_logits[None, :], dtype=torch.float32, device=objs.ppo.device)
    Et = torch.tensor(E[None, :, :], dtype=torch.float32, device=objs.ppo.device)
    with torch.no_grad():
        ppo_logits, value = objs.ppo.net(pl, Et)
        ppo_logits = ppo_logits.squeeze(0).cpu().numpy()
    hs = _heuristic_plan_score(prior_logits, E)
    mix = 0.65 * ppo_logits + 0.90 * hs
    if deterministic:
        idx = int(np.argmax(mix))
    else:
        probs = np.exp(mix - np.max(mix))
        probs = probs / np.sum(probs)
        idx = int(np.random.choice(len(probs), p=probs))
    return idx, float(value.item() if hasattr(value, 'item') else value), ppo_logits, hs


def _obs_heuristic_action(obs: dict, summary) -> int:
    raw = getattr(summary, "raw_stats", {}) or {}
    risk = float(obs.get("risk_proxy", raw.get("risk_proxy", 0.0)))
    events = list(obs.get("events", []))
    if not events:
        if risk >= 0.72:
            return 4
        if risk >= 0.45:
            return 3
        if risk >= 0.22:
            return 1
        return 0
    # 结合出现频次与严重度评估主导事件
    score_by_type = {}
    for e in events:
        et = str(e.get("type", "port_scan"))
        sev = float(e.get("severity", 0.0))
        score_by_type[et] = score_by_type.get(et, 0.0) + (0.6 + sev)
    dominant_type = max(score_by_type.items(), key=lambda kv: kv[1])[0]
    max_sev = max(float(e.get("severity", 0.0)) for e in events)

    if risk >= 0.78 or max_sev >= 0.92:
        return 4
    if dominant_type in ("outbound_conn", "proc_spawn"):
        if risk >= 0.40 or max_sev >= 0.72:
            return 3
        return 2
    if dominant_type == "port_scan":
        if risk >= 0.55:
            return 3
        if risk >= 0.26 or max_sev >= 0.55:
            return 2
        return 1
    if dominant_type == "auth_fail":
        if risk >= 0.55:
            return 2
        return 1
    if risk >= 0.55:
        return 3
    if risk >= 0.28:
        return 2
    return 1 if max_sev >= 0.45 else 0


def _aggregate_action_scores(plans, mix_scores: np.ndarray, n_actions: int = 5) -> np.ndarray:
    scores = np.zeros((n_actions,), dtype=np.float32)
    if len(plans) == 0:
        return scores
    m = np.asarray(mix_scores, dtype=np.float32)
    m = m - np.max(m)
    w = np.exp(m)
    w = w / (np.sum(w) + 1e-12)
    for j, plan in enumerate(plans):
        a = int(plan[0]) if plan else 0
        if 0 <= a < n_actions:
            scores[a] += 1.0 * w[j]
            # 对计划首两步一致的动作再给一点信用，鼓励稳定行动而非抖动
            if len(plan) >= 2 and int(plan[1]) == a:
                scores[a] += 0.20 * w[j]
    return scores

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/cc4.yaml")
    ap.add_argument("--region_id", type=int, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=200, help="每个episode最多步数")
    ap.add_argument("--episodes", type=int, default=1, help="运行多少个episode并汇总统计")
    ap.add_argument("--seed", type=int, default=None, help="覆盖cfg里的seed；每个episode会自动+ep偏移")
    ap.add_argument("--deterministic", action="store_true", help="PPO使用确定性动作（默认True更便于复现实验）")
    ap.add_argument("--stochastic", action="store_true", help="PPO使用随机动作（与--deterministic互斥）")
    ap.add_argument("--log_path", default="", help="将控制台日志同步写入该文件（UTF-8）")
    ap.add_argument("--jsonl_path", default="", help="将每步结构化结果写入JSONL文件（UTF-8）")
    ap.add_argument("--append", action="store_true", help="追加写入log/jsonl而不是覆盖")
    ap.add_argument("--cc4_eval", action="store_true", help="按CC4风格评测：100个episode、每个500步")
    ap.add_argument("--cc4_n_episodes", type=int, default=100, help="CC4风格评测的episode数（默认100）")
    ap.add_argument("--cc4_episode_steps", type=int, default=500, help="CC4风格评测每个episode步数（默认500）")
    args = ap.parse_args()

    if args.cc4_eval:
        args.episodes = int(args.cc4_n_episodes)
        args.steps = int(args.cc4_episode_steps)

    cfg_path = (ROOT / args.cfg).resolve()
    cfg = load_yaml(str(cfg_path))

    base_seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    set_seed(base_seed)

    objs = build_region_objects(cfg, region_id=args.region_id, device=args.device)
    # 对本地online客户端，显式覆盖每个episode步数上限，避免被配置中的100步截断
    if hasattr(objs.client, "max_steps"):
        try:
            objs.client.max_steps = int(args.steps)
        except Exception:
            pass

    ckpt_dir = (ROOT / cfg["paths"]["ckpt_dir"]).resolve()
    wm_path = ckpt_dir / f"world_model_region{args.region_id}.pt"
    ppo_path = ckpt_dir / f"ppo_posterior_region{args.region_id}.pt"

    if wm_path.exists():
        load_world_model(objs, wm_path)
        print("[OK] loaded world model:", wm_path)
    else:
        print("[WARN] world model not found:", wm_path)

    if ppo_path.exists():
        # 兼容PPO封装：objs.ppo.load(str(path))
        objs.ppo.load(str(ppo_path))
        print("[OK] loaded PPO:", ppo_path)
    else:
        print("[WARN] PPO not found:", ppo_path)

    deterministic = True
    if args.stochastic:
        deterministic = False
    if args.deterministic:
        deterministic = True

    log_mode = "a" if args.append else "w"
    flog = _safe_open_text(args.log_path, log_mode) if args.log_path else None
    fjsonl = _safe_open_text(args.jsonl_path, log_mode) if args.jsonl_path else None

    header = {
        "cfg": str(cfg_path),
        "region_id": args.region_id,
        "device": args.device,
        "steps_per_episode": args.steps,
        "episodes": args.episodes,
        "seed": base_seed,
        "deterministic": deterministic,
        "wm_path": str(wm_path),
        "ppo_path": str(ppo_path),
    }
    _write_line(flog, "# " + json.dumps(header, ensure_ascii=False))

    all_rewards = []
    all_actions = []
    all_choose_plan = []
    episode_returns = []
    episode_steps = []

    for ep in range(args.episodes):
        # 每个episode偏移seed，保证可重复又不完全相同
        ep_seed = int(base_seed + args.region_id * 1000 + ep)
        set_seed(ep_seed)

        try:
            obs = objs.client.reset(seed=ep_seed)
        except TypeError:
            obs = objs.client.reset()
        objs.summarizer.reset()

        ep_rewards = []
        ep_actions = []

        _write_line(flog, f"\n# ===== EPISODE {ep} =====")
        print(f"[EP={ep}] start")

        for t in range(args.steps):
            # 1) 更新摘要
            objs.summarizer.update(obs.get("events", []))
            summary = objs.summarizer.build(risk_proxy=float(obs.get("risk_proxy", 0.0)))

            # 2) LLM prior 生成候选计划
            prior_out = objs.llm_prior.generate(summary)

            # 3) 世界模型评估，构造证据与PPO输入
            prior_logits, E, plans, _ = build_evidence(cfg, objs.world_model, summary, prior_out)

            # 4) PPO posterior 选择计划索引（choose_plan），并融合观测启发式得到最终动作
            choose_plan, _, ppo_logits_dbg, hs_dbg = _hybrid_choose_plan(objs, prior_logits=prior_logits, E=E, deterministic=deterministic)
            mix_scores = 0.65 * np.asarray(ppo_logits_dbg, dtype=np.float32) + 0.90 * np.asarray(hs_dbg, dtype=np.float32)
            action_vote = _aggregate_action_scores(plans, mix_scores, n_actions=5)
            heuristic_action = _obs_heuristic_action(obs, summary)
            # 强化基于当前观测的即时决策；高风险时允许更果断升级
            risk_now = float(obs.get("risk_proxy", summary.raw_stats.get("risk_proxy", 0.0)))
            action_vote[heuristic_action] += 1.35 + 0.65 * risk_now
            # 高风险/高严重度时，优先服从启发式升级；否则取融合投票结果
            max_sev_now = 0.0
            for _e in obs.get("events", []):
                max_sev_now = max(max_sev_now, float(_e.get("severity", 0.0)))
            voted_action = int(np.argmax(action_vote))
            if risk_now >= 0.62 or max_sev_now >= 0.82:
                action_id = max(voted_action, heuristic_action)
            else:
                action_id = voted_action

            # 5) 执行动作
            obs, r, done, info = objs.client.step(action_id)

            ep_rewards.append(float(r))
            ep_actions.append(action_id)
            all_rewards.append(float(r))
            all_actions.append(action_id)
            all_choose_plan.append(choose_plan)

            line = f"[ep={ep:02d} t={t:04d}] choose_plan={choose_plan} action={action_id} reward={r:.3f} info={info}"
            print(line)
            _write_line(flog, line)

            # 结构化写入（便于你后面做统计/画图）
            if fjsonl is not None:
                rec = {
                    "episode": ep,
                    "t": t,
                    "choose_plan": choose_plan,
                    "action_id": action_id,
                    "heuristic_action": int(heuristic_action),
                    "reward": float(r),
                    "done": bool(done),
                    "region_id": int(args.region_id),
                    "ppo_logits": [float(x) for x in np.asarray(ppo_logits_dbg).tolist()],
                    "heuristic_scores": [float(x) for x in np.asarray(hs_dbg).tolist()],
                }
                # info里一般是可json化的dict，但保险起见做一下兜底
                try:
                    rec["info"] = info
                except Exception:
                    rec["info"] = str(info)
                _write_jsonl(fjsonl, rec)

            if done:
                _write_line(flog, f"# done at t={t}")
                break

        ep_sum = _summarize_episode(ep_rewards, ep_actions)
        ep_return = float(sum(ep_rewards)) if ep_rewards else 0.0
        episode_returns.append(ep_return)
        episode_steps.append(len(ep_rewards))
        _write_line(flog, "# episode_summary " + json.dumps({"episode": ep, "episode_return": ep_return, **ep_sum}, ensure_ascii=False))
        print(f"[EP={ep}] summary:", {**ep_sum, "episode_return": ep_return})

    # 全局汇总
    overall = _summarize_episode(all_rewards, all_actions)
    overall["unique_choose_plan"] = len(set(all_choose_plan))
    overall["top5_choose_plan"] = Counter(all_choose_plan).most_common(5)
    _write_line(flog, "\n# ===== OVERALL =====")
    _write_line(flog, "# overall_summary " + json.dumps(overall, ensure_ascii=False))
    print("[OVERALL] summary:", overall)

    if episode_returns:
        cc4_summary = {
            "episodes": len(episode_returns),
            "steps_per_episode": int(args.steps),
            "mean_reward": float(np.mean(np.asarray(episode_returns, dtype=np.float64))),
            "std_reward": float(np.std(np.asarray(episode_returns, dtype=np.float64), ddof=0)),
        }
        _write_line(flog, "# cc4_style_summary " + json.dumps(cc4_summary, ensure_ascii=False))
        print("[CC4-STYLE]", cc4_summary)

    if flog is not None:
        flog.close()
    if fjsonl is not None:
        fjsonl.close()


if __name__ == "__main__":
    main()

