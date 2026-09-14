# -*- coding: utf-8 -*-
"""
把 CC4 回放/部署轨迹 jsonl 转换为 chapter1 可用的扁平事件流 jsonl
兼容两类输入：
1) chapter2 导出的回放格式（top-level: t/obs/reward/done/info）
2) run_deploy_region.py 导出的部署格式（top-level: episode/t/choose_plan/action_id/reward/done/info）

说明（这版修复了你现在遇到的两个关键问题）：
- 修复1：部署 jsonl 没有 obs/events/risk 时，pos_rows 不再恒为 0（增加 fallback 打标签）
- 修复2：timestamp 不再直接用 rec["t"]（因为每个 episode 会从 0 重置），改为“文件内全局步号”，避免 uniq_ts 很少导致 T 上不去
"""

import argparse
import json
import os
import re
from typing import Any, Dict, List, Tuple


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _to_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


def infer_region_from_path(path: str) -> int:
    m = re.search(r"region(\d+)", os.path.basename(path).lower())
    return int(m.group(1)) if m else -1


def make_event_desc(ev: Dict[str, Any]) -> str:
    parts = [
        f"type={ev.get('type', '')}",
        f"src={ev.get('src', '')}",
        f"dst={ev.get('dst', '')}",
        f"zone={ev.get('zone', '')}",
        f"action={ev.get('action', '')}",
        f"target={ev.get('target_host', '')}",
        f"sev={ev.get('severity', '')}",
        f"success={ev.get('success', '')}",
    ]
    return " ".join(parts)


def _extract_obs_info(rec: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    obs = rec.get("obs", {}) or {}
    info = rec.get("info", {}) or {}
    if not isinstance(obs, dict):
        obs = {}
    if not isinstance(info, dict):
        info = {}
    # 兼容某些写法：obs/info 可能嵌在别的字段里（保底）
    if not obs and isinstance(rec.get("state"), dict):
        obs = rec["state"]
    return obs, info


def _extract_events(obs: Dict[str, Any], info: Dict[str, Any], rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    events = obs.get("events", None)
    if events is None:
        events = rec.get("events", None)
    if events is None:
        events = info.get("events", None)
    if not isinstance(events, list):
        return []
    out = []
    for ev in events:
        if isinstance(ev, dict):
            out.append(ev)
    return out


def _infer_label(
    rec: Dict[str, Any],
    obs: Dict[str, Any],
    info: Dict[str, Any],
    events: List[Dict[str, Any]],
    risk_thr: float,
    fallback_label_mode: str,
    reward_pos_thr: float,
) -> Tuple[int, str]:
    """
    返回: (is_attack, label_source)
    label_source 便于打印统计/排查
    """
    # 优先用显式信号（最靠谱）
    risk_proxy = _safe_float(
        obs.get("risk_proxy", rec.get("risk_proxy", info.get("risk_proxy", 0.0))), 0.0
    )
    risk = _safe_float(rec.get("risk", info.get("risk", risk_proxy)), risk_proxy)

    if len(events) > 0 or risk_proxy >= risk_thr or risk >= risk_thr:
        return 1, "explicit_event_or_risk"

    # 没有显式信号时，针对 deploy 日志做 fallback
    action_id = _safe_int(rec.get("action_id", rec.get("action", 0)), 0)
    reward = _safe_float(rec.get("reward", 0.0), 0.0)

    if fallback_label_mode == "none":
        return 0, "fallback_none"
    if fallback_label_mode == "action":
        return (1 if action_id != 0 else 0), "fallback_action"
    if fallback_label_mode == "reward":
        # CC4 这类日志 reward 越负通常越“异常/代价高”，默认用更负阈值判正
        return (1 if reward <= reward_pos_thr else 0), "fallback_reward"
    if fallback_label_mode == "action_or_reward":
        return (1 if (action_id != 0 or reward <= reward_pos_thr) else 0), "fallback_action_or_reward"

    # auto（默认）：部署日志常见情况优先 action；若 action 全0，再靠 reward 兜底
    if action_id != 0:
        return 1, "fallback_auto_action"
    if reward <= reward_pos_thr:
        return 1, "fallback_auto_reward"
    return 0, "fallback_auto_neg"


def convert_one_file(
    in_path: str,
    out_f,
    base_ts: int,
    step_sec: int = 300,
    risk_thr: float = 0.15,
    emit_empty_step: bool = True,
    fallback_label_mode: str = "auto",
    reward_pos_thr: float = -4.8,
) -> Dict[str, Any]:
    n_steps = 0                 # 文件内总步数（全局单调，用于 timestamp）
    n_rows = 0
    n_pos = 0
    region_fallback = infer_region_from_path(in_path)

    # 调试统计（帮助你看 pos_rows 为什么多/少）
    label_src_cnt: Dict[str, int] = {}
    action_nonzero_steps = 0

    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n_steps += 1

            # 注意：rec["t"] 在 deploy 日志里每个 episode 会从0重置，不能直接拿来做 timestamp
            t_local = _safe_int(rec.get("t", 0), 0)
            step_idx_global = n_steps - 1

            obs, info = _extract_obs_info(rec)
            reward = _safe_float(rec.get("reward", 0.0), 0.0)
            done = 1 if bool(rec.get("done", False)) else 0

            region_id = _safe_int(
                rec.get("region_id", info.get("region_id", obs.get("region_id", region_fallback))),
                region_fallback,
            )

            risk_proxy = _safe_float(
                obs.get("risk_proxy", rec.get("risk_proxy", info.get("risk_proxy", 0.0))), 0.0
            )
            risk = _safe_float(rec.get("risk", info.get("risk", risk_proxy)), risk_proxy)

            events = _extract_events(obs, info, rec)

            is_attack_step, label_src = _infer_label(
                rec=rec,
                obs=obs,
                info=info,
                events=events,
                risk_thr=risk_thr,
                fallback_label_mode=fallback_label_mode,
                reward_pos_thr=reward_pos_thr,
            )
            label_src_cnt[label_src] = label_src_cnt.get(label_src, 0) + 1

            action_id = _safe_int(rec.get("action_id", rec.get("action", 0)), 0)
            choose_plan = _safe_int(rec.get("choose_plan", -1), -1)
            if action_id != 0:
                action_nonzero_steps += 1

            # 用“文件内全局步号”映射时间戳，避免 episode 重置导致 uniq_ts 太少
            ts = base_ts + step_idx_global * step_sec

            # 无事件时：生成 heartbeat/decision 行（部署日志大多走这里）
            if len(events) == 0 and emit_empty_step:
                event_type = "decision" if ("action_id" in rec or "choose_plan" in rec) else "heartbeat"
                # 为了让第一章有可用类别特征，把决策信息放进 action/event_desc
                action_str = f"act_{action_id}" if ("action_id" in rec or "action" in rec) else ""
                event_desc = (
                    f"{event_type} no_event choose_plan={choose_plan} action_id={action_id} "
                    f"reward={reward:.6f} label_src={label_src}"
                )

                row = {
                    "timestamp": ts,
                    "agent_id": f"R{region_id}",
                    "event_type": event_type,
                    "is_attack": is_attack_step,
                    "attack_stage": "unknown",

                    # 数值字段
                    "severity": 0.0,
                    "risk_proxy": risk_proxy,
                    "risk": risk,
                    "reward": reward,
                    "done_flag": done,
                    "t_step": t_local,
                    "t_step_global": step_idx_global,

                    # 类别字段
                    "region_id": _to_str(region_id),
                    "src": "",
                    "dst": "",
                    "zone": "",
                    "action": action_str,
                    "target_host": "",
                    "success": "",

                    # 文本字段
                    "event_desc": event_desc,
                }
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_rows += 1
                n_pos += is_attack_step
                continue

            # 有事件：按事件级展开
            for ev in events:
                sev = _safe_float(ev.get("severity", 0.0), 0.0)

                row = {
                    "timestamp": ts,
                    "agent_id": f"R{region_id}",
                    "event_type": _to_str(ev.get("type", "unknown")),
                    "is_attack": is_attack_step,
                    "attack_stage": "unknown",

                    # 数值字段
                    "severity": sev,
                    "risk_proxy": risk_proxy,
                    "risk": risk,
                    "reward": reward,
                    "done_flag": done,
                    "t_step": t_local,
                    "t_step_global": step_idx_global,

                    # 类别字段
                    "region_id": _to_str(region_id),
                    "src": _to_str(ev.get("src", "")),
                    "dst": _to_str(ev.get("dst", "")),
                    "zone": _to_str(ev.get("zone", "")),
                    "action": _to_str(ev.get("action", f"act_{action_id}" if action_id != 0 else "")),
                    "target_host": _to_str(ev.get("target_host", "")),
                    "success": _to_str(ev.get("success", "")),

                    # 文本字段
                    "event_desc": make_event_desc(ev) + f" label_src={label_src}",
                }
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_rows += 1
                n_pos += is_attack_step

    return {
        "file": in_path,
        "steps": n_steps,
        "rows": n_rows,
        "pos_rows": n_pos,
        "base_ts_used": base_ts,
        "label_src_cnt": label_src_cnt,
        "action_nonzero_steps": action_nonzero_steps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="一个或多个 region jsonl 输入文件")
    ap.add_argument("--output", required=True, help="输出的 chapter1 扁平 jsonl")
    ap.add_argument("--step-sec", type=int, default=300, help="每个CC4步映射为多少秒（默认300秒）")
    ap.add_argument("--risk-thr", type=float, default=0.15, help="显式攻击标签阈值（risk_proxy/risk）")
    ap.add_argument("--no-empty-step", action="store_true", help="无事件步不生成 heartbeat/decision 行")

    # 新增：部署日志无 obs/events/risk 时的兜底打标签策略
    ap.add_argument(
        "--fallback-label-mode",
        type=str,
        default="auto",
        choices=["auto", "none", "action", "reward", "action_or_reward"],
        help="无显式事件/risk时的标签兜底方式（默认auto）",
    )
    ap.add_argument(
        "--reward-pos-thr",
        type=float,
        default=-4.8,
        help="fallback=reward/action_or_reward/auto 时，reward<=该阈值记为正样本（默认-4.8）",
    )

    args = ap.parse_args()

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # 给每个输入文件一个不同基准时间，避免文件之间时间戳重叠
    global_base_ts = 1700000000
    gap_sec_between_files = 86400  # 1天偏移

    summaries: List[Dict[str, Any]] = []
    total_rows = 0
    total_pos = 0

    with open(args.output, "w", encoding="utf-8") as out_f:
        for i, in_path in enumerate(args.inputs):
            base_ts = global_base_ts + i * gap_sec_between_files
            s = convert_one_file(
                in_path=in_path,
                out_f=out_f,
                base_ts=base_ts,
                step_sec=args.step_sec,
                risk_thr=args.risk_thr,
                emit_empty_step=(not args.no_empty_step),
                fallback_label_mode=args.fallback_label_mode,
                reward_pos_thr=args.reward_pos_thr,
            )
            summaries.append(s)
            total_rows += s["rows"]
            total_pos += s["pos_rows"]

    print("[OK] converted ->", args.output)
    for s in summaries:
        print(
            f"  - {os.path.basename(s['file'])}: "
            f"steps={s['steps']} rows={s['rows']} pos_rows={s['pos_rows']} "
            f"action_nonzero_steps={s['action_nonzero_steps']} base_ts={s['base_ts_used']}"
        )
        print(f"    label_src={s['label_src_cnt']}")
    print(f"[TOTAL] rows={total_rows} pos_rows={total_pos} neg_rows={total_rows-total_pos}")


if __name__ == "__main__":
    main()