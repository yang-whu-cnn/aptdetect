# src/state_summary.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

# 你后续对接真实CC4时：把你能从观测里拿到的字段映射成这些“统一特征键”
FEATURE_KEYS = [
    "cnt_auth_fail",
    "cnt_port_scan",
    "cnt_proc_spawn",
    "cnt_outbound_conn",
    "avg_severity",
    "unique_src",
    "unique_dst",
    # risk_proxy（风险代理分数）建议由轻量检测器/规则估计器给出；mock环境会直接提供
    "risk_proxy",
]


@dataclass
class StateSummary:
    vec: np.ndarray               # 结构化向量（用于世界模型/后验策略）
    text: str                     # 文本化摘要（用于LLM输入）
    raw_stats: Dict[str, float]   # 便于一致性检查点计算等


class SlidingWindowSummarizer:
    """滑动窗口聚合近期事件，输出“结构化+文本化”摘要。"""
    def __init__(self, window_size: int = 20, max_text_len: int = 512):
        self.window_size = int(window_size)
        self.max_text_len = int(max_text_len)
        self.buffer: List[Dict] = []

    def reset(self):
        self.buffer.clear()

    def update(self, events: List[Dict]):
        # events: List[{type, severity, src, dst, ...}]
        self.buffer.extend(events)
        if len(self.buffer) > self.window_size:
            self.buffer = self.buffer[-self.window_size:]

    def build(self, risk_proxy: float = 0.0) -> StateSummary:
        stats = {k: 0.0 for k in FEATURE_KEYS}
        if len(self.buffer) == 0:
            stats["risk_proxy"] = float(risk_proxy)
            vec = np.array([stats[k] for k in FEATURE_KEYS], dtype=np.float32)
            return StateSummary(vec=vec, text="", raw_stats=stats)

        sev = []
        srcs, dsts = set(), set()

        for e in self.buffer:
            t = str(e.get("type", ""))
            s = float(e.get("severity", 0.0))
            sev.append(s)
            srcs.add(str(e.get("src", "")))
            dsts.add(str(e.get("dst", "")))

            if t == "auth_fail":
                stats["cnt_auth_fail"] += 1.0
            elif t == "port_scan":
                stats["cnt_port_scan"] += 1.0
            elif t == "proc_spawn":
                stats["cnt_proc_spawn"] += 1.0
            elif t == "outbound_conn":
                stats["cnt_outbound_conn"] += 1.0

        stats["avg_severity"] = float(np.mean(sev)) if len(sev) else 0.0
        stats["unique_src"] = float(len(srcs))
        stats["unique_dst"] = float(len(dsts))
        stats["risk_proxy"] = float(risk_proxy)

        vec = np.array([stats[k] for k in FEATURE_KEYS], dtype=np.float32)

        # 文本化摘要：压缩为“事件类型计数 + 最高severity的几条”
        cnt_part = (
            f"auth_fail={int(stats['cnt_auth_fail'])}, "
            f"port_scan={int(stats['cnt_port_scan'])}, "
            f"proc_spawn={int(stats['cnt_proc_spawn'])}, "
            f"outbound_conn={int(stats['cnt_outbound_conn'])}, "
            f"avg_sev={stats['avg_severity']:.2f}, "
            f"risk={stats['risk_proxy']:.2f}"
        )

        # 取最近3条/最高severity的3条做可读描述
        last3 = self.buffer[-3:]
        top3 = sorted(self.buffer, key=lambda x: float(x.get("severity", 0.0)), reverse=True)[:3]

        def fmt(e: Dict) -> str:
            return f"{e.get('type','?')} sev={float(e.get('severity',0.0)):.2f} src={e.get('src','?')} dst={e.get('dst','?')}"

        text_lines = [
            "[counts] " + cnt_part,
            "[recent] " + " | ".join(fmt(e) for e in last3),
            "[top] " + " | ".join(fmt(e) for e in top3),
        ]
        text = "\n".join(text_lines)
        text = text[: self.max_text_len]

        return StateSummary(vec=vec, text=text, raw_stats=stats)


def checkpoint_satisfied(checkpoint, stats: dict) -> bool:
    """
    Make checkpoint check robust.
    - checkpoint can be: str rule, dict rule, list/tuple of rules, None.
    - stats is a dict of current/pseudo stats.
    """

    # 0) None / empty => satisfied
    if checkpoint is None:
        return True

    # 1) list/tuple => all satisfied
    if isinstance(checkpoint, (list, tuple)):
        return all(checkpoint_satisfied(cp, stats) for cp in checkpoint)

    # 2) dict rule => evaluate per-key
    # Supported forms:
    #   {"risk_proxy": 0.3}                  -> default op "<="
    #   {"risk_proxy": {"op": "<=", "value": 0.3}}
    #   {"risk_proxy": {"op": "<", "thr": 0.3}}
    if isinstance(checkpoint, dict):
        for k, cond in checkpoint.items():
            if stats is None or k not in stats:
                return False
            v = stats.get(k)

            # parse condition
            if isinstance(cond, dict):
                op = str(cond.get("op", "<=")).strip()
                if "value" in cond:
                    thr = float(cond["value"])
                elif "thr" in cond:
                    thr = float(cond["thr"])
                elif "threshold" in cond:
                    thr = float(cond["threshold"])
                else:
                    # fallback: try first numeric value in dict
                    num = None
                    for _kk, _vv in cond.items():
                        try:
                            num = float(_vv)
                            break
                        except Exception:
                            continue
                    if num is None:
                        return False
                    thr = float(num)
            else:
                op = "<="
                try:
                    thr = float(cond)
                except Exception:
                    return False

            # compare
            try:
                fv = float(v)
            except Exception:
                return False

            if op in ("<", "lt"):
                ok = fv < thr
            elif op in ("<=", "le"):
                ok = fv <= thr
            elif op in (">", "gt"):
                ok = fv > thr
            elif op in (">=", "ge"):
                ok = fv >= thr
            elif op in ("==", "eq"):
                ok = fv == thr
            elif op in ("!=", "ne"):
                ok = fv != thr
            else:
                # unknown op => treat as not satisfied
                return False

            if not ok:
                return False
        return True

    # 3) non-str => cast to str (avoid .strip crash)
    if not isinstance(checkpoint, str):
        checkpoint = str(checkpoint)

    # 4) original string-rule logic (keep compatibility)
    cp = checkpoint.strip().lower()

    # empty string => satisfied
    if cp == "":
        return True

    # --- Below is a conservative parser; supports common forms:
    # "risk_proxy<=0.3", "risk_proxy < 0.3", "risk_proxy>=0.1", etc.
    # If your original file already had more complex parsing, you can keep it,
    # just make sure it starts after the safe casting above.
    import re
    m = re.match(r"^\s*([a-zA-Z0-9_]+)\s*(<=|>=|==|!=|<|>)\s*([-+]?\d+(\.\d+)?)\s*$", cp)
    if not m:
        # unknown rule format => do not block training; treat as satisfied
        return True

    key, op, num, _ = m.groups()
    if stats is None or key not in stats:
        return False

    try:
        fv = float(stats.get(key))
        thr = float(num)
    except Exception:
        return False

    if op == "<":
        return fv < thr
    if op == "<=":
        return fv <= thr
    if op == ">":
        return fv > thr
    if op == ">=":
        return fv >= thr
    if op == "==":
        return fv == thr
    if op == "!=":
        return fv != thr

    return True

