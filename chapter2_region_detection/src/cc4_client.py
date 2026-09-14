# src/cc4_client.py
# -*- coding: utf-8 -*-
"""
CC4 Client：支持三种模式
- mock  : 纯本地随机/规则生成（用于快速联调）
- replay: 读 jsonl 回放
- online: 远程 HTTP（真实 CC4）；但当 base_url == "local" 时走本地仿真 online

接口统一：
    reset(seed=None) -> obs(dict)
    step(action) -> (obs, reward, done, info)
"""

from __future__ import annotations

import json
import os
import time
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


# ============== Base ==============

class CC4ClientBase:
    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        raise NotImplementedError

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        raise NotImplementedError


# ============== Mock / Local Sim ==============

_EVENT_TYPES = [
    "host_compromise",
    "suspicious_login",
    "process_injection",
    "lateral_movement",
    "c2_beacon",
    "exfiltration",
    "benign_activity",
]


def _mk_event(rng: random.Random, t: int) -> Dict[str, Any]:
    et = rng.choices(
        population=_EVENT_TYPES,
        weights=[0.08, 0.10, 0.07, 0.06, 0.06, 0.03, 0.60],
        k=1
    )[0]
    sev = {
        "host_compromise": 3,
        "suspicious_login": 2,
        "process_injection": 2,
        "lateral_movement": 3,
        "c2_beacon": 2,
        "exfiltration": 4,
        "benign_activity": 0,
    }[et]
    return {
        "type": et,
        "time": t,
        "severity": sev,
        "host": f"host-{rng.randint(1, 50)}",
    }


class MockCC4Client(CC4ClientBase):
    """
    本地随机“事件流”仿真：
    - obs: {"t": int, "events": [...], "meta": {...}}
    - reward: 用简单启发式（可替换为你自己的 reward_shaping）
    """

    def __init__(self, max_steps: int = 100, seed: int = 0):
        self.max_steps = int(max_steps)
        self._seed = int(seed)
        self._rng = random.Random(self._seed)
        self.t = 0
        self.done = False

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            self._seed = int(seed)
        self._rng = random.Random(self._seed)
        self.t = 0
        self.done = False
        return self._make_obs()

    def _make_obs(self) -> Dict[str, Any]:
        # 每步生成 1~5 个事件
        n = self._rng.randint(1, 5)
        events = [_mk_event(self._rng, self.t) for _ in range(n)]
        obs = {
            "t": self.t,
            "events": events,
            "meta": {
                "sim": "mock_local_online",
                "seed": self._seed,
            }
        }
        return obs

    def _reward(self, obs: Dict[str, Any], action: int) -> float:
        """
        简单启发式：
        - 有高危事件出现而你采取更强动作 -> 更“合理”
        - 误报/过度动作 -> 惩罚
        """
        # 最高严重度
        max_sev = 0
        for e in obs.get("events", []):
            max_sev = max(max_sev, int(e.get("severity", 0)))

        # action 强度假设：0..4（你如果动作空间不同，按需改映射）
        intensity = int(action)

        # 目标：高危时强一点、低危时弱一点
        # 简单打分：-(偏差^2) - 轻微步长惩罚
        target = min(max_sev, 4)  # 0..4
        dev = (intensity - target)
        r = - (dev * dev) - 0.05 * self.t
        return float(r)

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self.done:
            # 允许外部继续 step，但直接返回终止
            obs = self._make_obs()
            return obs, 0.0, True, {"msg": "episode already done"}

        obs = self._make_obs()
        r = self._reward(obs, action)

        self.t += 1
        self.done = (self.t >= self.max_steps)

        info = {"t": self.t, "max_steps": self.max_steps}
        return obs, r, self.done, info


# ============== Replay ==============

class ReplayCC4Client(CC4ClientBase):
    """
    读取 jsonl 回放。
    兼容两种常见行格式：
      A) {"obs": {...}, "reward": x, "done": bool, "info": {...}}
      B) {"observation": {...}, "r": x, "d": bool, "info": {...}}

    兼容项目其它位置的调用（不改变原回放逻辑）：
      ReplayCC4Client(path)
      ReplayCC4Client(path, cfg=cfg, recompute_reward=True)
    """

    def __init__(
        self,
        jsonl_path: str,
        cfg: Optional[dict] = None,
        recompute_reward: bool = False,
        **kwargs,
    ):
        # 仅做“参数兼容”，不改变原行为
        self.jsonl_path = jsonl_path
        self.cfg = cfg
        self.recompute_reward = bool(recompute_reward)
        self._rows: List[Dict[str, Any]] = []
        self._idx = 0
        self._warned_missing_reward = False
        self._load()

    def _load(self):
        if not os.path.exists(self.jsonl_path):
            raise FileNotFoundError(f"Replay jsonl not found: {self.jsonl_path}")

        rows = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        if not rows:
            raise ValueError(f"Empty replay file: {self.jsonl_path}")

        self._rows = rows
        self._idx = 0

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        self._idx = 0
        first = self._rows[0]
        return first.get("obs") or first.get("observation") or {}

    def _get_reward_from_row(self, row: Dict[str, Any]) -> float:
        # 原逻辑：优先使用文件里存的 reward/r
        if "reward" in row or "r" in row:
            return float(row.get("reward", row.get("r", 0.0)))

        # 若文件没有 reward 字段：保持兼容（给0），并仅提示一次
        if self.recompute_reward and (not self._warned_missing_reward):
            print("[WARN] replay row has no reward/r field; fallback reward=0.0 (recompute_reward ignored).")
            self._warned_missing_reward = True
        return 0.0

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        self._idx += 1
        if self._idx >= len(self._rows):
            last = self._rows[-1]
            obs = last.get("obs") or last.get("observation") or {}
            return obs, self._get_reward_from_row(last), True, last.get("info", {})

        row = self._rows[self._idx]
        obs = row.get("obs") or row.get("observation") or {}
        r = self._get_reward_from_row(row)
        d = bool(row.get("done", row.get("d", False)))
        info = row.get("info", {})
        return obs, r, d, info


# ============== Online (Remote or Local Fallback) ==============


class OnlineCC4Client:
    """
    本地 online 仿真客户端（base_url=local）。

    目标：
    1) 不同 region 在相同 base seed 下仍有可复现但不同的轨迹；
    2) 动作对风险演化和回报有实质影响，避免策略长期退化为 action=0；
    3) 保持 reward 为环境 step 返回的单步回报，供部署/评估脚本直接汇总。
    """

    _REGION_PROFILES = {
        0: {"name": "hq",      "init_risk": 0.28, "drift": 0.018, "scan_bias": 1.20, "proc_bias": 0.95, "out_bias": 0.90, "auth_bias": 1.00, "eff": [0.00, 0.045, 0.085, 0.135, 0.180]},
        1: {"name": "edge_a",  "init_risk": 0.35, "drift": 0.022, "scan_bias": 0.95, "proc_bias": 1.20, "out_bias": 1.00, "auth_bias": 0.95, "eff": [0.00, 0.044, 0.088, 0.140, 0.185]},
        2: {"name": "edge_b",  "init_risk": 0.42, "drift": 0.025, "scan_bias": 0.90, "proc_bias": 0.95, "out_bias": 1.25, "auth_bias": 1.00, "eff": [0.00, 0.040, 0.080, 0.128, 0.172]},
        3: {"name": "branch",  "init_risk": 0.32, "drift": 0.020, "scan_bias": 1.00, "proc_bias": 1.00, "out_bias": 0.95, "auth_bias": 1.20, "eff": [0.00, 0.046, 0.087, 0.138, 0.182]},
    }

    def __init__(self, base_url: str = "local", api_key: str = None, max_steps_local: int = 100, seed_local: int = 0, region_id: int = 0, cfg: dict = None, **kwargs):
        if cfg is not None:
            online_cfg = cfg.get("online", {}) if isinstance(cfg, dict) else {}
            cc4_cfg = cfg.get("cc4", {}) if isinstance(cfg, dict) else {}
            train_cfg = cfg.get("train", {}) if isinstance(cfg, dict) else {}
            base_url = online_cfg.get("base_url", cc4_cfg.get("base_url", base_url))
            api_key = online_cfg.get("api_key", cc4_cfg.get("api_key", api_key))
            if "max_steps_local" in online_cfg:
                max_steps_local = int(online_cfg["max_steps_local"])
            elif "max_steps_local" in cc4_cfg:
                max_steps_local = int(cc4_cfg["max_steps_local"])
            elif "episode_max_steps" in online_cfg:
                max_steps_local = int(online_cfg["episode_max_steps"])
            elif "episode_max_steps" in cfg:
                max_steps_local = int(cfg["episode_max_steps"])
            elif "max_episode_steps" in train_cfg:
                max_steps_local = int(train_cfg["max_episode_steps"])
            else:
                max_steps_local = int(max_steps_local)
            base_seed = int(cfg.get("seed", seed_local))
            # 注意：这里保留 region 扰动；reset 时还会继续加 region 偏移，保证多区域分化。
            seed_local = base_seed + int(region_id) * 997

        self.base_url = base_url
        self.api_key = api_key
        self.max_steps = int(max_steps_local)
        self.region_id = int(region_id)
        self.profile = dict(self._REGION_PROFILES.get(self.region_id, self._REGION_PROFILES[0]))

        if self.base_url != "local":
            raise RuntimeError(
                f"你当前配置 base_url={self.base_url} 不是 local，但你又拿不到真实 CC4 URL/API Key。"
                f"请把 configs 里 cc4.base_url 设为 local。"
            )

        self._base_seed = int(seed_local)
        self.rng = np.random.RandomState(self._base_seed)
        self.t = 0
        self.risk = float(self.profile["init_risk"])
        self.last_events = []

    def reset(self, seed: Optional[int] = None):
        # 关键修复：不要让所有 region 在相同 episode seed 下完全重合。
        actual_seed = self._base_seed if seed is None else int(seed) + self.region_id * 1009
        self.rng = np.random.RandomState(actual_seed)
        self.t = 0
        self.risk = float(self.profile["init_risk"] + 0.02 * self.rng.rand())
        events = self._gen_events(init=True)
        self.last_events = list(events)
        obs = {
            "t": self.t,
            "events": events,
            "risk_proxy": float(self.risk),
            "region_id": self.region_id,
            "region_name": self.profile["name"],
        }
        return obs

    def step(self, action_id: int):
        action_id = int(action_id)
        prev_risk = float(self.risk)
        pressure = self._threat_pressure()
        # 先让攻击面自然演化
        self.risk = float(np.clip(self.risk + pressure, 0.0, 1.0))

        # 再施加动作效果；不同 region 的动作有效性略有差异
        eff = float(self.profile["eff"][min(max(action_id, 0), 4)])
        # 当前风险越高，处置动作越有价值；低风险下强处置会带来额外副作用
        adaptive = eff * (0.95 + 1.20 * self.risk)
        self.risk = float(np.clip(self.risk - adaptive, 0.0, 1.0))

        self.t += 1
        events = self._gen_events(init=False)
        self.last_events = list(events)

        dominant_type, dominant_sev = self._dominant_event(events)
        target_action = self._suggest_action(dominant_type, self.risk)
        mismatch = abs(action_id - target_action)
        action_cost = [0.03, 0.10, 0.22, 0.38, 0.55][min(max(action_id, 0), 4)]
        overreact = 0.0
        if prev_risk < 0.25 and action_id >= 3:
            overreact = 0.22 + 0.08 * action_id
        # 回报逻辑：更强地奖励降险和动作匹配，减轻基础惩罚，便于策略形成积极处置
        risk_improve = max(0.0, prev_risk - self.risk)
        match_bonus = 0.35 if mismatch == 0 else (0.12 if mismatch == 1 else 0.0)
        safe_bonus = 0.18 if (self.risk < 0.20 and action_id <= 1) else 0.0
        reward = (
            5.6 * risk_improve
            - 2.9 * self.risk
            - 0.28 * mismatch
            - 0.25 * action_cost
            - overreact
            + 0.28 * float(dominant_sev)
            + match_bonus
            + safe_bonus
        )

        done = self.t >= self.max_steps
        obs = {
            "t": self.t,
            "events": events,
            "risk_proxy": float(self.risk),
            "region_id": self.region_id,
            "region_name": self.profile["name"],
        }
        info = {
            "t": self.t,
            "max_steps": self.max_steps,
            "dominant_type": dominant_type,
            "target_action": int(target_action),
            "prev_risk": float(prev_risk),
            "risk": float(self.risk),
        }
        return obs, float(reward), bool(done), info

    def _threat_pressure(self) -> float:
        noise = float(self.rng.normal(0.0, 0.015))
        burst = 0.0
        if self.rng.rand() < (0.10 + 0.15 * self.risk):
            burst = float(0.03 + 0.04 * self.rng.rand())
        return float(self.profile["drift"] + noise + burst)

    def _rand_ip(self):
        return f"10.{self.rng.randint(0,256)}.{self.rng.randint(0,256)}.{self.rng.randint(1,255)}"

    def _event_weights(self):
        # 根据 region profile 调整各类事件的出现倾向
        return np.array([
            0.22 * self.profile["auth_bias"],
            0.28 * self.profile["scan_bias"],
            0.24 * self.profile["proc_bias"],
            0.26 * self.profile["out_bias"],
        ], dtype=np.float64)

    def _gen_events(self, init: bool):
        if init:
            n = 3 + int(self.region_id % 2)
        else:
            base = 1 + int(self.risk * 5)
            n = int(np.clip(base + self.rng.randint(-1, 2), 1, 7))

        types = ["auth_fail", "port_scan", "proc_spawn", "outbound_conn"]
        weights = self._event_weights()
        weights = weights / np.sum(weights)
        sev_base = float(np.clip(self.risk + 0.08 * self.rng.rand(), 0.0, 1.0))
        events = []
        for _ in range(n):
            idx = int(self.rng.choice(len(types), p=weights))
            et = types[idx]
            sev = float(np.clip(sev_base + 0.10 * self.rng.randn(), 0.0, 1.0))
            events.append({"type": et, "severity": sev, "src": self._rand_ip(), "dst": self._rand_ip()})
        return events

    def _dominant_event(self, events):
        if not events:
            return "port_scan", float(self.risk)
        e = max(events, key=lambda x: float(x.get("severity", 0.0)))
        return str(e.get("type", "port_scan")), float(e.get("severity", self.risk))

    def _suggest_action(self, dominant_type: str, risk: float) -> int:
        # 用于本地 online 的“隐式最优动作”映射，便于测试策略是否学会随态势变化决策
        if risk >= 0.72:
            return 4
        if dominant_type == "outbound_conn":
            return 3 if risk >= 0.42 else 2
        if dominant_type == "proc_spawn":
            return 3 if risk >= 0.35 else 2
        if dominant_type == "port_scan":
            return 2 if risk >= 0.30 else 1
        if dominant_type == "auth_fail":
            return 1 if risk < 0.48 else 2
        return 0

