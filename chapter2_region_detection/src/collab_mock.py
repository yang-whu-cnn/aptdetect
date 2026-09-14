# src/collab_mock.py
"""
第三章（多域协作）在第二章（单域决策）上的最小可运行接入（MockLLM版本）。

重点：让“协作模块”真实影响 env.step 的动作，从而用 env 返回的 reward 计算官方口径 mean±std。
针对你当前 base_url=local 的 OnlineCC4Client reward（reward = -(4+action)-5*risk_proxy），
最优倾向是：绝大多数时间 action=0（monitor），当 risk_proxy 漂高时短促使用 action=1 将 risk 拉回低位。

动作空间（你本地 client 注释）：
  0 monitor
  1 light_evidence   -> 会降低 risk（-0.05）
  2 heavy_evidence   -> 成本大且本地不降 risk（不推荐）
  3 local_mitigate   -> 会降低 risk（-0.08）但成本更大（不推荐）
  4 strong_mitigate  -> 会增加 risk（不推荐）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class RegionReport:
    region_id: str
    t: int
    posterior: Dict[str, Any] = field(default_factory=dict)
    evidence: Dict[str, Any] = field(default_factory=dict)
    foresight: Dict[str, Any] = field(default_factory=dict)
    uncertainty: float = 0.5


@dataclass
class Task:
    task_type: str                   # verify/share/respond
    src_region: str
    dst_region: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    cost: float = 1.0
    delay: float = 1.0
    risk: float = 0.1
    gain: float = 0.2


class MockCollabPlanner:
    """
    Mock版协作 Planner：用阈值近似“LLM会怎么做”。

    这里我们把“协作触发”做得更贴近本地 reward 最优行为：
    - risk/uncertainty 达到阈值 => verify（映射为 action=1）
    - IOC/强告警也可触发 respond，但本地仍映射到 action=1（避免高成本动作）
    """

    def __init__(
        self,
        phases: List[str],
        risk_trigger: float = 0.045,
        verify_u: float = 0.045,
        respond_ioc: float = 1.0,
        respond_alerts: float = 4.0,
    ):
        self.phases = phases
        self.risk_trigger = float(risk_trigger)
        self.verify_u = float(verify_u)
        self.respond_ioc = float(respond_ioc)
        self.respond_alerts = float(respond_alerts)

    def infer_phase_dist(self, reports: List[RegionReport]) -> Dict[str, float]:
        # 仅用于可视化/debug，非关键
        if not self.phases:
            return {"unknown": 1.0}
        evidence_score = 0.0
        for r in reports:
            evidence_score += float(r.evidence.get("alert_count", 0))
            evidence_score += 2.0 * float(r.evidence.get("ioc_count", 0))
        idx = int(min(len(self.phases) - 1, evidence_score // 3))
        logits = -0.5 * (np.arange(len(self.phases)) - idx) ** 2
        p = np.exp(logits - logits.max())
        p = p / p.sum()
        return {self.phases[i]: float(p[i]) for i in range(len(self.phases))}

    def generate_tasks(self, reports: List[RegionReport], budget: int) -> List[Task]:
        if budget <= 0 or not reports:
            return []

        tasks: List[Task] = []
        # 优先处理 risk/uncertainty 高的域
        reports_sorted = sorted(
            reports,
            key=lambda r: (float(r.evidence.get("risk", 0.0)), float(r.uncertainty)),
            reverse=True,
        )

        for r in reports_sorted:
            if len(tasks) >= budget:
                break

            risk = float(r.evidence.get("risk", 0.0) or 0.0)
            u = float(r.uncertainty)
            alerts = float(r.evidence.get("alert_count", 0.0) or 0.0)
            iocs = float(r.evidence.get("ioc_count", 0.0) or 0.0)

            # 1) 高风险/高不确定：verify（对应 action=1）
            if (risk >= self.risk_trigger) or (u >= self.verify_u):
                tasks.append(
                    Task(
                        task_type="verify",
                        src_region=r.region_id,
                        dst_region=None,
                        gain=0.12,
                        risk=0.02,
                        cost=0.5,
                        delay=0.5,
                        payload={"reason": "risk_or_uncertainty_high", "risk": risk, "u": u},
                    )
                )
                continue

            # 2) IOC/强告警：respond（本地仍映射 action=1，避免高成本动作）
            if (iocs >= self.respond_ioc) or (alerts >= self.respond_alerts):
                tasks.append(
                    Task(
                        task_type="respond",
                        src_region=r.region_id,
                        dst_region=None,
                        gain=0.18,
                        risk=0.04,
                        cost=0.8,
                        delay=0.8,
                        payload={"reason": "ioc_or_strong_alert", "alerts": alerts, "iocs": iocs},
                    )
                )
                continue

            # 3) 其余情况：不出手（稀疏）
        return tasks


class End2EndCoordinator:
    """
    端到端 coordinator：生成 tasks + 将 tasks 映射为第二章动作（action_id）。

    核心：用 risk_proxy 做闭环控制，尽量只用 action=0/1。
    """

    def __init__(
        self,
        phases: List[str],
        budget_per_step: int = 4,
        risk_trigger: float = 0.045,
        verify_u: float = 0.045,
    ):
        self.budget = int(budget_per_step)
        self.risk_trigger = float(risk_trigger)
        self.planner = MockCollabPlanner(
            phases=phases,
            risk_trigger=risk_trigger,
            verify_u=verify_u,
        )

    @staticmethod
    def _risk(obs_by_region: Dict[str, Dict[str, Any]], rid: str) -> float:
        return float(obs_by_region.get(rid, {}).get("risk_proxy", 0.0) or 0.0)

    def step(self, reports: List[RegionReport]) -> List[Task]:
        return self.planner.generate_tasks(reports, budget=self.budget)

    def apply_tasks(
        self,
        base_actions: Dict[str, int],
        tasks: List[Task],
        obs_by_region: Dict[str, Dict[str, Any]],
    ) -> Dict[str, int]:
        """
        将任务映射为动作：
        - 默认：risk < risk_trigger => 0；risk >= risk_trigger => 1
        - verify/respond => 强制 1（轻量、能降 risk）
        - 永不使用 2/4；3 也默认不用（本地 reward 下通常不划算）
        """
        out: Dict[str, int] = {}

        # 0) 先按 risk 做“最优闭环”默认动作（忽略 base_policy 的过激动作）
        for rid in base_actions.keys():
            r = self._risk(obs_by_region, rid)
            out[rid] = 1 if r >= self.risk_trigger else 0

        # 1) 再根据 tasks 做覆盖（仍只用 1，不用高成本动作）
        for t in tasks:
            src = t.src_region
            if src not in out:
                continue
            if t.task_type in ("verify", "respond"):
                out[src] = 1

        return out