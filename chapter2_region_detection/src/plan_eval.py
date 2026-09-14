# src/plan_eval.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from src.action_space import action_cost, action_intensity
from src.state_summary import checkpoint_satisfied
from src.world_model import DynamicsEnsemble


@dataclass
class Evidence:
    """前瞻证据向量 e_i = [G_i, U_i, C_i, Cost_i]
    - G_i：预测综合评估量（收益-代价，折扣累积），用集成均值
    - U_i：不确定性，用集成成员对同一计划的预测回报分歧（方差）
    - C_i：一致性（检查点覆盖率）
    - Cost_i：累计动作开销
    """
    G: float
    U: float
    C: float
    Cost: float

    def as_vec(self) -> np.ndarray:
        return np.array([self.G, self.U, self.C, self.Cost], dtype=np.float32)


def compute_early_gain(risk_seq: np.ndarray, high_thr: float) -> Tuple[float, int]:
    """EarlyGain：首次达到高风险阈值越早，收益越大；未达到则0。"""
    idx = np.where(risk_seq >= high_thr)[0]
    if len(idx) == 0:
        return 0.0, -1
    t_hit = int(idx[0])
    early = 1.0 / (1.0 + t_hit)
    return float(early), t_hit


def compute_fprisk(risk: float, intensity: int, low_thr: float, strong_thr: int) -> float:
    if (risk <= low_thr) and (intensity >= strong_thr):
        return float(intensity ** 2)
    return 0.0


def _plan_return_from_risk(risk_seq, plan, cfg):
    # ---- 1) 读取 plan_score 参数（都给默认值，避免再 KeyError）----
    ps = cfg.get("plan_score", {}) or {}
    gamma = float(ps.get("gamma", 0.9))
    lambda_risk = float(ps.get("lambda_risk", 1.0))
    lambda_cost = float(ps.get("lambda_cost", 0.2))
    lambda_delay = float(ps.get("lambda_delay", 0.1))
    delay_scale = float(ps.get("delay_scale", 1.0))

    # ---- 2) 兼容：cfg 里没有 action_space 时自动造一个 ----
    act_space = cfg.get("action_space", None)
    if act_space is None:
        # action_dim 优先从 world_model.action_dim 取；取不到就从 plan 的最大动作 id 推断
        wm_cfg = cfg.get("world_model", {}) or {}
        action_dim = wm_cfg.get("action_dim", None)
        if action_dim is None:
            try:
                action_dim = int(max(plan)) + 1
            except Exception:
                action_dim = 1

        # 默认：每个动作 cost=1，delay=1（你后面也可以在 yaml 里显式覆盖）
        action_costs = [1.0] * int(action_dim)
        action_delays = [1.0] * int(action_dim)
    else:
        # 允许两种写法：
        # action_space:
        #   costs: [..]
        #   delays: [..]
        action_costs = act_space.get("costs", None)
        action_delays = act_space.get("delays", None)

        # 如果没配就给默认值
        wm_cfg = cfg.get("world_model", {}) or {}
        action_dim = wm_cfg.get("action_dim", None)
        if action_dim is None:
            try:
                action_dim = int(max(plan)) + 1
            except Exception:
                action_dim = 1

        if action_costs is None:
            action_costs = [1.0] * int(action_dim)
        if action_delays is None:
            action_delays = [1.0] * int(action_dim)

    # ---- 3) 计算折扣回报：只用 risk + cost + delay（足够让 posterior 先跑通）----
    # risk_seq: shape [H] or list
    G = 0.0
    for h, a in enumerate(plan):
        r = -lambda_risk * float(risk_seq[h])
        a_int = int(a)
        if 0 <= a_int < len(action_costs):
            r -= lambda_cost * float(action_costs[a_int])
        else:
            r -= lambda_cost * 1.0

        if 0 <= a_int < len(action_delays):
            r -= lambda_delay * float(action_delays[a_int]) * delay_scale
        else:
            r -= lambda_delay * 1.0 * delay_scale

        G += (gamma ** h) * r

    return float(G)


def evaluate_plan_with_model(
        model: DynamicsEnsemble,
        s0: np.ndarray,
        plan: List[int],
        checkpoints: List[str],
        cfg: Dict,
) -> Evidence:
    wm = cfg["world_model"]
    # 兼容：world_model.rollout_horizon 可能缺失
    wm_cfg = cfg.get("world_model", {}) or {}
    ps_cfg = cfg.get("plan_score", {}) or {}
    lp_cfg = cfg.get("llm_prior", {}) or {}

    horizon = int(
        wm_cfg.get(
            "rollout_horizon",
            lp_cfg.get(
                "plan_horizon",
                ps_cfg.get("plan_horizon", len(plan) if "plan" in locals() else 5),
            ),
        )
    )

    # 兼容：world_model.risk_index 可能缺失
    wm_cfg = cfg.get("world_model", {}) or {}

    # 1) 优先读配置
    risk_idx = wm_cfg.get("risk_index", None)

    # 2) 没配则尝试从 summary/raw_stats 推断（如果你们把 risk_proxy 放进去）
    #    常见做法：state vec 最后一维 / 或固定某一维作为 risk 代理
    if risk_idx is None:
        # 优先使用显式的风险代理 key：risk_proxy
        # 如果你的 StateSummary.vec 有固定布局，先用最后一维更稳
        risk_idx = wm_cfg.get("state_dim", None)
        if isinstance(risk_idx, int) and risk_idx > 0:
            risk_idx = risk_idx - 1
        else:
            risk_idx = -1  # 实在不行就用最后一维

    risk_idx = int(risk_idx)

    # 1) 集成均值轨迹：用于一致性C（检查点覆盖率）
    traj_mean, _ = model.rollout(s0, plan, horizon=horizon)
    risk_mean_seq = traj_mean[1:, risk_idx]  # (H,)

    # 2) 预测综合评估量 G：用“集成成员”分别计算回报，再取均值
    G_members = []
    for mi in range(len(model.models)):
        s = s0.astype(np.float32)
        risk_seq = []
        for t in range(horizon):
            a = int(plan[t])
            s = model.predict_next_by_member(mi, s, a)
            risk_seq.append(float(s[risk_idx]))
        Gm = _plan_return_from_risk(np.array(risk_seq, dtype=np.float32), plan, cfg)
        G_members.append(Gm)

    G = float(np.mean(G_members))
    U = float(np.var(G_members))  # 预测回报分歧（方差）

    # 3) Cost：累计动作开销（与模型无关）
    cost_sum = float(np.sum([action_cost(int(a)) for a in plan]))

    # 4) 一致性 C：检查点覆盖率（预测窗口内任意时刻满足即覆盖）
    covered = 0
    for cp in checkpoints:
        ok = False
        for t in range(1, horizon + 1):
            pseudo_stats = {
                "cnt_auth_fail": float(traj_mean[t, 0]),
                "cnt_port_scan": float(traj_mean[t, 1]),
                "cnt_proc_spawn": float(traj_mean[t, 2]),
                "cnt_outbound_conn": float(traj_mean[t, 3]),
            }
            if checkpoint_satisfied(cp, pseudo_stats):
                ok = True
                break
        covered += 1 if ok else 0
    C = float(covered / max(1, len(checkpoints)))

    return Evidence(G=G, U=U, C=C, Cost=cost_sum)


def score_plans(plans, world_model, s_vec, cfg, risk_proxy=0.0):
    """
    env_region.build_evidence() 期望的接口：
      plan_scores, checkpoints = plan_eval.score_plans(...)
    """
    import numpy as np
    scores = []
    checkpoints = []
    for plan in plans:
        G, R, C, D, ret = evaluate_plan_with_model(world_model, s_vec, plan, cfg, risk_proxy)
        scores.append(float(ret))
        checkpoints.append({
            "EarlyGain": float(G),
            "FPRisk": float(R),
            "ActCost": float(C),
            "Delay": float(D),
            "return": float(ret),
            "plan": list(plan),
        })
    return np.asarray(scores, dtype=np.float32), checkpoints
