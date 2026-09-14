# src/world_model.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ProbDynamicsNet(nn.Module):
    """对角高斯的一步转移模型：输入(s,a)->输出 next_state 的 mean/logvar
    说明（对齐方法文档）：
    - 用小规模集成（ensemble）近似学习 CC4 单区域环境的转移 dynamics
    - 集成内部的“预测分歧”可作为不确定性（U）的代理
    """
    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions
        in_dim = state_dim + n_actions

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, state_dim)
        self.logvar = nn.Linear(hidden_dim, state_dim)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # a: (B,) -> onehot
        oh = torch.zeros((s.size(0), self.n_actions), device=s.device)
        oh.scatter_(1, a.view(-1, 1), 1.0)
        x = torch.cat([s, oh], dim=1)
        h = self.net(x)
        mu = self.mu(h)
        logvar = torch.clamp(self.logvar(h), -10.0, 3.0)
        return mu, logvar


def gaussian_nll(x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # 对角高斯NLL（忽略常数项）
    return 0.5 * (logvar + (x - mu) ** 2 / torch.exp(logvar)).mean()


@dataclass
class EnsembleConfig:
    ensemble_size: int = 3
    hidden_dim: int = 128
    lr: float = 5e-4
    batch_size: int = 256
    epochs: int = 8


class DynamicsEnsemble:
    """集成世界模型：用于前瞻滚动、并输出预测分歧/方差。"""
    def __init__(self, state_dim: int, n_actions: int, cfg: EnsembleConfig, device: str = "cpu"):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.cfg = cfg
        self.device = torch.device(device)

        self.models: List[ProbDynamicsNet] = [
            ProbDynamicsNet(state_dim, n_actions, hidden_dim=cfg.hidden_dim).to(self.device)
            for _ in range(cfg.ensemble_size)
        ]
        self.opts = [optim.Adam(m.parameters(), lr=cfg.lr) for m in self.models]

    def train(self, s: np.ndarray, a: np.ndarray, s2: np.ndarray):
        s_t = torch.tensor(s, dtype=torch.float32, device=self.device)
        a_t = torch.tensor(a, dtype=torch.long, device=self.device)
        s2_t = torch.tensor(s2, dtype=torch.float32, device=self.device)

        n = s_t.size(0)
        bs = self.cfg.batch_size

        for m in self.models:
            m.train()

        for _ in range(self.cfg.epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, bs):
                idx = perm[i:i+bs]
                sb, ab, s2b = s_t[idx], a_t[idx], s2_t[idx]
                for model, opt in zip(self.models, self.opts):
                    mu, logvar = model(sb, ab)
                    loss = gaussian_nll(s2b, mu, logvar)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

        for m in self.models:
            m.eval()

    @torch.no_grad()
    def predict_next(self, s: np.ndarray, a: int) -> Tuple[np.ndarray, np.ndarray]:
        """集成预测：返回 mean_next 与 total_var（epistemic+aleatoric）。"""
        s_t = torch.tensor(s[None, :], dtype=torch.float32, device=self.device)
        a_t = torch.tensor([a], dtype=torch.long, device=self.device)

        mus = []
        vars_ = []
        for model in self.models:
            mu, logvar = model(s_t, a_t)
            mus.append(mu.squeeze(0).cpu().numpy())
            vars_.append(torch.exp(logvar).squeeze(0).cpu().numpy())
        mus = np.stack(mus, axis=0)
        vars_ = np.stack(vars_, axis=0)

        mean_mu = mus.mean(axis=0)
        epistemic = mus.var(axis=0)           # 预测分歧（模型间均值方差）
        aleatoric = vars_.mean(axis=0)        # 模型内部噪声
        total_var = epistemic + aleatoric
        return mean_mu.astype(np.float32), total_var.astype(np.float32)

    @torch.no_grad()
    def predict_next_by_member(self, member_idx: int, s: np.ndarray, a: int) -> np.ndarray:
        """单个成员模型的均值预测，用于计算“预测回报分歧”。"""
        m = self.models[int(member_idx)]
        s_t = torch.tensor(s[None, :], dtype=torch.float32, device=self.device)
        a_t = torch.tensor([a], dtype=torch.long, device=self.device)
        mu, _ = m(s_t, a_t)
        return mu.squeeze(0).cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def rollout(self, s0: np.ndarray, plan: List[int], horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """沿计划做集成均值滚动预测，输出预测轨迹 states (horizon+1, D) 和 每步方差 vars (horizon, D)。"""
        s = s0.astype(np.float32)
        traj = [s.copy()]
        var_traj = []
        for t in range(horizon):
            a = int(plan[t])
            s_mean, s_var = self.predict_next(s, a)
            traj.append(s_mean)
            var_traj.append(s_var)
            s = s_mean
        return np.stack(traj, axis=0), np.stack(var_traj, axis=0)
