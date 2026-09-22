# src/ppo_posterior.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class PPOConfig:
    lr: float = 3e-4
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    gae_lambda: float = 0.95
    gamma: float = 0.99
    rollout_len: int = 128
    minibatch_size: int = 256
    update_epochs: int = 4


class EvidenceScoreNet(nn.Module):
    """f_theta(e_i)：把前瞻证据向量映射成候选评分"""
    def __init__(self, e_dim: int = 4, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(e_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, E: torch.Tensor) -> torch.Tensor:
        # E: (B, N, e_dim) -> (B, N)
        return self.net(E).squeeze(-1)


class ActorCritic(nn.Module):
    def __init__(self, n_candidates: int, e_dim: int = 4, hidden: int = 64):
        super().__init__()
        self.n_candidates = n_candidates
        self.score_net = EvidenceScoreNet(e_dim=e_dim, hidden=hidden)

        # critic：用候选证据的池化统计（mean/max）估计 V
        self.v = nn.Sequential(
            nn.Linear(e_dim * 2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

        # 可学习融合系数 β（调节先验项的尺度）
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, prior_logits: torch.Tensor, E: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回：action logits（B,N），value（B,）"""
        f = self.score_net(E)  # (B,N)
        logits = self.beta * prior_logits + f

        e_mean = E.mean(dim=1)
        e_max = E.max(dim=1).values
        v_in = torch.cat([e_mean, e_max], dim=1)
        value = self.v(v_in).squeeze(-1)
        return logits, value


def compute_gae(rewards, dones, values, gamma, lam):
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    lastgaelam = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]
        next_value = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        adv[t] = lastgaelam
    ret = adv + values[:T]
    return adv, ret


class PPOPosteriorAgent:
    """PPO训练“选择候选计划索引”的后验策略。"""
    def __init__(self, n_candidates: int, cfg: PPOConfig, device: str = "cpu"):
        self.n = int(n_candidates)
        self.cfg = cfg
        self.device = torch.device(device)
        self.net = ActorCritic(n_candidates=self.n).to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=cfg.lr)

    @torch.no_grad()
    def act(self, prior_logits: np.ndarray, E: np.ndarray, deterministic: bool = False):
        pl = torch.tensor(prior_logits[None, :], dtype=torch.float32, device=self.device)
        Et = torch.tensor(E[None, :, :], dtype=torch.float32, device=self.device)
        logits, value = self.net(pl, Et)
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            a = torch.argmax(logits, dim=1)
        else:
            a = dist.sample()
        logp = dist.log_prob(a)
        return int(a.item()), float(logp.item()), float(value.item())

    def update(self, traj: Dict[str, np.ndarray]):
        cfg = self.cfg
        T, N = traj["prior_logits"].shape
        assert N == self.n

        with torch.no_grad():
            pl = torch.tensor(traj["prior_logits"], dtype=torch.float32, device=self.device)
            E = torch.tensor(traj["E"], dtype=torch.float32, device=self.device)
            _, values = self.net(pl, E)
            values_np = values.cpu().numpy()

        adv, ret = compute_gae(
            rewards=traj["rewards"],
            dones=traj["dones"],
            values=values_np,
            gamma=cfg.gamma,
            lam=cfg.gae_lambda,
        )
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        pl = torch.tensor(traj["prior_logits"], dtype=torch.float32, device=self.device)
        E = torch.tensor(traj["E"], dtype=torch.float32, device=self.device)
        actions = torch.tensor(traj["actions"], dtype=torch.long, device=self.device)
        logp_old = torch.tensor(traj["logp_old"], dtype=torch.float32, device=self.device)
        adv_t = torch.tensor(adv, dtype=torch.float32, device=self.device)
        ret_t = torch.tensor(ret, dtype=torch.float32, device=self.device)

        for _ in range(cfg.update_epochs):
            idx = torch.randperm(T, device=self.device)
            for start in range(0, T, cfg.minibatch_size):
                mb = idx[start:start+cfg.minibatch_size]
                logits, value = self.net(pl[mb], E[mb])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(actions[mb])
                entropy = dist.entropy().mean()

                ratio = torch.exp(logp - logp_old[mb])
                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv_t[mb]
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = ((value - ret_t[mb]) ** 2).mean()
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.opt.step()

    def save(self, path: str):
        torch.save(self.net.state_dict(), path)

    def load(self, path: str):
        self.net.load_state_dict(torch.load(path, map_location=self.device))
