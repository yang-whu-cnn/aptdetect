# src/ppo_agent.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # logits for each feature index
        return self.net(s)


class CriticNet(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s).squeeze(-1)


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    ppo_epochs: int = 10
    device: str = "cpu"
    k_flip: int = 3


class PPOAgent:
    def __init__(self, state_dim: int, action_dim: int, cfg: PPOConfig):
        self.cfg = cfg
        self.actor = ActorNet(state_dim, action_dim).to(cfg.device)
        self.critic = CriticNet(state_dim).to(cfg.device)

        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

    @torch.no_grad()
    def act(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        输出：flip_indices（≤k），以及 log_prob（用于PPO更新）
        """
        s = torch.tensor(state, dtype=torch.float32, device=self.cfg.device).unsqueeze(0)
        logits = self.actor(s).squeeze(0)  # (D,)
        probs = torch.sigmoid(logits)

        # 选择 top-k 作为翻转位（简单稳定，后续也可改采样）
        topk = torch.topk(probs, k=self.cfg.k_flip).indices.cpu().numpy()

        # 近似log_prob：取所选位的log(p)
        logp = torch.log(probs[topk] + 1e-8).sum().item()
        return topk.astype(np.int64), float(logp)

    def _compute_gae(self, rewards, values, dones, last_value):
        adv = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.cfg.gamma * last_value * mask - values[t]
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * mask * gae
            adv[t] = gae
            last_value = values[t]
        returns = adv + values
        return adv, returns

    def update(self, traj: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        traj keys:
          states: (N,state_dim)
          actions: (N,k)
          old_logp: (N,)
          rewards: (N,)
          dones: (N,)
        """
        device = self.cfg.device
        states = torch.tensor(traj["states"], dtype=torch.float32, device=device)
        actions = traj["actions"]  # numpy
        old_logp = torch.tensor(traj["old_logp"], dtype=torch.float32, device=device)
        rewards = traj["rewards"].astype(np.float32)
        dones = traj["dones"].astype(np.float32)

        with torch.no_grad():
            values = self.critic(states).cpu().numpy()
            last_value = 0.0
            adv, returns = self._compute_gae(rewards, values, dones, last_value)
            adv = (adv - adv.mean()) / (adv.std() + 1e-6)

        adv_t = torch.tensor(adv, dtype=torch.float32, device=device)
        ret_t = torch.tensor(returns, dtype=torch.float32, device=device)

        # PPO多轮更新
        for _ in range(self.cfg.ppo_epochs):
            logits = self.actor(states)           # (N,D)
            probs = torch.sigmoid(logits)         # (N,D)
            v = self.critic(states)               # (N,)

            # 计算新log_prob（按top-k位累加）
            new_logp_list = []
            for i in range(len(states)):
                idx = actions[i]
                p = probs[i, idx]
                new_logp_list.append(torch.log(p + 1e-8).sum())
            new_logp = torch.stack(new_logp_list, dim=0)

            ratio = torch.exp(new_logp - old_logp)
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * adv_t
            loss_actor = -torch.min(surr1, surr2).mean()

            loss_critic = F.mse_loss(v, ret_t)

            self.opt_actor.zero_grad()
            loss_actor.backward()
            self.opt_actor.step()

            self.opt_critic.zero_grad()
            loss_critic.backward()
            self.opt_critic.step()

        return {"loss_actor": float(loss_actor.item()), "loss_critic": float(loss_critic.item())}
