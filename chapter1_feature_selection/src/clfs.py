# src/clfs.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class ContrastivePairsDataset(Dataset):
    """构造正样本：相邻窗口 (x_t, x_{t+1})"""
    def __init__(self, X: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self) -> int:
        return max(0, len(self.X) - 1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.X[idx + 1]


class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ProjectionHead(nn.Module):
    def __init__(self, embed_dim: int = 128, proj_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, proj_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """
    轻量NT-Xent：
    - 2B样本拼接
    - 每个样本的正样本是它的paired view
    """
    B = z1.size(0)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)  # (2B, D)

    sim = torch.matmul(z, z.T) / temperature  # (2B,2B)
    mask = torch.eye(2 * B, device=z.device).bool()
    sim = sim.masked_fill(mask, -1e9)

    # 正样本索引：i 的正样本是 i+B（或 i-B）
    pos = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    loss = F.cross_entropy(sim, pos)
    return loss


@dataclass
class CLFSModelBundle:
    encoder: MLPEncoder
    projector: ProjectionHead


def train_clfs(
    X: np.ndarray,
    embed_dim: int = 128,
    proj_dim: int = 64,
    lr: float = 1e-3,
    batch_size: int = 256,
    epochs: int = 10,
    temperature: float = 0.2,
    device: str = "cpu",
) -> CLFSModelBundle:
    ds = ContrastivePairsDataset(X)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    encoder = MLPEncoder(X.shape[1], embed_dim=embed_dim).to(device)
    projector = ProjectionHead(embed_dim=embed_dim, proj_dim=proj_dim).to(device)

    opt = torch.optim.Adam(list(encoder.parameters()) + list(projector.parameters()), lr=lr)

    encoder.train()
    projector.train()

    for ep in range(epochs):
        total = 0.0
        n = 0
        for x1, x2 in tqdm(dl, desc=f"CL-FS epoch {ep+1}/{epochs}"):
            x1 = x1.to(device)
            x2 = x2.to(device)

            z1 = projector(encoder(x1))
            z2 = projector(encoder(x2))

            loss = nt_xent_loss(z1, z2, temperature=temperature)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += float(loss.item())
            n += 1
        print(f"[CL-FS] epoch={ep+1} loss={total/max(n,1):.4f}")

    return CLFSModelBundle(encoder=encoder, projector=projector)


@torch.no_grad()
def encode_windows(encoder: MLPEncoder, X: np.ndarray, device: str = "cpu", batch_size: int = 512) -> np.ndarray:
    encoder.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    Z = []
    for i in range(0, len(X_t), batch_size):
        z = encoder(X_t[i:i+batch_size].to(device)).cpu().numpy()
        Z.append(z)
    return np.vstack(Z).astype(np.float32)
