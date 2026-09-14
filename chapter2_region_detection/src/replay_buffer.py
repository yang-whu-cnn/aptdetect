# src/replay_buffer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    s2: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 200000):
        self.capacity = int(capacity)
        self.buf: List[Transition] = []
        self.pos = 0

    def __len__(self):
        return len(self.buf)

    def add(self, tr: Transition):
        if len(self.buf) < self.capacity:
            self.buf.append(tr)
        else:
            self.buf[self.pos] = tr
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = np.random.randint(0, len(self.buf), size=batch_size)
        s = np.stack([self.buf[i].s for i in idx]).astype(np.float32)
        a = np.array([self.buf[i].a for i in idx], dtype=np.int64)
        r = np.array([self.buf[i].r for i in idx], dtype=np.float32)
        s2 = np.stack([self.buf[i].s2 for i in idx]).astype(np.float32)
        d = np.array([self.buf[i].done for i in idx], dtype=np.float32)
        return s, a, r, s2, d
