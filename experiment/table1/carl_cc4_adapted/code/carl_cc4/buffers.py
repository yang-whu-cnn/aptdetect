from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CARLTransition:
    episode_seed: int
    source: str
    state: tuple[float, ...]
    action: int
    reward: float
    next_state: tuple[float, ...]
    duration: int


class RealSyntheticBuffer:
    SYNTHETIC_ROLLOUTS_PER_REAL = 8
    def __init__(self): self.real: list[CARLTransition] = []; self.synthetic: list[CARLTransition] = []

    def add_real(self, transition: CARLTransition) -> None:
        if transition.source != "real": raise ValueError("real buffer requires source=real")
        self.real.append(transition)

    def add_synthetic_rollouts(self, *, origin_episode_seed: int,
                               rollouts: list[list[CARLTransition]]) -> None:
        if len(rollouts) != self.SYNTHETIC_ROLLOUTS_PER_REAL:
            raise ValueError("CARL requires exactly 8 synthetic rollouts per real rollout")
        for rollout in rollouts:
            for transition in rollout:
                if transition.source != "synthetic" or transition.episode_seed != origin_episode_seed:
                    raise ValueError("synthetic buffer seed/source isolation violation")
                self.synthetic.append(transition)

    def fixed_ratio_batch(self, *, real_count: int, synthetic_per_real: int):
        if synthetic_per_real <= 0 or len(self.real) < real_count or len(self.synthetic) < real_count*synthetic_per_real:
            raise ValueError("insufficient samples for fixed synthetic ratio")
        return self.real[:real_count], self.synthetic[:real_count*synthetic_per_real]
