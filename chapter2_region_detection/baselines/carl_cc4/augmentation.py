from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ImaginationGate:
    status: str
    requested_horizon: int
    effective_horizon: int | None
    formal_eligible: bool
    disclosure: str


def gate_imagination_horizon(*, requested_horizon: int = 256, frozen_wm_horizon: int = 4,
                             long_horizon_gate_passed: bool = False,
                             allow_disclosed_truncation: bool = False) -> ImaginationGate:
    if requested_horizon <= frozen_wm_horizon or long_horizon_gate_passed:
        return ImaginationGate("PASS", requested_horizon, requested_horizon, True, "validated horizon")
    if allow_disclosed_truncation:
        return ImaginationGate("ADAPTED_TRUNCATED", requested_horizon, frozen_wm_horizon, False,
                               "H=256 blocked; explicitly truncated to frozen WM H=4")
    return ImaginationGate("BLOCKED", requested_horizon, None, False,
                           "frozen WM H=4 has no validated H=256 extrapolation")


class TrainingTimeAugmentor:
    """Dyna/SimPLe augmentation only; intentionally exposes no plan/act method."""
    def __init__(self, world_model, reward_scm, gate: ImaginationGate):
        if gate.status == "BLOCKED": raise RuntimeError(gate.disclosure)
        self.world_model = world_model; self.reward_scm = reward_scm; self.gate = gate
        self.training = True

    def freeze_for_evaluation(self):
        self.training = False
