from __future__ import annotations

from dataclasses import dataclass


ACTION_COST = {0: 0.0, 1: 0.0, 2: 1.0, 3: 1.0}


@dataclass(frozen=True)
class CAICSRewardConfig:
    alpha_comp: float = 0.025
    beta_comp: float = 2.0
    cleared_terminal_bonus: float = 1000.0
    severe_failure_terminal_penalty: float = -1000.0
    isolate_state_weight: float = 0.0
    isolate_change_weight: float = 0.0


def caics_reward(*, active_incidents: int, incident_change: int, action_id: int,
                 attack_cleared_terminal: bool = False,
                 severe_failure_terminal: bool = False,
                 config: CAICSRewardConfig | None = None) -> float:
    """Explicit standard-CAICS-to-CC4 training reward mapping."""
    cfg = config or CAICSRewardConfig()
    if active_incidents < 0 or action_id not in ACTION_COST:
        raise ValueError("invalid incident count or A4 action")
    if attack_cleared_terminal and severe_failure_terminal:
        raise ValueError("terminal outcomes are mutually exclusive")
    value = (-cfg.alpha_comp * active_incidents
             - cfg.beta_comp * max(0, int(incident_change))
             - ACTION_COST[action_id])
    # No isolate action exists in A4; both isolate terms are frozen to zero.
    if attack_cleared_terminal:
        value += cfg.cleared_terminal_bonus
    if severe_failure_terminal:
        value += cfg.severe_failure_terminal_penalty
    return float(value)
