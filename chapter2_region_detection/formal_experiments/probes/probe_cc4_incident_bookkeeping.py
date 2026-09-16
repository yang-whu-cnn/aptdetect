from __future__ import annotations

import argparse
import json
import os
import sys

from dataclasses import dataclass
from pathlib import Path
from typing import Any


THIS = Path(__file__).resolve()
PROJ = THIS.parents[2]

if str(PROJ) not in sys.path:
    sys.path.insert(
        0,
        str(PROJ),
    )


CYBORG_ROOT = os.environ.get(
    "CYBORG_ROOT",
    r"D:\python1\cage-challenge-4",
)

if (
    CYBORG_ROOT
    and CYBORG_ROOT not in sys.path
):
    sys.path.insert(
        0,
        CYBORG_ROOT,
    )


from CybORG import CybORG  # type: ignore

from CybORG.Agents import (  # type: ignore
    SleepAgent,
)

from CybORG.Shared.BlueRewardMachine import (  # type: ignore
    BlueRewardMachine,
)

from CybORG.Simulator.Scenarios import (  # type: ignore
    EnterpriseScenarioGenerator,
)

from CybORG.Simulator.Actions.GreenActions.GreenLocalWork import (  # type: ignore
    GreenLocalWork,
)

from CybORG.Simulator.Actions.AbstractActions import (  # type: ignore
    Restore,
)


PROBE_SUBNET = (
    "restricted_zone_a_subnet"
)

BLUE_AGENT = (
    "blue_agent_0"
)


@dataclass
class IncidentBookkeeping:
    """
    仅供 reward / evaluation bookkeeping。

    这里使用的 red presence 属于 hidden truth，
    后续绝不能进入：

    - FormalStateEncoder
    - LLM input
    - PPO input
    - CEM / UG-CEM planner state

    A3.5 的目的就是先把这条边界冻结。
    """

    incident_event_id: str
    incident_host_id: str

    t_compromise: int | None = None
    t_normal: int | None = None

    incident_host_lwf_count: int = 0
    incident_host_lwf_penalty: float = 0.0

    def record_presence_transition(
        self,
        *,
        before: bool,
        after: bool,
        tick_after: int,
    ) -> None:

        if (
            self.t_compromise is None
            and not before
            and after
        ):
            self.t_compromise = int(
                tick_after
            )

        if (
            self.t_compromise is not None
            and self.t_normal is None
            and before
            and not after
        ):
            self.t_normal = int(
                tick_after
            )

    def record_local_work_failure(
        self,
        *,
        hostname: str,
        failed: bool,
        raw_lwf_penalty: float,
    ) -> bool:
        """
        只统计当前 incident host。

        返回：
            本次 failure 是否计入
            incident-host metric。
        """

        if not failed:
            return False

        if (
            hostname
            != self.incident_host_id
        ):
            return False

        self.incident_host_lwf_count += 1

        self.incident_host_lwf_penalty += (
            float(
                raw_lwf_penalty
            )
        )

        return True

    @property
    def attack_eradication_time(
        self,
    ) -> int | None:

        if (
            self.t_compromise is None
            or self.t_normal is None
        ):
            return None

        return int(
            self.t_normal
            - self.t_compromise
        )


def make_cyborg(
    seed: int,
    steps: int,
):
    """
    A3.5 使用 SleepAgent 控制 background noise。

    注意：
    GreenLocalWork 是通过真实 CC4 action 执行的，
    只是为了 probe 可重复性，
    用 skip_valid_action_check=True
    显式注入到 Green SleepAgent。

    这不是正式训练环境配置。
    """

    sg = EnterpriseScenarioGenerator(
        blue_agent_class=(
            SleepAgent
        ),
        green_agent_class=(
            SleepAgent
        ),
        red_agent_class=(
            SleepAgent
        ),
        steps=int(
            steps
        ),
    )

    cyborg = CybORG(
        scenario_generator=sg,
        seed=int(
            seed
        ),
    )

    cyborg.reset(
        seed=int(
            seed
        )
    )

    return cyborg


def subnet_name_for_host(
    controller,
    hostname: str,
) -> str:

    subnet = (
        controller
        .state
        .hostname_subnet_map[
            hostname
        ]
    )

    value = getattr(
        subnet,
        "value",
        subnet,
    )

    return str(
        value
    )


def find_probe_hosts(
    controller,
) -> tuple[
    str,
    str,
]:
    """
    restricted_zone_a 至少有多个 user hosts。

    选两个不同 host：

    target_host:
        当前 incident host。

    other_host:
        用来证明其他 host 的 LWF
        虽会进入 official team reward，
        但不得计入当前 incident metric。
    """

    prefix = (
        PROBE_SUBNET
        + "_user_host_"
    )

    hosts = sorted(
        hostname
        for hostname
        in controller.state.hosts
        if hostname.startswith(
            prefix
        )
    )

    if len(hosts) < 2:
        raise RuntimeError(
            "A3.5 probe requires "
            "at least two user hosts "
            f"in {PROBE_SUBNET}; "
            f"got {hosts}"
        )

    return (
        hosts[0],
        hosts[1],
    )


def find_green_session(
    controller,
    hostname: str,
) -> tuple[
    str,
    int,
]:
    """
    找到指定 host 上真实的 Green agent/session。
    """

    host = (
        controller
        .state
        .hosts[
            hostname
        ]
    )

    for (
        agent_name,
        session_ids,
    ) in host.sessions.items():

        if (
            "green_agent"
            not in str(
                agent_name
            )
        ):
            continue

        for session_id in session_ids:

            session = (
                controller
                .state
                .sessions[
                    agent_name
                ].get(
                    session_id
                )
            )

            if (
                session is not None
                and bool(
                    session.active
                )
            ):
                return (
                    str(
                        agent_name
                    ),
                    int(
                        session_id
                    ),
                )

    raise RuntimeError(
        f"no active Green session "
        f"on host {hostname}"
    )


def ground_truth_red_sessions(
    controller,
    hostname: str,
) -> list[
    dict[str, Any]
]:
    """
    HIDDEN TRUTH.

    只用于：
        reward/evaluation bookkeeping probe。

    不能作为 planner 输入。
    """

    host = (
        controller
        .state
        .hosts[
            hostname
        ]
    )

    out = []

    for (
        agent_name,
        session_ids,
    ) in host.sessions.items():

        if (
            "red_agent"
            not in str(
                agent_name
            )
        ):
            continue

        for session_id in session_ids:

            session = (
                controller
                .state
                .sessions
                .get(
                    agent_name,
                    {},
                )
                .get(
                    session_id
                )
            )

            if session is None:
                continue

            out.append(
                {
                    "agent":
                        str(
                            agent_name
                        ),

                    "session_id":
                        int(
                            session_id
                        ),

                    "username":
                        str(
                            session.username
                        ),

                    "active":
                        bool(
                            session.active
                        ),
                }
            )

    return out


def ground_truth_red_presence(
    controller,
    hostname: str,
) -> bool:
    """
    HIDDEN TRUTH bookkeeping only。
    """

    sessions = (
        ground_truth_red_sessions(
            controller,
            hostname,
        )
    )

    return any(
        bool(
            item[
                "active"
            ]
        )
        for item in sessions
    )


def set_active_services_reliability(
    controller,
    hostname: str,
    reliability: int,
) -> dict[
    str,
    int,
]:
    """
    Probe-only deterministic intervention。

    GreenLocalWork 是否失败由 service reliability
    决定。为了避免随机结果：

        100 -> 必定成功
        0   -> 必定失败

    正式实验绝不会修改 service reliability。
    """

    if reliability not in (
        0,
        100,
    ):
        raise ValueError(
            "probe reliability must "
            "be 0 or 100"
        )

    host = (
        controller
        .state
        .hosts[
            hostname
        ]
    )

    before = {}

    for (
        service_name,
        service,
    ) in host.services.items():

        if not bool(
            service.active
        ):
            continue

        before[
            str(
                service_name
            )
        ] = int(
            service
            .get_service_reliability()
        )

        # CC4 Service 本身没有直接 setter。
        # 本操作只属于 deterministic probe。
        service._percent_reliable = int(
            reliability
        )

    if not before:
        raise RuntimeError(
            f"{hostname}: "
            "no active services "
            "for GreenLocalWork probe"
        )

    return before


def blue_reward_snapshot(
    rewards,
    blue_agent: str,
) -> dict[
    str,
    Any,
]:
    if blue_agent not in rewards:
        raise RuntimeError(
            f"{blue_agent} missing "
            "from reward output"
        )

    raw = rewards[
        blue_agent
    ]

    components = {
        str(
            key
        ): float(
            value
        )
        for (
            key,
            value,
        ) in raw.items()
    }

    return {
        "components":
            components,

        "total":
            float(
                sum(
                    components.values()
                )
            ),

        "blue_reward_machine":
            float(
                components.get(
                    "BlueRewardMachine",
                    0.0,
                )
            ),

        "action_cost":
            float(
                components.get(
                    "action_cost",
                    0.0,
                )
            ),
    }


def progress_snapshot(
    controller,
    agent: str,
):
    item = (
        controller
        .actions_in_progress
        .get(
            agent
        )
    )

    if item is None:
        return None

    action = item[
        "action"
    ]

    return {
        "action_class":
            type(
                action
            ).__name__,

        "action_repr":
            str(
                action
            ),

        "duration":
            int(
                action.duration
            ),

        "remaining_ticks":
            int(
                item[
                    "remaining_ticks"
                ]
            ),
    }


def observation_success(
    observations,
    agent: str,
) -> tuple[
    bool,
    bool,
    str,
]:
    if agent not in observations:
        raise RuntimeError(
            f"{agent}: observation missing"
        )

    raw = (
        observations[
            agent
        ].get(
            "success"
        )
    )

    return (
        raw == True,
        raw == False,
        repr(
            raw
        ),
    )


def run_step(
    *,
    cyborg,
    blue_agent: str,
    actions: dict,
    skip_valid_action_check: bool,
):
    controller = (
        cyborg
        .environment_controller
    )

    tick_start = int(
        controller.step_count
    )

    submitted = {
        str(
            agent
        ): {
            "action_class":
                type(
                    action
                ).__name__,

            "action_repr":
                str(
                    action
                ),

            "duration":
                int(
                    action.duration
                ),
        }
        for (
            agent,
            action,
        ) in actions.items()
    }

    (
        observations,
        rewards,
        dones,
        _,
    ) = cyborg.parallel_step(
        actions=actions,
        skip_valid_action_check=bool(
            skip_valid_action_check
        ),
    )

    tick_end = int(
        controller.step_count
    )

    reward = (
        blue_reward_snapshot(
            rewards,
            blue_agent,
        )
    )

    return {
        "tick_start":
            tick_start,

        "tick_end":
            tick_end,

        "submitted":
            submitted,

        "blue_reward":
            reward,

        "done":
            bool(
                dones.get(
                    blue_agent,
                    False,
                )
            ),
    }, observations


def official_lwf_penalty(
    controller,
    hostname: str,
) -> float:

    subnet = (
        subnet_name_for_host(
            controller,
            hostname,
        )
    )

    phase = int(
        controller
        .state
        .mission_phase
    )

    brm = BlueRewardMachine(
        "Blue"
    )

    phase_rewards = (
        brm.get_phase_rewards(
            phase
        )
    )

    return float(
        phase_rewards[
            subnet
        ][
            "LWF"
        ]
    )


def make_green_local_work(
    controller,
    *,
    green_agent: str,
    green_session: int,
    hostname: str,
    phishing_error_rate: float,
):
    ip_address = (
        controller
        .hostname_ip_map[
            hostname
        ]
    )

    return GreenLocalWork(
        agent=green_agent,
        session_id=int(
            green_session
        ),
        ip_address=ip_address,
        fp_detection_rate=0.0,
        phishing_error_rate=float(
            phishing_error_rate
        ),
    )


def main():
    parser = (
        argparse.ArgumentParser(
            description=(
                "Probe CC4 incident-level "
                "compromise/recovery and "
                "incident-host LWF bookkeeping."
            )
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--out",
        default=(
            "outputs/cc4_probe/"
            "incident_bookkeeping.json"
        ),
    )

    args = parser.parse_args()

    cyborg = make_cyborg(
        seed=int(
            args.seed
        ),
        steps=int(
            args.steps
        ),
    )

    controller = (
        cyborg
        .environment_controller
    )

    (
        target_host,
        other_host,
    ) = find_probe_hosts(
        controller
    )

    (
        target_green_agent,
        target_green_session,
    ) = find_green_session(
        controller,
        target_host,
    )

    (
        other_green_agent,
        other_green_session,
    ) = find_green_session(
        controller,
        other_host,
    )

    incident_event_id = (
        f"seed-{args.seed}:"
        f"{target_host}:"
        "incident-0"
    )

    bookkeeping = (
        IncidentBookkeeping(
            incident_event_id=(
                incident_event_id
            ),
            incident_host_id=(
                target_host
            ),
        )
    )

    timeline = []

    official_team_return = 0.0

    # =========================================================
    # Stage 0
    # Initial ground truth
    # =========================================================

    initial_red = (
        ground_truth_red_presence(
            controller,
            target_host,
        )
    )

    if initial_red:
        raise RuntimeError(
            "target host is already "
            "compromised at probe start"
        )

    # =========================================================
    # Stage 1
    # Deterministically create one real CC4 phishing foothold.
    #
    # Healthy service reliability=100
    # -> GreenLocalWork succeeds
    # -> phishing_error_rate=1
    # -> real PhishingEmail sub-action
    # -> real RedAbstractSession
    # =========================================================

    set_active_services_reliability(
        controller,
        target_host,
        100,
    )

    before_red = (
        ground_truth_red_presence(
            controller,
            target_host,
        )
    )

    compromise_action = (
        make_green_local_work(
            controller,
            green_agent=(
                target_green_agent
            ),
            green_session=(
                target_green_session
            ),
            hostname=(
                target_host
            ),
            phishing_error_rate=1.0,
        )
    )

    (
        compromise_step,
        compromise_obs,
    ) = run_step(
        cyborg=cyborg,
        blue_agent=BLUE_AGENT,
        actions={
            target_green_agent:
                compromise_action
        },
        # Green is SleepAgent in this deterministic
        # probe; bypass only the action-space check.
        skip_valid_action_check=True,
    )

    after_red = (
        ground_truth_red_presence(
            controller,
            target_host,
        )
    )

    bookkeeping.record_presence_transition(
        before=before_red,
        after=after_red,
        tick_after=(
            compromise_step[
                "tick_end"
            ]
        ),
    )

    (
        compromise_success,
        compromise_failed,
        compromise_success_repr,
    ) = observation_success(
        compromise_obs,
        target_green_agent,
    )

    if not compromise_success:
        raise RuntimeError(
            "deterministic phishing "
            "GreenLocalWork did not succeed: "
            f"{compromise_success_repr}"
        )

    if not after_red:
        raise RuntimeError(
            "phishing probe did not "
            "create red presence "
            "on incident host"
        )

    official_team_return += (
        compromise_step[
            "blue_reward"
        ][
            "total"
        ]
    )

    timeline.append(
        {
            "stage":
                "compromise_created",

            **compromise_step,

            "incident_red_before":
                before_red,

            "incident_red_after":
                after_red,

            "incident_red_sessions":
                ground_truth_red_sessions(
                    controller,
                    target_host,
                ),

            "green_success":
                compromise_success,

            "green_failure":
                compromise_failed,
        }
    )

    # =========================================================
    # Stage 2
    # Force ONE incident-host GreenLocalWork failure.
    #
    # This must:
    # - contribute to official Blue reward;
    # - count as incident-host LWF.
    # =========================================================

    set_active_services_reliability(
        controller,
        target_host,
        0,
    )

    target_expected_lwf = (
        official_lwf_penalty(
            controller,
            target_host,
        )
    )

    target_lwf_action = (
        make_green_local_work(
            controller,
            green_agent=(
                target_green_agent
            ),
            green_session=(
                target_green_session
            ),
            hostname=(
                target_host
            ),
            phishing_error_rate=0.0,
        )
    )

    (
        target_lwf_step,
        target_lwf_obs,
    ) = run_step(
        cyborg=cyborg,
        blue_agent=BLUE_AGENT,
        actions={
            target_green_agent:
                target_lwf_action
        },
        skip_valid_action_check=True,
    )

    (
        target_success,
        target_failed,
        target_success_repr,
    ) = observation_success(
        target_lwf_obs,
        target_green_agent,
    )

    if not target_failed:
        raise RuntimeError(
            "incident-host "
            "GreenLocalWork was "
            "expected to fail: "
            f"{target_success_repr}"
        )

    target_observed_brm = float(
        target_lwf_step[
            "blue_reward"
        ][
            "blue_reward_machine"
        ]
    )

    if (
        target_observed_brm
        != target_expected_lwf
    ):
        raise RuntimeError(
            "incident-host LWF reward "
            "mismatch: "
            f"observed="
            f"{target_observed_brm}, "
            f"expected="
            f"{target_expected_lwf}"
        )

    target_counted = (
        bookkeeping
        .record_local_work_failure(
            hostname=target_host,
            failed=target_failed,
            raw_lwf_penalty=(
                target_expected_lwf
            ),
        )
    )

    if not target_counted:
        raise RuntimeError(
            "incident-host LWF "
            "was not counted"
        )

    if (
        bookkeeping
        .incident_host_lwf_count
        != 1
    ):
        raise RuntimeError(
            "incident-host LWF "
            "count must be 1"
        )

    if not ground_truth_red_presence(
        controller,
        target_host,
    ):
        raise RuntimeError(
            "incident disappeared "
            "before Restore"
        )

    official_team_return += (
        target_lwf_step[
            "blue_reward"
        ][
            "total"
        ]
    )

    timeline.append(
        {
            "stage":
                "incident_host_lwf",

            **target_lwf_step,

            "hostname":
                target_host,

            "expected_raw_lwf_penalty":
                target_expected_lwf,

            "green_success":
                target_success,

            "green_failure":
                target_failed,

            "counted_for_incident":
                target_counted,

            "incident_lwf_count_after":
                (
                    bookkeeping
                    .incident_host_lwf_count
                ),
        }
    )

    # =========================================================
    # Stage 3
    # Force one OTHER-HOST LWF.
    #
    # Critical audit:
    #
    # official Blue team reward changes,
    # BUT current incident-host metric MUST NOT change.
    # =========================================================

    set_active_services_reliability(
        controller,
        other_host,
        0,
    )

    other_expected_lwf = (
        official_lwf_penalty(
            controller,
            other_host,
        )
    )

    count_before_other = (
        bookkeeping
        .incident_host_lwf_count
    )

    penalty_before_other = (
        bookkeeping
        .incident_host_lwf_penalty
    )

    other_lwf_action = (
        make_green_local_work(
            controller,
            green_agent=(
                other_green_agent
            ),
            green_session=(
                other_green_session
            ),
            hostname=(
                other_host
            ),
            phishing_error_rate=0.0,
        )
    )

    (
        other_lwf_step,
        other_lwf_obs,
    ) = run_step(
        cyborg=cyborg,
        blue_agent=BLUE_AGENT,
        actions={
            other_green_agent:
                other_lwf_action
        },
        skip_valid_action_check=True,
    )

    (
        other_success,
        other_failed,
        other_success_repr,
    ) = observation_success(
        other_lwf_obs,
        other_green_agent,
    )

    if not other_failed:
        raise RuntimeError(
            "other-host GreenLocalWork "
            "was expected to fail: "
            f"{other_success_repr}"
        )

    other_observed_brm = float(
        other_lwf_step[
            "blue_reward"
        ][
            "blue_reward_machine"
        ]
    )

    if (
        other_observed_brm
        != other_expected_lwf
    ):
        raise RuntimeError(
            "other-host LWF reward mismatch: "
            f"observed="
            f"{other_observed_brm}, "
            f"expected="
            f"{other_expected_lwf}"
        )

    other_counted = (
        bookkeeping
        .record_local_work_failure(
            hostname=other_host,
            failed=other_failed,
            raw_lwf_penalty=(
                other_expected_lwf
            ),
        )
    )

    if other_counted:
        raise RuntimeError(
            "other-host LWF was "
            "incorrectly counted "
            "for current incident"
        )

    if (
        bookkeeping
        .incident_host_lwf_count
        != count_before_other
    ):
        raise RuntimeError(
            "other-host LWF changed "
            "incident-host count"
        )

    if (
        bookkeeping
        .incident_host_lwf_penalty
        != penalty_before_other
    ):
        raise RuntimeError(
            "other-host LWF changed "
            "incident-host penalty"
        )

    # Yet official team reward MUST see it.
    if other_observed_brm == 0.0:
        raise RuntimeError(
            "other-host LWF should "
            "still affect official "
            "Blue team reward"
        )

    official_team_return += (
        other_lwf_step[
            "blue_reward"
        ][
            "total"
        ]
    )

    timeline.append(
        {
            "stage":
                "other_host_lwf",

            **other_lwf_step,

            "hostname":
                other_host,

            "expected_raw_lwf_penalty":
                other_expected_lwf,

            "green_success":
                other_success,

            "green_failure":
                other_failed,

            "counted_for_incident":
                other_counted,

            "incident_lwf_count_after":
                (
                    bookkeeping
                    .incident_host_lwf_count
                ),
        }
    )

    # =========================================================
    # Stage 4
    # Restore current incident host.
    #
    # Real Restore duration = 5.
    # Red presence must remain during busy ticks
    # and disappear after Restore actually executes.
    # =========================================================

    restore_action = Restore(
        session=0,
        agent=BLUE_AGENT,
        hostname=target_host,
    )

    restore_duration = int(
        restore_action.duration
    )

    if restore_duration != 5:
        raise RuntimeError(
            "Restore duration changed: "
            f"{restore_duration}"
        )

    restore_ticks = []

    for restore_tick in range(
        1,
        restore_duration + 1,
    ):
        red_before = (
            ground_truth_red_presence(
                controller,
                target_host,
            )
        )

        if restore_tick == 1:
            step_actions = {
                BLUE_AGENT:
                    restore_action
            }
        else:
            # SleepAgent defaults are submitted
            # automatically, but while Restore is busy
            # they must not replace it.
            step_actions = {}

        (
            restore_step,
            _restore_obs,
        ) = run_step(
            cyborg=cyborg,
            blue_agent=BLUE_AGENT,
            actions=step_actions,
            skip_valid_action_check=False,
        )

        red_after = (
            ground_truth_red_presence(
                controller,
                target_host,
            )
        )

        bookkeeping.record_presence_transition(
            before=red_before,
            after=red_after,
            tick_after=(
                restore_step[
                    "tick_end"
                ]
            ),
        )

        progress_after = (
            progress_snapshot(
                controller,
                BLUE_AGENT,
            )
        )

        if (
            restore_tick
            < restore_duration
        ):
            if not red_after:
                raise RuntimeError(
                    "incident recovered "
                    "before Restore completed"
                )

            if progress_after is None:
                raise RuntimeError(
                    "Restore unexpectedly "
                    "not in progress"
                )

        else:
            if red_after:
                raise RuntimeError(
                    "red presence remains "
                    "after Restore completed"
                )

            if progress_after is not None:
                raise RuntimeError(
                    "Restore still marked "
                    "in progress after "
                    "duration=5"
                )

        official_team_return += (
            restore_step[
                "blue_reward"
            ][
                "total"
            ]
        )

        restore_record = {
            "stage":
                (
                    "restore_tick_"
                    f"{restore_tick}"
                ),

            **restore_step,

            "restore_tick":
                int(
                    restore_tick
                ),

            "incident_red_before":
                red_before,

            "incident_red_after":
                red_after,

            "progress_after":
                progress_after,
        }

        restore_ticks.append(
            restore_record
        )

        timeline.append(
            restore_record
        )

    # =========================================================
    # Final contract assertions
    # =========================================================

    if bookkeeping.t_compromise is None:
        raise RuntimeError(
            "t_compromise was not recorded"
        )

    if bookkeeping.t_normal is None:
        raise RuntimeError(
            "t_normal was not recorded"
        )

    eradication_time = (
        bookkeeping
        .attack_eradication_time
    )

    if (
        eradication_time is None
        or eradication_time <= 0
    ):
        raise RuntimeError(
            "invalid attack "
            "eradication time"
        )

    if (
        bookkeeping
        .incident_host_lwf_count
        != 1
    ):
        raise RuntimeError(
            "final incident-host "
            "LWF count must equal 1"
        )

    # Raw CC4 LWF is a negative reward.
    #
    # For later response-objective cost form,
    # it is useful to also expose the positive cost.
    incident_host_lwf_cost = float(
        -bookkeeping
        .incident_host_lwf_penalty
    )

    result = {
        "seed":
            int(
                args.seed
            ),

        "paper_metric_contract": {
            "attack_eradication_time":
                (
                    "t_normal - "
                    "t_compromise"
                ),

            "host_work_fail_paper_term":
                (
                    "Host Work Fail / "
                    "normal operation failure"
                ),

            "cc4_source_term":
                "LWF",

            "cc4_source_meaning":
                "Local Work Fails",

            "cc4_trigger_action":
                "GreenLocalWork",

            "incident_scope":
                (
                    "current incident "
                    "host only"
                ),
        },

        "truth_separation": {
            "ground_truth_scope":
                (
                    "reward_and_evaluation_"
                    "bookkeeping_only"
                ),

            "ground_truth_red_presence":
                (
                    "CC4 state host "
                    "red sessions"
                ),

            "hidden_truth_used_for_"
            "planner_action_selection":
                False,

            "hidden_truth_allowed_in_"
            "formal_planner_state":
                False,
        },

        "incident": {
            "incident_event_id":
                (
                    bookkeeping
                    .incident_event_id
                ),

            "incident_host_id":
                (
                    bookkeeping
                    .incident_host_id
                ),

            "other_host_id":
                other_host,

            "t_compromise":
                (
                    bookkeeping
                    .t_compromise
                ),

            "t_normal":
                (
                    bookkeeping
                    .t_normal
                ),

            "attack_eradication_time":
                eradication_time,

            "incident_host_lwf_count":
                (
                    bookkeeping
                    .incident_host_lwf_count
                ),

            "incident_host_lwf_"
            "raw_penalty":
                (
                    bookkeeping
                    .incident_host_lwf_penalty
                ),

            "incident_host_lwf_cost":
                incident_host_lwf_cost,
        },

        "isolation_checks": {
            "incident_host_lwf_"
            "counted":
                target_counted,

            "other_host_lwf_"
            "official_team_penalty":
                other_observed_brm,

            "other_host_lwf_"
            "counted_for_incident":
                other_counted,

            "incident_count_after_"
            "other_host_failure":
                (
                    bookkeeping
                    .incident_host_lwf_count
                ),
        },

        "official_cc4": {
            "team_episode_return_"
            "over_probe_ticks":
                float(
                    official_team_return
                ),

            "note":
                (
                    "external evaluation "
                    "quantity; not the "
                    "incident-specific "
                    "response objective"
                ),
        },

        "restore": {
            "duration":
                restore_duration,

            "ticks":
                restore_ticks,
        },

        "timeline":
            timeline,
    }

    # =========================================================
    # Console summary
    # =========================================================

    print()
    print(
        "=" * 80
    )

    print(
        "[A3.5 INCIDENT BOOKKEEPING PROBE]"
    )

    print(
        "incident_event_id:",
        bookkeeping.incident_event_id,
    )

    print(
        "incident_host:",
        target_host,
    )

    print(
        "other_host:",
        other_host,
    )

    print(
        "t_compromise:",
        bookkeeping.t_compromise,
    )

    print(
        "t_normal:",
        bookkeeping.t_normal,
    )

    print(
        "attack_eradication_time:",
        eradication_time,
    )

    print(
        "incident_host_lwf_count:",
        (
            bookkeeping
            .incident_host_lwf_count
        ),
    )

    print(
        "incident_host_lwf_raw_penalty:",
        (
            bookkeeping
            .incident_host_lwf_penalty
        ),
    )

    print(
        "other_host_lwf_team_penalty:",
        other_observed_brm,
    )

    print(
        "other_host_lwf_counted_for_incident:",
        other_counted,
    )

    print(
        "official_team_return_over_probe:",
        official_team_return,
    )

    print()
    print(
        "[GROUND TRUTH SCOPE]"
    )

    print(
        "bookkeeping_only = True"
    )

    print(
        "planner_hidden_truth_input = False"
    )

    out_path = Path(
        args.out
    )

    if not out_path.is_absolute():
        out_path = (
            PROJ
            /
            out_path
        ).resolve()

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        out_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(
        "[OK] incident bookkeeping "
        "probe saved:"
    )

    print(
        out_path
    )


if __name__ == "__main__":
    main()