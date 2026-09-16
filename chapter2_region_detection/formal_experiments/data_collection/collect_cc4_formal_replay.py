from __future__ import annotations

import argparse
import json
import os
import sys

from collections import Counter
from pathlib import Path
from typing import Any, Sequence


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
    FiniteStateRedAgent,
    EnterpriseGreenAgent,
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

from CybORG.Agents.Wrappers.BlueFixedActionWrapper import (  # type: ignore
    BlueFixedActionWrapper,
)

from formal_experiments.data_collection.decision_replay import (
    DecisionEpochReplayCollector,
    DecisionEpochTransition,
)

from formal_experiments.data_collection.incident_response import (
    IncidentResponseBookkeeper,
    LocalWorkFailure,
)

from shared.action_contract import (
    N_ACTIONS,
    get_action,
)

from shared.cyborg_action_adapter import (
    CybORGActionAdapter,
)

from shared.cyborg_target_resolver import (
    observable_target_family_availability,
)

from shared.formal_state import (
    BLUE_AGENTS,
    FORMAL_STATE_DIM,
    FormalStateEncoder,
    ObservableHostEvidenceTracker,
)


# ============================================================
# Frozen A4.5 seed protocol
# ============================================================

DEFAULT_SPLIT_SEEDS = {
    "train":
        tuple(
            range(
                1000,
                1032,
            )
        ),

    "validation":
        tuple(
            range(
                2000,
                2008,
            )
        ),

    "calibration":
        tuple(
            range(
                3000,
                3008,
            )
        ),

    "test":
        tuple(
            range(
                4000,
                4020,
            )
        ),
}


EXECUTED_FAMILY_TO_ACTION_ID = {
    "Sleep": 0,
    "Analyse": 1,
    "Remove": 2,
    "Restore": 3,
}


# ============================================================
# Seed / exploration helpers
# ============================================================

def parse_seed_csv(
    value: str,
) -> list[int]:
    result = []

    for raw in str(
        value
    ).split(","):
        raw = raw.strip()

        if raw:
            result.append(
                int(raw)
            )

    if not result:
        raise ValueError(
            "at least one seed required"
        )

    if len(result) != len(
        set(result)
    ):
        raise ValueError(
            "duplicate seeds are not allowed"
        )

    return result


def seeds_for_split(
    split: str,
    explicit: str | None,
) -> list[int]:
    split = str(
        split
    )

    if split not in (
        "train",
        "validation",
    ):
        raise ValueError(
            "A4.5a only permits "
            "train or validation"
        )

    if explicit is not None:
        return parse_seed_csv(
            explicit
        )

    return list(
        DEFAULT_SPLIT_SEEDS[
            split
        ]
    )


def stratified_requested_action_id(
    *,
    episode_seed: int,
    agent_name: str,
    decision_index: int,
) -> int:

    if agent_name not in BLUE_AGENTS:
        raise ValueError(
            f"unsupported agent: {agent_name}"
        )

    if (
        isinstance(
            decision_index,
            bool,
        )
        or not isinstance(
            decision_index,
            int,
        )
        or decision_index < 0
    ):
        raise ValueError(
            "decision_index must be "
            "a non-negative int"
        )

    agent_offset = (
        BLUE_AGENTS.index(
            agent_name
        )
    )

    return int(
        (
            int(episode_seed)
            + agent_offset
            + decision_index
        )
        % N_ACTIONS
    )


# ============================================================
# CC4 environment
# ============================================================

def make_env(
    *,
    seed: int,
    steps: int,
    pad_spaces: bool,
):
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,

        green_agent_class=(
            EnterpriseGreenAgent
        ),

        red_agent_class=(
            FiniteStateRedAgent
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

    env = BlueFixedActionWrapper(
        cyborg,

        pad_spaces=bool(
            pad_spaces
        ),
    )

    (
        observations,
        info,
    ) = env.reset(
        seed=int(
            seed
        )
    )

    return (
        env,
        observations,
        info,
    )


# ============================================================
# Hidden truth — reward/evaluation ONLY
# ============================================================

def ground_truth_red_presence(
    controller,
    hostname: str,
) -> bool:

    hostname = str(
        hostname
    )

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
            "red_agent"
            not in str(
                agent_name
            )
        ):
            continue

        sessions = (
            controller
            .state
            .sessions
            .get(
                agent_name,
                {},
            )
        )

        for session_id in session_ids:
            session = (
                sessions.get(
                    session_id
                )
            )

            if (
                session is not None
                and bool(
                    session.active
                )
            ):
                return True

    return False


def red_presence_for_hosts(
    controller,
    hostnames: Sequence[str],
) -> dict[str, bool]:

    return {
        str(hostname):
            ground_truth_red_presence(
                controller,
                str(
                    hostname
                ),
            )

        for hostname
        in hostnames
    }


def official_lwf_penalty(
    controller,
    hostname: str,
) -> float:

    subnet = (
        controller
        .state
        .hostname_subnet_map[
            hostname
        ]
    )

    subnet_name = str(
        getattr(
            subnet,
            "value",
            subnet,
        )
    )

    mission_phase = int(
        controller
        .state
        .mission_phase
    )

    reward_machine = (
        BlueRewardMachine(
            "Blue"
        )
    )

    phase_rewards = (
        reward_machine
        .get_phase_rewards(
            mission_phase
        )
    )

    return float(
        phase_rewards[
            subnet_name
        ][
            "LWF"
        ]
    )


def green_local_work_failures(
    controller,
) -> list[
    LocalWorkFailure
]:
    result = []

    for (
        agent_name,
        executed_actions,
    ) in (
        controller.action.items()
    ):

        if (
            "green_agent"
            not in str(
                agent_name
            )
        ):
            continue

        if not executed_actions:
            continue

        observation_set = (
            controller
            .observation
            .get(
                agent_name
            )
        )

        if (
            observation_set
            is None
            or not observation_set
            .observations
        ):
            continue

        success = (
            observation_set
            .observations[
                0
            ]
            .data
            .get(
                "success"
            )
        )

        sessions = (
            controller
            .state
            .sessions
            .get(
                agent_name,
                {},
            )
            .values()
        )

        has_active_session = any(
            bool(
                session.active
            )
            for session
            in sessions
        )

        if not has_active_session:
            continue

        for action in executed_actions:

            if not isinstance(
                action,
                GreenLocalWork,
            ):
                continue

            if not (
                success == False
            ):
                continue

            hostname = str(
                controller
                .state
                .ip_addresses[
                    action.ip_address
                ]
            )

            result.append(
                LocalWorkFailure(
                    hostname=hostname,

                    raw_lwf_penalty=(
                        official_lwf_penalty(
                            controller,
                            hostname,
                        )
                    ),
                )
            )

    return result


# ============================================================
# Observation success helper
# ============================================================

def observation_success_bool(
    observation: dict,
) -> bool | None:

    raw = observation.get(
        "success"
    )

    if raw == True:
        return True

    if raw == False:
        return False

    name = getattr(
        raw,
        "name",
        None,
    )

    if name is not None:
        text = str(
            name
        ).upper()

        if text == "TRUE":
            return True

        if text == "FALSE":
            return False

    return None


# ============================================================
# A4.1c planner-visible availability
# ============================================================

def planner_visible_target_availability(
    *,
    env,
    agent_name: str,
    observable_host_scores,
) -> tuple[
    bool,
    dict[str, bool],
]:
    """
    只使用：

        wrapper action_labels
        wrapper action_mask
        observable host scores

    当前 frozen CC4 中 Analyse / Remove / Restore
    应共享相同 host availability。

    如果三者出现差异，直接失败，
    不允许强行压缩成一个 state bit。
    """

    availability = (
        observable_target_family_availability(
            labels=(
                env.action_labels(
                    agent_name
                )
            ),

            mask=(
                env.action_mask(
                    agent_name
                )
            ),

            observable_host_scores=(
                observable_host_scores
            ),
        )
    )

    values = set(
        availability.values()
    )

    if len(values) != 1:
        raise RuntimeError(
            f"{agent_name}: targeted "
            "family availability "
            "diverged: "
            f"{availability}"
        )

    return (
        bool(
            next(
                iter(
                    values
                )
            )
        ),

        availability,
    )


def encode_decision_state(
    *,
    env,
    encoder: FormalStateEncoder,
    tracker: ObservableHostEvidenceTracker,
    observation,
    agent_name: str,
    global_tick: int,
    episode_steps: int,
):
    """
    正式 decision-epoch state helper。

    evidence + action availability
    全部来自 planner-visible 信息。
    """

    host_scores = (
        tracker
        .observable_host_scores(
            global_tick
        )
    )

    (
        any_valid_target,
        family_availability,
    ) = (
        planner_visible_target_availability(
            env=env,

            agent_name=(
                agent_name
            ),

            observable_host_scores=(
                host_scores
            ),
        )
    )

    state = encoder.encode(
        tracker=tracker,

        observation=observation,

        global_tick=(
            global_tick
        ),

        episode_steps=(
            episode_steps
        ),

        any_valid_observable_target=(
            any_valid_target
        ),
    )

    return (
        state,
        host_scores,
        family_availability,
    )


# ============================================================
# One episode
# ============================================================

def collect_episode(
    *,
    seed: int,
    steps: int,
    pad_spaces: bool,
) -> tuple[
    tuple[
        DecisionEpochTransition,
        ...,
    ],
    dict[str, Any],
]:

    (
        env,
        reset_observations,
        _reset_info,
    ) = make_env(
        seed=seed,
        steps=steps,
        pad_spaces=pad_spaces,
    )

    controller = (
        env.env
        .environment_controller
    )

    adapter = (
        CybORGActionAdapter()
    )

    encoder = (
        FormalStateEncoder()
    )

    replay = (
        DecisionEpochReplayCollector(
            episode_seed=int(
                seed
            )
        )
    )

    trackers = {}
    bookkeepers = {}

    current_observation = {}

    decision_index = {
        agent: 0
        for agent
        in BLUE_AGENTS
    }

    active = {
        agent: None
        for agent
        in BLUE_AGENTS
    }

    done_agents: set[str] = set()

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    for agent in BLUE_AGENTS:

        if (
            agent
            not in reset_observations
        ):
            raise RuntimeError(
                f"{agent}: missing "
                "reset observation"
            )

        tracker = (
            ObservableHostEvidenceTracker(
                agent
            )
        )

        tracker.reset(
            reset_observations[
                agent
            ]
        )

        trackers[
            agent
        ] = tracker

        current_observation[
            agent
        ] = (
            reset_observations[
                agent
            ]
        )

        bookkeeper = (
            IncidentResponseBookkeeper(
                episode_seed=int(
                    seed
                ),

                agent_name=agent,
            )
        )

        bookkeeper.reset(
            initial_red_presence_by_host=(
                red_presence_for_hosts(
                    controller,
                    tracker.inventory,
                )
            ),

            global_tick=int(
                controller.step_count
            ),
        )

        bookkeepers[
            agent
        ] = bookkeeper

    # --------------------------------------------------------
    # GLOBAL TICK LOOP
    # --------------------------------------------------------

    while (
        int(
            controller.step_count
        )
        < int(
            steps
        )

        and len(
            done_agents
        )
        < len(
            BLUE_AGENTS
        )
    ):

        tick_start = int(
            controller.step_count
        )

        joint_actions = {}

        # ====================================================
        # Decision epochs
        # ====================================================

        for agent in BLUE_AGENTS:

            if agent in done_agents:
                continue

            if (
                active[
                    agent
                ]
                is not None
            ):
                continue

            tracker = (
                trackers[
                    agent
                ]
            )

            (
                state,
                host_scores,
                family_availability,
            ) = encode_decision_state(
                env=env,

                encoder=encoder,

                tracker=tracker,

                observation=(
                    current_observation[
                        agent
                    ]
                ),

                agent_name=agent,

                global_tick=(
                    tick_start
                ),

                episode_steps=int(
                    steps
                ),
            )

            action_id = (
                stratified_requested_action_id(
                    episode_seed=int(
                        seed
                    ),

                    agent_name=agent,

                    decision_index=(
                        decision_index[
                            agent
                        ]
                    ),
                )
            )

            resolution = (
                adapter.resolve(
                    env=env,

                    agent_name=agent,

                    action_id=(
                        action_id
                    ),

                    observable_host_scores=(
                        host_scores
                    ),
                )
            )

            # =================================================
            # A4.1c exact consistency assertion
            # =================================================

            requested_contract = (
                get_action(
                    action_id
                )
            )

            requested_family = str(
                requested_contract
                .cyborg_action
            )

            if requested_family == "Sleep":

                if resolution.fallback:
                    raise RuntimeError(
                        f"{agent}: requested "
                        "Sleep must never "
                        "fallback"
                    )

                if (
                    resolution
                    .executed_action_family
                    != "Sleep"
                ):
                    raise RuntimeError(
                        f"{agent}: requested "
                        "Sleep must execute "
                        "Sleep"
                    )

            else:

                if (
                    requested_family
                    not in family_availability
                ):
                    raise RuntimeError(
                        f"{agent}: missing "
                        "availability for "
                        f"{requested_family}"
                    )

                expected_available = bool(
                    family_availability[
                        requested_family
                    ]
                )

                expected_fallback = (
                    not expected_available
                )

                if (
                    bool(
                        resolution.fallback
                    )
                    != expected_fallback
                ):
                    raise RuntimeError(
                        f"{agent}: planner-visible "
                        "availability disagrees "
                        "with production adapter: "
                        f"requested_family="
                        f"{requested_family}, "
                        f"availability="
                        f"{family_availability}, "
                        f"fallback="
                        f"{resolution.fallback}"
                    )

                if expected_fallback:

                    if (
                        resolution
                        .executed_action_family
                        != "Sleep"
                    ):
                        raise RuntimeError(
                            f"{agent}: fallback "
                            "must execute Sleep"
                        )

                else:

                    if (
                        resolution
                        .executed_action_family
                        != requested_family
                    ):
                        raise RuntimeError(
                            f"{agent}: valid "
                            "targeted action "
                            "changed family: "
                            f"requested="
                            f"{requested_family}, "
                            f"executed="
                            f"{resolution.executed_action_family}"
                        )

            action_objects = list(
                env.actions(
                    agent
                )
            )

            executed_index = int(
                resolution
                .executed_index
            )

            if not (
                0
                <= executed_index
                < len(
                    action_objects
                )
            ):
                raise RuntimeError(
                    f"{agent}: invalid "
                    "executed index"
                )

            action_object = (
                action_objects[
                    executed_index
                ]
            )

            actual_family = (
                type(
                    action_object
                ).__name__
            )

            if (
                actual_family
                != resolution
                .executed_action_family
            ):
                raise RuntimeError(
                    f"{agent}: adapter/"
                    "action-object family "
                    "mismatch"
                )

            executed_duration = int(
                action_object.duration
            )

            if executed_duration <= 0:
                raise RuntimeError(
                    f"{agent}: invalid "
                    "executed duration"
                )

            replay.begin_decision(
                agent_name=agent,

                decision_index=(
                    decision_index[
                        agent
                    ]
                ),

                global_tick_start=(
                    tick_start
                ),

                state=state,

                resolution=resolution,
            )

            active[
                agent
            ] = {
                "ready_at":
                    int(
                        tick_start
                        + executed_duration
                    ),

                "executed_family":
                    str(
                        resolution
                        .executed_action_family
                    ),

                "target_host":
                    resolution
                    .target_host,

                "executed_duration":
                    executed_duration,
            }

            joint_actions[
                agent
            ] = executed_index

        if not replay.open_agents:
            raise RuntimeError(
                "no open Blue decisions "
                "before environment step"
            )

        # ====================================================
        # One real global CC4 tick
        # ====================================================

        (
            observations,
            rewards,
            terminated,
            truncated,
            _info,
        ) = env.step(
            actions=joint_actions
        )

        tick_end = int(
            controller.step_count
        )

        if (
            tick_end
            != tick_start + 1
        ):
            raise RuntimeError(
                "CC4 step did not advance "
                "exactly one global tick"
            )

        all_lwf = (
            green_local_work_failures(
                controller
            )
        )

        # ====================================================
        # Per-agent accounting / next state
        # ====================================================

        for agent in BLUE_AGENTS:

            if agent in done_agents:
                continue

            if agent not in observations:
                raise RuntimeError(
                    f"{agent}: missing "
                    "post-step observation"
                )

            observation = (
                observations[
                    agent
                ]
            )

            tracker = (
                trackers[
                    agent
                ]
            )

            bookkeeper = (
                bookkeepers[
                    agent
                ]
            )

            inventory_set = set(
                tracker.inventory
            )

            scoped_lwf = [
                failure
                for failure
                in all_lwf
                if (
                    failure.hostname
                    in inventory_set
                )
            ]

            accounting = (
                bookkeeper.record_tick(
                    global_tick_end=(
                        tick_end
                    ),

                    red_presence_after=(
                        red_presence_for_hosts(
                            controller,
                            tracker.inventory,
                        )
                    ),

                    local_work_failures=(
                        scoped_lwf
                    ),
                )
            )

            if (
                agent
                not in replay.open_agents
            ):
                raise RuntimeError(
                    f"{agent}: no replay "
                    "decision open"
                )

            replay.record_tick(
                agent_name=agent,

                global_tick_end=(
                    tick_end
                ),

                official_reward=float(
                    rewards.get(
                        agent,
                        0.0,
                    )
                ),

                **accounting
                .to_replay_kwargs(),
            )

            meta = (
                active[
                    agent
                ]
            )

            if meta is None:
                raise RuntimeError(
                    f"{agent}: replay open "
                    "but scheduler inactive"
                )

            ready_at = int(
                meta[
                    "ready_at"
                ]
            )

            if (
                tick_end
                > ready_at
            ):
                raise RuntimeError(
                    f"{agent}: scheduler "
                    "missed decision epoch"
                )

            completed = (
                tick_end
                == ready_at
            )

            done = bool(
                terminated.get(
                    agent,
                    False,
                )
                or
                truncated.get(
                    agent,
                    False,
                )
            )

            completion_success = (
                observation_success_bool(
                    observation
                )
                if completed
                else None
            )

            # ------------------------------------------------
            # Observable tracker update
            # ------------------------------------------------

            tracker.update(
                observation=observation,

                global_tick=(
                    tick_end
                ),

                completed_action_family=(
                    str(
                        meta[
                            "executed_family"
                        ]
                    )
                    if completed
                    else None
                ),

                completed_target_host=(
                    meta[
                        "target_host"
                    ]
                    if completed
                    else None
                ),

                completed_action_success=(
                    completion_success
                    if completed
                    else None
                ),
            )

            current_observation[
                agent
            ] = observation

            # ------------------------------------------------
            # Decision closes
            # ------------------------------------------------

            if (
                completed
                or done
            ):
                (
                    next_state,
                    _next_host_scores,
                    _next_family_availability,
                ) = encode_decision_state(
                    env=env,

                    encoder=encoder,

                    tracker=tracker,

                    observation=(
                        observation
                    ),

                    agent_name=agent,

                    global_tick=(
                        tick_end
                    ),

                    episode_steps=int(
                        steps
                    ),
                )

                replay.end_decision(
                    agent_name=agent,

                    global_tick_end=(
                        tick_end
                    ),

                    next_state=(
                        next_state
                    ),

                    done=done,

                    completed_action_success=(
                        completion_success
                        if completed
                        else None
                    ),
                )

                active[
                    agent
                ] = None

                decision_index[
                    agent
                ] += 1

            if done:
                done_agents.add(
                    agent
                )

    # --------------------------------------------------------
    # End-of-episode consistency
    # --------------------------------------------------------

    if replay.open_agents:
        raise RuntimeError(
            "episode ended with "
            "unclosed decisions: "
            f"{replay.open_agents}"
        )

    transitions = (
        replay.buffer.transitions
    )

    if not transitions:
        raise RuntimeError(
            "episode produced "
            "zero replay transitions"
        )

    episode_summary = {
        "seed":
            int(
                seed
            ),

        "global_ticks":
            int(
                controller.step_count
            ),

        "transition_count":
            len(
                transitions
            ),

        "per_agent_transitions": {
            agent:
                sum(
                    1
                    for item
                    in transitions
                    if (
                        item.agent_name
                        == agent
                    )
                )

            for agent
            in BLUE_AGENTS
        },

        "incident_event_count": {
            agent:
                len(
                    bookkeepers[
                        agent
                    ]
                    .all_events
                )

            for agent
            in BLUE_AGENTS
        },

        "unresolved_incident_count": {
            agent:
                int(
                    bookkeepers[
                        agent
                    ]
                    .unresolved_event_count
                )

            for agent
            in BLUE_AGENTS
        },
    }

    return (
        transitions,
        episode_summary,
    )


# ============================================================
# Summary
# ============================================================

def build_collection_summary(
    *,
    split: str,
    seeds: Sequence[int],
    steps: int,
    transitions: Sequence[
        DecisionEpochTransition
    ],
    episode_summaries: Sequence[
        dict[str, Any]
    ],
) -> dict[str, Any]:

    requested = Counter(
        int(
            item
            .requested_action_id
        )
        for item
        in transitions
    )

    executed = Counter(
        str(
            item
            .executed_action_family
        )
        for item
        in transitions
    )

    per_agent = Counter(
        str(
            item.agent_name
        )
        for item
        in transitions
    )

    targeted = [
        item
        for item
        in transitions
        if (
            int(
                item
                .requested_action_id
            )
            != 0
        )
    ]

    valid_targeted = [
        item
        for item
        in targeted
        if not bool(
            item.fallback
        )
    ]

    fallback_count = sum(
        1
        for item
        in transitions
        if bool(
            item.fallback
        )
    )

    incomplete_count = sum(
        1
        for item
        in transitions
        if not bool(
            item.action_completed
        )
    )

    incident_hosts = sorted(
        {
            host
            for item
            in transitions
            for host
            in item.incident_host_ids
        }
    )

    requested_counts = {
        str(
            action_id
        ):
            int(
                requested.get(
                    action_id,
                    0,
                )
            )

        for action_id
        in range(
            N_ACTIONS
        )
    }

    missing_requested = [
        action_id
        for action_id
        in range(
            N_ACTIONS
        )
        if (
            requested_counts[
                str(
                    action_id
                )
            ]
            <= 0
        )
    ]

    if missing_requested:
        raise RuntimeError(
            "collection missing "
            "requested action IDs: "
            f"{missing_requested}"
        )

    missing_agents = [
        agent
        for agent
        in BLUE_AGENTS
        if (
            per_agent.get(
                agent,
                0,
            )
            <= 0
        )
    ]

    if missing_agents:
        raise RuntimeError(
            "collection missing "
            f"agents: {missing_agents}"
        )

    targeted_count = len(
        targeted
    )

    return {
        "split":
            str(
                split
            ),

        "seeds":
            [
                int(
                    seed
                )
                for seed
                in seeds
            ],

        "episode_count":
            len(
                seeds
            ),

        "episode_steps":
            int(
                steps
            ),

        "state_dim":
            int(
                FORMAL_STATE_DIM
            ),

        "n_actions":
            int(
                N_ACTIONS
            ),

        "exploration_policy":
            (
                "deterministic stratified "
                "round-robin requested actions; "
                "observable target scores only"
            ),

        "transition_count":
            len(
                transitions
            ),

        "completed_transition_count":
            int(
                len(
                    transitions
                )
                -
                incomplete_count
            ),

        "incomplete_terminal_count":
            int(
                incomplete_count
            ),

        "requested_action_counts":
            requested_counts,

        "executed_family_counts": {
            family:
                int(
                    executed.get(
                        family,
                        0,
                    )
                )

            for family
            in (
                "Sleep",
                "Analyse",
                "Remove",
                "Restore",
            )
        },

        "targeted_requested_count":
            int(
                targeted_count
            ),

        "valid_targeted_count":
            len(
                valid_targeted
            ),

        "valid_target_rate":
            (
                0.0
                if targeted_count == 0
                else float(
                    len(
                        valid_targeted
                    )
                    /
                    targeted_count
                )
            ),

        "fallback_count":
            int(
                fallback_count
            ),

        "fallback_rate":
            float(
                fallback_count
                /
                len(
                    transitions
                )
            ),

        "per_agent_transition_counts": {
            agent:
                int(
                    per_agent.get(
                        agent,
                        0,
                    )
                )

            for agent
            in BLUE_AGENTS
        },

        "incident_host_count":
            len(
                incident_hosts
            ),

        "incident_hosts":
            incident_hosts,

        "episodes":
            list(
                episode_summaries
            ),
    }


# ============================================================
# IO
# ============================================================

def save_jsonl(
    path: Path,
    transitions: Sequence[
        DecisionEpochTransition
    ],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        for item in transitions:
            handle.write(
                json.dumps(
                    item.to_jsonable(),
                    ensure_ascii=False,
                )
            )

            handle.write(
                "\n"
            )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "A4.5a formal CC4 "
            "decision-epoch replay collection."
        )
    )

    parser.add_argument(
        "--split",

        choices=(
            "train",
            "validation",
        ),

        required=True,
    )

    parser.add_argument(
        "--seeds",

        default=None,

        help=(
            "Optional comma-separated "
            "override. A4.5 smoke only."
        ),
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--pad-spaces",
        action="store_true",
    )

    parser.add_argument(
        "--out-dir",

        default=(
            "outputs/formal_replay"
        ),
    )

    args = (
        parser.parse_args()
    )

    if args.steps <= 0:
        raise ValueError(
            "--steps must be > 0"
        )

    seeds = seeds_for_split(
        args.split,
        args.seeds,
    )

    all_transitions = []
    episode_summaries = []

    for position, seed in enumerate(
        seeds,
        start=1,
    ):
        print()

        print(
            "=" * 80
        )

        print(
            "[A4.5a COLLECT]",
            f"split={args.split}",
            f"seed={seed}",
            f"episode="
            f"{position}/{len(seeds)}",
        )

        (
            transitions,
            episode_summary,
        ) = collect_episode(
            seed=int(
                seed
            ),

            steps=int(
                args.steps
            ),

            pad_spaces=bool(
                args.pad_spaces
            ),
        )

        all_transitions.extend(
            transitions
        )

        episode_summaries.append(
            episode_summary
        )

        print(
            "  transitions=",
            len(
                transitions
            ),
        )

        print(
            "  per_agent=",
            episode_summary[
                "per_agent_transitions"
            ],
        )

        print(
            "  incident_events=",
            episode_summary[
                "incident_event_count"
            ],
        )

    summary = (
        build_collection_summary(
            split=args.split,

            seeds=seeds,

            steps=int(
                args.steps
            ),

            transitions=(
                all_transitions
            ),

            episode_summaries=(
                episode_summaries
            ),
        )
    )

    out_dir = Path(
        args.out_dir
    )

    if not out_dir.is_absolute():
        out_dir = (
            PROJ
            /
            out_dir
        ).resolve()

    replay_path = (
        out_dir
        /
        f"{args.split}.jsonl"
    )

    summary_path = (
        out_dir
        /
        f"{args.split}_summary.json"
    )

    save_jsonl(
        replay_path,
        all_transitions,
    )

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(
        "=" * 80
    )

    print(
        "[A4.5a SUMMARY]"
    )

    print(
        "split:",
        summary[
            "split"
        ],
    )

    print(
        "episodes:",
        summary[
            "episode_count"
        ],
    )

    print(
        "transitions:",
        summary[
            "transition_count"
        ],
    )

    print(
        "requested:",
        summary[
            "requested_action_counts"
        ],
    )

    print(
        "executed:",
        summary[
            "executed_family_counts"
        ],
    )

    print(
        "valid_target_rate:",
        summary[
            "valid_target_rate"
        ],
    )

    print(
        "fallback_rate:",
        summary[
            "fallback_rate"
        ],
    )

    print(
        "incident_host_count:",
        summary[
            "incident_host_count"
        ],
    )

    print()
    print(
        "[OK] replay:"
    )

    print(
        replay_path
    )

    print(
        "[OK] summary:"
    )

    print(
        summary_path
    )


if __name__ == "__main__":
    main()