from __future__ import annotations

import argparse
import json
import os
import sys

from pathlib import Path
from typing import Any


# ============================================================
# Project / CybORG path
# ============================================================

THIS = Path(__file__).resolve()
PROJ = THIS.parents[2]

if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))


CYBORG_ROOT = os.environ.get(
    "CYBORG_ROOT",
    r"D:\python1\cage-challenge-4",
)

if CYBORG_ROOT and CYBORG_ROOT not in sys.path:
    sys.path.insert(0, CYBORG_ROOT)


from CybORG import CybORG  # type: ignore

from CybORG.Agents import (  # type: ignore
    SleepAgent,
    FiniteStateRedAgent,
    EnterpriseGreenAgent,
)

from CybORG.Simulator.Scenarios import (  # type: ignore
    EnterpriseScenarioGenerator,
)

from CybORG.Agents.Wrappers.BlueFixedActionWrapper import (  # type: ignore
    BlueFixedActionWrapper,
)

from shared.cyborg_action_adapter import (
    CybORGActionAdapter,
)


# ============================================================
# Frozen A3 contracts
# ============================================================

EXPECTED_AGENTS = [
    "blue_agent_0",
    "blue_agent_1",
    "blue_agent_2",
    "blue_agent_3",
    "blue_agent_4",
]


# A3.1 已真实确认。
#
# 这里只作为 integration probe oracle，
# 绝不能进入 production adapter。
EXPECTED_UNPADDED_SPACE = {
    "blue_agent_0": 82,
    "blue_agent_1": 82,
    "blue_agent_2": 82,
    "blue_agent_3": 82,
    "blue_agent_4": 242,
}


EXPECTED_PADDED_SPACE = {
    agent: 242
    for agent in EXPECTED_AGENTS
}


EXPECTED_SUBNET_COUNT = {
    "blue_agent_0": 1,
    "blue_agent_1": 1,
    "blue_agent_2": 1,
    "blue_agent_3": 1,
    "blue_agent_4": 3,
}


ACTION_CASES = {
    0: {
        "name": "no_op",
        "family": "Sleep",
        "duration": 1,
    },

    1: {
        "name": "analyse",
        "family": "Analyse",
        "duration": 2,
    },

    2: {
        "name": "remove",
        "family": "Remove",
        "duration": 3,
    },

    3: {
        "name": "restore",
        "family": "Restore",
        "duration": 5,
    },
}


FAMILY_DURATION = {
    str(case["family"]):
        int(case["duration"])
    for case in ACTION_CASES.values()
}


# ============================================================
# A3.6 async mixed-duration contract
# ============================================================
#
# 每个 Blue agent 连续执行两个动作。
#
# 第一动作 duration 不同，因此会自然产生
# asynchronous decision epochs。
#
# 第二动作进一步验证：
# ready agent 可以重新提交，
# 其他 busy agent 必须完全省略。
# ============================================================

ASYNC_ACTION_QUEUES = {
    "blue_agent_0": [
        0,  # Sleep   d=1
        3,  # Restore d=5
    ],

    "blue_agent_1": [
        1,  # Analyse d=2
        2,  # Remove  d=3
    ],

    "blue_agent_2": [
        2,  # Remove  d=3
        1,  # Analyse d=2
    ],

    "blue_agent_3": [
        3,  # Restore d=5
        0,  # Sleep   d=1
    ],

    "blue_agent_4": [
        1,  # Analyse d=2
        3,  # Restore d=5
    ],
}


# Scheduler-relative launch time。
#
# t=0:
#   五个 agent 首次 launch
#
# t=1:
#   blue_0 Sleep 完成
#   -> launch Restore
#
# t=2:
#   blue_1 / blue_4 Analyse 完成
#   -> launch Remove / Restore
#
# t=3:
#   blue_2 Remove 完成
#   -> launch Analyse
#
# t=5:
#   blue_3 Restore 完成
#   -> launch Sleep
#
ASYNC_EXPECTED_LAUNCHES = {
    0: [
        "blue_agent_0",
        "blue_agent_1",
        "blue_agent_2",
        "blue_agent_3",
        "blue_agent_4",
    ],

    1: [
        "blue_agent_0",
    ],

    2: [
        "blue_agent_1",
        "blue_agent_4",
    ],

    3: [
        "blue_agent_2",
    ],

    5: [
        "blue_agent_3",
    ],
}


# 若所有 targeted actions 都正常解析，
# 最后 blue_agent_4 Restore 在 t=7 完成。
ASYNC_EXPECTED_FINAL_TIME = 7


# ============================================================
# Environment
# ============================================================

def make_env(
    seed: int,
    steps: int,
    pad_spaces: bool,
):
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=int(steps),
    )

    cyborg = CybORG(
        scenario_generator=sg,
        seed=int(seed),
    )

    env = BlueFixedActionWrapper(
        cyborg,
        pad_spaces=bool(pad_spaces),
    )

    env.reset(
        seed=int(seed)
    )

    return env


# ============================================================
# Agent discovery
# ============================================================

def blue_agents(
    env,
) -> list[str]:
    agents = sorted(
        str(agent)
        for agent
        in env.action_spaces().keys()
        if "blue_agent_" in str(agent)
    )

    if agents != EXPECTED_AGENTS:
        raise RuntimeError(
            "unexpected Blue agents: "
            f"{agents}"
        )

    return agents


# ============================================================
# Independent probe oracle
# ============================================================

def independent_valid_candidates(
    env,
    agent: str,
    family: str,
) -> list[dict[str, Any]]:
    """
    独立于 production resolver 的 probe oracle。

    规则：

        1. 先过滤 action_mask=True；
        2. 再严格解析 action label；
        3. 不使用 production resolver
           生成自己的 expected result。

    这样避免循环验证。
    """

    labels = list(
        env.action_labels(
            agent
        )
    )

    mask = list(
        env.action_mask(
            agent
        )
    )

    if len(labels) != len(mask):
        raise RuntimeError(
            f"{agent}: "
            "labels/mask mismatch"
        )

    result = []

    for index, (
        label,
        valid,
    ) in enumerate(
        zip(
            labels,
            mask,
        )
    ):
        if not bool(valid):
            continue

        text = str(
            label
        ).strip()

        if family == "Sleep":

            if text != "Sleep":
                continue

            target = None

        else:

            prefix = (
                family
                + " "
            )

            if not text.startswith(
                prefix
            ):
                continue

            target = text[
                len(prefix):
            ].strip()

            if not target:
                continue

        result.append(
            {
                "index":
                    int(index),

                "label":
                    text,

                "target_host":
                    target,
            }
        )

    return result


# ============================================================
# Probe-only observable scores
# ============================================================

def synthetic_observable_scores(
    candidates: list[
        dict[str, Any]
    ],
) -> tuple[
    dict[str, float],
    str,
]:
    """
    A3.6 probe-only synthetic evidence。

    绝不能进入 A4 FormalState 或正式 planner。

    将候选 host 按 hostname 排序，
    然后赋严格递增 score：

        host0 -> 1
        host1 -> 2
        ...

    因此 expected target 唯一、
    deterministic。
    """

    hosts = sorted(
        {
            str(
                item[
                    "target_host"
                ]
            )
            for item in candidates
            if (
                item[
                    "target_host"
                ]
                is not None
            )
        }
    )

    if not hosts:
        raise RuntimeError(
            "no host candidates"
        )

    scores = {
        host:
            float(
                index + 1
            )
        for (
            index,
            host,
        )
        in enumerate(hosts)
    }

    expected_target = (
        hosts[-1]
    )

    return (
        scores,
        expected_target,
    )


# ============================================================
# Controller oracle helpers
# ============================================================
#
# IMPORTANT:
#
# 以下 controller internals
# 只能用于 probe ASSERTION。
#
# 绝不能用于 async scheduler
# 判断哪个 agent ready。
# ============================================================

def controller_progress(
    controller,
    agent: str,
):
    item = (
        controller
        .actions_in_progress
        .get(agent)
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
            str(action),

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


def controller_executed(
    controller,
    agent: str,
) -> list[
    dict[str, Any]
]:
    actions = (
        controller
        .action
        .get(
            agent,
            [],
        )
    )

    return [
        {
            "action_class":
                type(
                    action
                ).__name__,

            "action_repr":
                str(action),

            "duration":
                int(
                    action.duration
                ),
        }
        for action
        in actions
    ]


# ============================================================
# Layout validation
# ============================================================

def validate_layout(
    env,
    pad_spaces: bool,
):
    agents = blue_agents(
        env
    )

    expected_spaces = (
        EXPECTED_PADDED_SPACE
        if pad_spaces
        else EXPECTED_UNPADDED_SPACE
    )

    result = {}

    for agent in agents:

        labels = list(
            env.action_labels(
                agent
            )
        )

        mask = list(
            env.action_mask(
                agent
            )
        )

        hosts = [
            str(value)
            for value
            in env.hosts(
                agent
            )
        ]

        subnets = [
            str(value)
            for value
            in env.subnets(
                agent
            )
        ]

        n = int(
            env.action_spaces()[
                agent
            ].n
        )

        if n != len(labels):
            raise RuntimeError(
                f"{agent}: "
                f"space.n={n}, "
                f"labels={len(labels)}"
            )

        if len(labels) != len(mask):
            raise RuntimeError(
                f"{agent}: "
                "labels/mask mismatch"
            )

        if (
            n
            != expected_spaces[
                agent
            ]
        ):
            raise RuntimeError(
                f"{agent}: "
                f"action-space size={n}, "
                f"expected="
                f"{expected_spaces[agent]}"
            )

        if (
            len(subnets)
            != EXPECTED_SUBNET_COUNT[
                agent
            ]
        ):
            raise RuntimeError(
                f"{agent}: "
                f"subnets={subnets}, "
                "unexpected subnet count"
            )

        family_valid_counts = {}

        for case in (
            ACTION_CASES.values()
        ):
            family = str(
                case[
                    "family"
                ]
            )

            family_valid_counts[
                family
            ] = len(
                independent_valid_candidates(
                    env,
                    agent,
                    family,
                )
            )

            if (
                family_valid_counts[
                    family
                ]
                <= 0
            ):
                raise RuntimeError(
                    f"{agent}: "
                    f"no valid {family}"
                )

        result[
            agent
        ] = {
            "action_space_n":
                n,

            "n_valid_mask":
                int(
                    sum(
                        bool(value)
                        for value
                        in mask
                    )
                ),

            "wrapper_host_count":
                len(hosts),

            "subnets":
                subnets,

            "family_valid_counts":
                family_valid_counts,
        }

    return result


# ============================================================
# Single-agent dynamic resolver
# ============================================================

def resolve_single_action(
    *,
    env,
    adapter: CybORGActionAdapter,
    agent: str,
    action_id: int,
):
    """
    每次真正 launch 前都重新读取当前 wrapper：

        action_labels
        action_mask
        actions

    然后重新调用 production adapter。

    绝不缓存 raw action index。

    返回：

        executed_index
        resolution_record
        actual executed duration

    actual duration 来自 underlying
    executed action object.duration。

    因此如果未来发生：

        requested Restore d=5
            ↓
        fallback Sleep d=1

    scheduler 会按 1 tick 重新 ready，
    而不是错误等待 5 tick。
    """

    action_id = int(
        action_id
    )

    case = ACTION_CASES[
        action_id
    ]

    requested_family = str(
        case[
            "family"
        ]
    )

    # --------------------------------------------------------
    # Independent expected result
    # --------------------------------------------------------

    candidates = (
        independent_valid_candidates(
            env,
            agent,
            requested_family,
        )
    )

    expected_target = None
    expected_index = None

    if requested_family == "Sleep":

        if len(candidates) != 1:
            raise RuntimeError(
                f"{agent}: expected exactly "
                "one valid real Sleep, "
                f"got={candidates}"
            )

        scores = None

        expected_index = int(
            candidates[0][
                "index"
            ]
        )

    elif candidates:

        (
            scores,
            expected_target,
        ) = (
            synthetic_observable_scores(
                candidates
            )
        )

        expected_candidates = [
            item
            for item
            in candidates
            if (
                item[
                    "target_host"
                ]
                == expected_target
            )
        ]

        if len(
            expected_candidates
        ) != 1:
            raise RuntimeError(
                f"{agent}: expected "
                "one deterministic "
                f"{requested_family} "
                f"candidate for "
                f"{expected_target}"
            )

        expected_index = int(
            expected_candidates[
                0
            ][
                "index"
            ]
        )

    else:

        # Dynamic no-target case.
        #
        # Production adapter 应 fallback Sleep。
        scores = {}

    # --------------------------------------------------------
    # Production adapter
    # --------------------------------------------------------

    resolution = (
        adapter.resolve(
            env=env,
            agent_name=agent,
            action_id=action_id,
            observable_host_scores=scores,
        )
    )

    if (
        resolution
        .requested_action_id
        != action_id
    ):
        raise RuntimeError(
            f"{agent}: requested "
            "action ID mismatch"
        )

    if (
        resolution
        .requested_cyborg_family
        != requested_family
    ):
        raise RuntimeError(
            f"{agent}: requested "
            "family mismatch"
        )

    # --------------------------------------------------------
    # Refresh current wrapper again
    # --------------------------------------------------------

    labels = list(
        env.action_labels(
            agent
        )
    )

    mask = list(
        env.action_mask(
            agent
        )
    )

    actions = list(
        env.actions(
            agent
        )
    )

    index = int(
        resolution
        .executed_index
    )

    if not (
        0
        <= index
        < len(labels)
    ):
        raise RuntimeError(
            f"{agent}: invalid "
            f"executed index={index}"
        )

    if not bool(
        mask[index]
    ):
        raise RuntimeError(
            f"{agent}: adapter selected "
            "mask=False slot"
        )

    wrapper_label = str(
        labels[
            index
        ]
    ).strip()

    if (
        wrapper_label
        != resolution
        .executed_label
    ):
        raise RuntimeError(
            f"{agent}: executed label "
            "does not match current wrapper"
        )

    action_obj = actions[
        index
    ]

    executed_family = (
        type(
            action_obj
        ).__name__
    )

    if (
        executed_family
        != resolution
        .executed_action_family
    ):
        raise RuntimeError(
            f"{agent}: resolution "
            "executed family="
            f"{resolution.executed_action_family}, "
            "underlying class="
            f"{executed_family}"
        )

    # --------------------------------------------------------
    # Requested action expected result
    # --------------------------------------------------------

    if requested_family == "Sleep":

        if resolution.fallback:
            raise RuntimeError(
                f"{agent}: Sleep must "
                "not be fallback"
            )

        if (
            resolution
            .target_host
            is not None
        ):
            raise RuntimeError(
                f"{agent}: Sleep cannot "
                "have host target"
            )

        if (
            expected_index
            is not None
            and index
            != expected_index
        ):
            raise RuntimeError(
                f"{agent}: Sleep index="
                f"{index}, expected="
                f"{expected_index}"
            )

    elif candidates:

        # 当前确实存在合法 target，
        # 因此不应该 fallback。
        if resolution.fallback:
            raise RuntimeError(
                f"{agent}: unexpected "
                f"{requested_family} "
                "fallback: "
                f"{resolution.fallback_reason}"
            )

        if (
            resolution.target_host
            != expected_target
        ):
            raise RuntimeError(
                f"{agent}: target="
                f"{resolution.target_host}, "
                f"expected="
                f"{expected_target}"
            )

        if (
            expected_index
            is not None
            and index
            != expected_index
        ):
            raise RuntimeError(
                f"{agent}: executed "
                f"index={index}, "
                f"expected={expected_index}"
            )

    else:

        # targeted action 当前无合法 target，
        # production adapter 必须 fallback Sleep。
        if not resolution.fallback:
            raise RuntimeError(
                f"{agent}: expected "
                f"{requested_family} "
                "fallback but fallback=False"
            )

        if (
            executed_family
            != "Sleep"
        ):
            raise RuntimeError(
                f"{agent}: fallback "
                "must execute Sleep; "
                f"got={executed_family}"
            )

        if (
            resolution.target_host
            is not None
        ):
            raise RuntimeError(
                f"{agent}: fallback Sleep "
                "cannot have target"
            )

    # --------------------------------------------------------
    # Underlying target oracle
    # --------------------------------------------------------

    if (
        executed_family
        != "Sleep"
    ):
        object_target = getattr(
            action_obj,
            "hostname",
            None,
        )

        if (
            str(
                object_target
            )
            != str(
                resolution
                .target_host
            )
        ):
            raise RuntimeError(
                f"{agent}: underlying "
                "action target mismatch"
            )

    # --------------------------------------------------------
    # CRITICAL:
    # actual executed duration
    # --------------------------------------------------------

    executed_duration = int(
        action_obj.duration
    )

    if (
        executed_family
        not in FAMILY_DURATION
    ):
        raise RuntimeError(
            f"{agent}: unsupported "
            "executed family="
            f"{executed_family}"
        )

    expected_duration = (
        FAMILY_DURATION[
            executed_family
        ]
    )

    if (
        executed_duration
        != expected_duration
    ):
        raise RuntimeError(
            f"{agent}: underlying "
            f"{executed_family} "
            f"duration="
            f"{executed_duration}, "
            f"expected="
            f"{expected_duration}"
        )

    record = {
        "requested_action_id":
            int(
                resolution
                .requested_action_id
            ),

        "requested_action_name":
            str(
                resolution
                .requested_action_name
            ),

        "requested_family":
            str(
                resolution
                .requested_cyborg_family
            ),

        "executed_index":
            index,

        "executed_label":
            str(
                resolution
                .executed_label
            ),

        "executed_family":
            str(
                resolution
                .executed_action_family
            ),

        "executed_duration":
            int(
                executed_duration
            ),

        "target_host":
            resolution
            .target_host,

        "fallback":
            bool(
                resolution
                .fallback
            ),

        "fallback_reason":
            resolution
            .fallback_reason,

        "synthetic_scores":
            scores,

        "expected_target":
            expected_target,
    }

    return (
        index,
        record,
        executed_duration,
    )


# ============================================================
# Original synchronous joint action resolver
# ============================================================

def resolve_joint_actions(
    env,
    action_id: int,
):
    """
    A3.6 原始 synchronous integration case。

    五个 Blue agent 同时提交
    同一个 high-level action family。
    """

    adapter = (
        CybORGActionAdapter()
    )

    action_id = int(
        action_id
    )

    expected_family = str(
        ACTION_CASES[
            action_id
        ][
            "family"
        ]
    )

    joint_actions = {}
    resolutions = {}

    for agent in blue_agents(
        env
    ):
        (
            index,
            record,
            _executed_duration,
        ) = resolve_single_action(
            env=env,
            adapter=adapter,
            agent=agent,
            action_id=action_id,
        )

        # synchronous case 使用 fresh env，
        # 当前所有四种 action family
        # 应均存在合法 action。
        if record[
            "fallback"
        ]:
            raise RuntimeError(
                f"{agent}: unexpected "
                f"fallback for "
                f"{expected_family}"
            )

        if (
            record[
                "executed_family"
            ]
            != expected_family
        ):
            raise RuntimeError(
                f"{agent}: executed "
                f"family="
                f"{record['executed_family']}, "
                f"expected="
                f"{expected_family}"
            )

        joint_actions[
            agent
        ] = index

        resolutions[
            agent
        ] = record

    return (
        joint_actions,
        resolutions,
    )


# ============================================================
# Original synchronous tick validator
# ============================================================

def validate_tick_state(
    *,
    controller,
    agents: list[str],
    family: str,
    duration: int,
    local_tick: int,
):
    result = {}

    for agent in agents:

        progress = (
            controller_progress(
                controller,
                agent,
            )
        )

        executed = (
            controller_executed(
                controller,
                agent,
            )
        )

        executed_classes = [
            item[
                "action_class"
            ]
            for item
            in executed
        ]

        if (
            local_tick
            < duration
        ):
            if progress is None:
                raise RuntimeError(
                    f"{agent}: {family} "
                    "ended too early"
                )

            if (
                progress[
                    "action_class"
                ]
                != family
            ):
                raise RuntimeError(
                    f"{agent}: in-progress "
                    "family mismatch"
                )

            expected_remaining = (
                duration
                - local_tick
            )

            if (
                progress[
                    "remaining_ticks"
                ]
                != expected_remaining
            ):
                raise RuntimeError(
                    f"{agent}: remaining="
                    f"{progress['remaining_ticks']}, "
                    f"expected="
                    f"{expected_remaining}"
                )

            # CC4 internal controller：
            # multi-tick action 尚未完成时，
            # 当前 tick 实际执行 Sleep。
            if (
                executed_classes
                != ["Sleep"]
            ):
                raise RuntimeError(
                    f"{agent}: busy tick "
                    "must execute Sleep; "
                    f"got="
                    f"{executed_classes}"
                )

            decision_available = (
                False
            )

        else:

            if progress is not None:
                raise RuntimeError(
                    f"{agent}: action remains "
                    "busy after completion"
                )

            if (
                executed_classes
                != [family]
            ):
                raise RuntimeError(
                    f"{agent}: completion "
                    f"tick executed="
                    f"{executed_classes}, "
                    f"expected="
                    f"{[family]}"
                )

            decision_available = (
                True
            )

        result[
            agent
        ] = {
            "progress":
                progress,

            "controller_executed":
                executed,

            "decision_available_after":
                decision_available,
        }

    return result


# ============================================================
# Original synchronous action-family case
# ============================================================

def run_action_case(
    *,
    seed: int,
    steps: int,
    pad_spaces: bool,
    action_id: int,
):
    """
    每个 family 使用一个全新 CC4 env，
    防止 Analyse / Remove / Restore
    相互污染。
    """

    env = make_env(
        seed=seed,
        steps=steps,
        pad_spaces=pad_spaces,
    )

    agents = blue_agents(
        env
    )

    case = ACTION_CASES[
        action_id
    ]

    family = str(
        case[
            "family"
        ]
    )

    duration = int(
        case[
            "duration"
        ]
    )

    (
        joint_actions,
        resolutions,
    ) = resolve_joint_actions(
        env,
        action_id,
    )

    controller = (
        env.env
        .environment_controller
    )

    start_tick = int(
        controller.step_count
    )

    tick_records = []

    # --------------------------------------------------------
    # First tick:
    # all five Blue agents launch
    # same high-level family.
    # --------------------------------------------------------

    env.step(
        actions=joint_actions
    )

    tick_records.append(
        {
            "local_tick":
                1,

            "global_tick":
                int(
                    controller
                    .step_count
                ),

            "agents":
                validate_tick_state(
                    controller=controller,
                    agents=agents,
                    family=family,
                    duration=duration,
                    local_tick=1,
                ),
        }
    )

    # --------------------------------------------------------
    # Remaining busy ticks.
    # --------------------------------------------------------

    for local_tick in range(
        2,
        duration + 1,
    ):
        env.step(
            actions={}
        )

        tick_records.append(
            {
                "local_tick":
                    int(
                        local_tick
                    ),

                "global_tick":
                    int(
                        controller
                        .step_count
                    ),

                "agents":
                    validate_tick_state(
                        controller=controller,
                        agents=agents,
                        family=family,
                        duration=duration,
                        local_tick=local_tick,
                    ),
            }
        )

    end_tick = int(
        controller.step_count
    )

    decision_dt = (
        end_tick
        - start_tick
    )

    if (
        decision_dt
        != duration
    ):
        raise RuntimeError(
            f"{family}: "
            f"decision_dt="
            f"{decision_dt}, "
            f"expected="
            f"{duration}"
        )

    return {
        "action_id":
            int(
                action_id
            ),

        "action_name":
            str(
                case[
                    "name"
                ]
            ),

        "family":
            family,

        "duration":
            duration,

        "global_tick_start":
            start_tick,

        "global_tick_end":
            end_tick,

        "decision_dt":
            decision_dt,

        "resolutions":
            resolutions,

        "ticks":
            tick_records,
    }


# ============================================================
# Original synchronous seed / layout run
# ============================================================

def run_one(
    *,
    seed: int,
    steps: int,
    pad_spaces: bool,
):
    layout_env = make_env(
        seed=seed,
        steps=steps,
        pad_spaces=pad_spaces,
    )

    layout = validate_layout(
        layout_env,
        pad_spaces,
    )

    actions = {}

    for action_id in (
        0,
        1,
        2,
        3,
    ):
        actions[
            str(
                action_id
            )
        ] = run_action_case(
            seed=seed,
            steps=steps,
            pad_spaces=pad_spaces,
            action_id=action_id,
        )

    return {
        "seed":
            int(seed),

        "pad_spaces":
            bool(
                pad_spaces
            ),

        "layout":
            layout,

        "actions":
            actions,
    }


# ============================================================
# NEW A3.6 asynchronous mixed-duration run
# ============================================================

def run_async_one(
    *,
    seed: int,
    steps: int,
    pad_spaces: bool,
):
    """
    真正验证 asynchronous multi-agent
    decision readiness。

    ==========================================================
    Scheduler rule
    ==========================================================

    Scheduler 判断某 agent 是否 ready 的信息
    只能来自：

        scheduler-local active metadata
        scheduler-local ready_at

    ready_at 只由：

        launch_time
        +
        actual executed action.duration

    得到。

    ==========================================================
    Forbidden readiness sources
    ==========================================================

    scheduler 决策时绝不能读取：

        controller.actions_in_progress
        controller.action
        action_mask

    其中：

        action_labels / action_mask

    只允许在 agent 已经 scheduler-ready 后，
    用于正式 adapter 解析当前可执行底层动作。

    controller internals 只允许在 env.step 之后
    做 oracle assertion。
    """

    env = make_env(
        seed=seed,
        steps=steps,
        pad_spaces=pad_spaces,
    )

    agents = blue_agents(
        env
    )

    layout = validate_layout(
        env,
        pad_spaces,
    )

    adapter = (
        CybORGActionAdapter()
    )

    controller = (
        env.env
        .environment_controller
    )

    start_global_tick = int(
        controller.step_count
    )

    # 每个 agent 下一条 queue action
    queue_pos = {
        agent: 0
        for agent
        in agents
    }

    # ========================================================
    # Scheduler-owned readiness
    #
    # 这是 readiness 的唯一来源。
    # ========================================================

    ready_at = {
        agent: 0
        for agent
        in agents
    }

    # 当前 scheduler 认为正在执行的动作。
    #
    # 注意：
    # 这里完全是 local metadata，
    # 不是 controller.actions_in_progress。
    active: dict[
        str,
        dict[str, Any] | None,
    ] = {
        agent: None
        for agent
        in agents
    }

    launches = []
    completions = []
    tick_records = []

    local_time = 0

    while True:

        pending = any(
            queue_pos[
                agent
            ]
            <
            len(
                ASYNC_ACTION_QUEUES[
                    agent
                ]
            )
            for agent
            in agents
        )

        running = any(
            active[
                agent
            ]
            is not None
            for agent
            in agents
        )

        if (
            not pending
            and not running
        ):
            break

        # ====================================================
        # 1. Scheduler decides who is ready.
        #
        # CRITICAL:
        # No controller state is read here.
        # ====================================================

        launch_agents = [
            agent
            for agent
            in agents
            if (
                active[
                    agent
                ]
                is None

                and

                queue_pos[
                    agent
                ]
                <
                len(
                    ASYNC_ACTION_QUEUES[
                        agent
                    ]
                )

                and

                ready_at[
                    agent
                ]
                == local_time
            )
        ]

        expected_launch_agents = (
            ASYNC_EXPECTED_LAUNCHES
            .get(
                local_time,
                [],
            )
        )

        if (
            launch_agents
            != expected_launch_agents
        ):
            raise RuntimeError(
                "async launch schedule "
                "mismatch at "
                f"t={local_time}: "
                f"got={launch_agents}, "
                f"expected="
                f"{expected_launch_agents}"
            )

        # 哪些 agent 在本 tick 开始时仍 busy。
        #
        # 只来自 scheduler-local active。
        scheduler_busy_before = [
            agent
            for agent
            in agents
            if (
                active[
                    agent
                ]
                is not None
            )
        ]

        joint_actions = {}

        launch_records_this_tick = []

        # ====================================================
        # 2. Launch only ready agents.
        # ====================================================

        for agent in (
            launch_agents
        ):

            # ------------------------------------------------
            # Controller check is ORACLE ONLY.
            #
            # Scheduler has already decided agent is ready.
            # ------------------------------------------------

            if (
                controller_progress(
                    controller,
                    agent,
                )
                is not None
            ):
                raise RuntimeError(
                    f"{agent}: scheduler "
                    "says ready at "
                    f"t={local_time}, "
                    "but controller still "
                    "has action in progress"
                )

            position = int(
                queue_pos[
                    agent
                ]
            )

            action_id = int(
                ASYNC_ACTION_QUEUES[
                    agent
                ][
                    position
                ]
            )

            (
                index,
                resolution_record,
                executed_duration,
            ) = resolve_single_action(
                env=env,
                adapter=adapter,
                agent=agent,
                action_id=action_id,
            )

            joint_actions[
                agent
            ] = int(index)

            queue_pos[
                agent
            ] += 1

            # =================================================
            # CRITICAL:
            #
            # readiness 使用 ACTUAL executed duration。
            #
            # 不能使用 requested nominal duration。
            # =================================================

            next_ready_at = (
                local_time
                + int(
                    executed_duration
                )
            )

            ready_at[
                agent
            ] = int(
                next_ready_at
            )

            active[
                agent
            ] = {
                "queue_position":
                    position,

                "requested_action_id":
                    action_id,

                "executed_family":
                    str(
                        resolution_record[
                            "executed_family"
                        ]
                    ),

                "executed_duration":
                    int(
                        executed_duration
                    ),

                "launch_time":
                    int(
                        local_time
                    ),

                "ready_at":
                    int(
                        next_ready_at
                    ),
            }

            launch_record = {
                "scheduler_time":
                    int(
                        local_time
                    ),

                "agent":
                    agent,

                "queue_position":
                    position,

                **resolution_record,

                "scheduler_ready_at":
                    int(
                        next_ready_at
                    ),
            }

            launches.append(
                launch_record
            )

            launch_records_this_tick.append(
                launch_record
            )

        # ====================================================
        # 3. Busy agents MUST NOT be explicitly submitted.
        #
        # No Sleep filler.
        # ====================================================

        submitted_agents = sorted(
            joint_actions.keys()
        )

        for busy_agent in (
            scheduler_busy_before
        ):
            if (
                busy_agent
                in joint_actions
            ):
                raise RuntimeError(
                    f"{busy_agent}: "
                    "busy agent was "
                    "incorrectly submitted"
                )

        # ====================================================
        # 4. One real CC4 global tick.
        #
        # Missing agents are intentionally omitted.
        # ====================================================

        env.step(
            actions=joint_actions
        )

        end_local_time = (
            local_time + 1
        )

        actual_global_tick = int(
            controller.step_count
        )

        expected_global_tick = (
            start_global_tick
            + end_local_time
        )

        if (
            actual_global_tick
            != expected_global_tick
        ):
            raise RuntimeError(
                "global tick mismatch: "
                f"got="
                f"{actual_global_tick}, "
                f"expected="
                f"{expected_global_tick}"
            )

        # ====================================================
        # 5. AFTER-THE-FACT controller oracle.
        #
        # 此时 controller 才允许参与检查。
        # ====================================================

        oracle_after = {}

        for agent in agents:

            meta = active[
                agent
            ]

            if meta is None:

                # Queue 已完成或者 agent idle。
                #
                # CC4 background SleepAgent
                # 可能产生自己的默认 Sleep。
                #
                # 这不属于 scheduler launch，
                # 所以这里不作为错误。
                continue

            progress = (
                controller_progress(
                    controller,
                    agent,
                )
            )

            executed = (
                controller_executed(
                    controller,
                    agent,
                )
            )

            executed_classes = [
                item[
                    "action_class"
                ]
                for item
                in executed
            ]

            ready_time = int(
                meta[
                    "ready_at"
                ]
            )

            family = str(
                meta[
                    "executed_family"
                ]
            )

            # ------------------------------------------------
            # Still busy
            # ------------------------------------------------

            if (
                end_local_time
                < ready_time
            ):

                if progress is None:
                    raise RuntimeError(
                        f"{agent}: "
                        f"{family} "
                        "ended too early"
                    )

                if (
                    progress[
                        "action_class"
                    ]
                    != family
                ):
                    raise RuntimeError(
                        f"{agent}: "
                        "in-progress "
                        "family mismatch: "
                        f"got="
                        f"{progress['action_class']}, "
                        f"expected={family}"
                    )

                expected_remaining = (
                    ready_time
                    - end_local_time
                )

                if (
                    progress[
                        "remaining_ticks"
                    ]
                    != expected_remaining
                ):
                    raise RuntimeError(
                        f"{agent}: "
                        "remaining ticks="
                        f"{progress['remaining_ticks']}, "
                        f"expected="
                        f"{expected_remaining}"
                    )

                # CybORG controller semantics:
                # multi-tick busy tick executes Sleep.
                if (
                    executed_classes
                    != ["Sleep"]
                ):
                    raise RuntimeError(
                        f"{agent}: busy "
                        "controller execution "
                        "must be Sleep; "
                        f"got="
                        f"{executed_classes}"
                    )

                oracle_after[
                    agent
                ] = {
                    "scheduler_state":
                        "busy",

                    "ready_at":
                        ready_time,

                    "controller_progress":
                        progress,

                    "controller_executed":
                        executed,
                }

            # ------------------------------------------------
            # Completion exactly at scheduler ready_at
            # ------------------------------------------------

            elif (
                end_local_time
                == ready_time
            ):

                if progress is not None:
                    raise RuntimeError(
                        f"{agent}: controller "
                        "still busy at "
                        "scheduler ready time"
                    )

                if (
                    executed_classes
                    != [family]
                ):
                    raise RuntimeError(
                        f"{agent}: completion "
                        "executed="
                        f"{executed_classes}, "
                        f"expected="
                        f"{[family]}"
                    )

                completion = {
                    "scheduler_time":
                        int(
                            end_local_time
                        ),

                    "agent":
                        agent,

                    "queue_position":
                        int(
                            meta[
                                "queue_position"
                            ]
                        ),

                    "executed_family":
                        family,

                    "launch_time":
                        int(
                            meta[
                                "launch_time"
                            ]
                        ),

                    "executed_duration":
                        int(
                            meta[
                                "executed_duration"
                            ]
                        ),
                }

                completions.append(
                    completion
                )

                oracle_after[
                    agent
                ] = {
                    "scheduler_state":
                        "completed",

                    "ready_at":
                        ready_time,

                    "controller_progress":
                        None,

                    "controller_executed":
                        executed,
                }

                # Scheduler-local transition:
                # agent becomes available.
                active[
                    agent
                ] = None

            else:

                # 若出现 end_local_time > ready_at，
                # 说明 scheduler 已经错过 decision epoch。
                raise RuntimeError(
                    f"{agent}: scheduler "
                    "passed ready_at="
                    f"{ready_time}, "
                    f"current="
                    f"{end_local_time}"
                )

        tick_records.append(
            {
                "scheduler_time_start":
                    int(
                        local_time
                    ),

                "scheduler_time_end":
                    int(
                        end_local_time
                    ),

                "global_tick":
                    actual_global_tick,

                "scheduler_busy_before":
                    scheduler_busy_before,

                "submitted_agents":
                    submitted_agents,

                "launches":
                    launch_records_this_tick,

                "oracle_after":
                    oracle_after,
            }
        )

        local_time = (
            end_local_time
        )

        # Safety guard.
        if local_time > 20:
            raise RuntimeError(
                "async probe exceeded "
                "safety horizon"
            )

    # ========================================================
    # Final async contract assertions
    # ========================================================

    expected_launch_count = (
        len(agents)
        * 2
    )

    if (
        len(launches)
        != expected_launch_count
    ):
        raise RuntimeError(
            "unexpected async "
            "launch count: "
            f"{len(launches)}, "
            f"expected="
            f"{expected_launch_count}"
        )

    if (
        len(completions)
        != expected_launch_count
    ):
        raise RuntimeError(
            "unexpected async "
            "completion count: "
            f"{len(completions)}, "
            f"expected="
            f"{expected_launch_count}"
        )

    if (
        local_time
        != ASYNC_EXPECTED_FINAL_TIME
    ):
        raise RuntimeError(
            "unexpected async "
            "final scheduler time: "
            f"{local_time}, "
            f"expected="
            f"{ASYNC_EXPECTED_FINAL_TIME}"
        )

    for agent in agents:

        if (
            queue_pos[
                agent
            ]
            != len(
                ASYNC_ACTION_QUEUES[
                    agent
                ]
            )
        ):
            raise RuntimeError(
                f"{agent}: async "
                "queue not exhausted"
            )

        if (
            active[
                agent
            ]
            is not None
        ):
            raise RuntimeError(
                f"{agent}: async "
                "probe finished "
                "while still busy"
            )

    # Verify exact launch schedule from records.
    actual_launch_schedule = {}

    for item in launches:

        t = int(
            item[
                "scheduler_time"
            ]
        )

        actual_launch_schedule.setdefault(
            t,
            [],
        ).append(
            str(
                item[
                    "agent"
                ]
            )
        )

    for t in (
        actual_launch_schedule
    ):
        actual_launch_schedule[
            t
        ] = sorted(
            actual_launch_schedule[
                t
            ]
        )

    expected_schedule = {
        int(t):
            sorted(agents_at_t)
        for (
            t,
            agents_at_t,
        )
        in (
            ASYNC_EXPECTED_LAUNCHES
            .items()
        )
    }

    if (
        actual_launch_schedule
        != expected_schedule
    ):
        raise RuntimeError(
            "final async launch "
            "schedule mismatch: "
            f"got="
            f"{actual_launch_schedule}, "
            f"expected="
            f"{expected_schedule}"
        )

    return {
        "seed":
            int(seed),

        "pad_spaces":
            bool(
                pad_spaces
            ),

        "layout":
            layout,

        "action_queues":
            ASYNC_ACTION_QUEUES,

        "expected_launch_schedule":
            ASYNC_EXPECTED_LAUNCHES,

        "actual_launch_schedule":
            actual_launch_schedule,

        "launches":
            launches,

        "completions":
            completions,

        "ticks":
            tick_records,

        "final_scheduler_time":
            int(
                local_time
            ),

        "launch_count":
            len(
                launches
            ),

        "completion_count":
            len(
                completions
            ),

        "all_pass":
            True,
    }


# ============================================================
# CLI seed parser
# ============================================================

def parse_seeds(
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

    return result


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "A3.6 real CC4 five-Blue-agent "
            "four-action synchronous + "
            "asynchronous integration probe."
        )
    )

    parser.add_argument(
        "--seeds",
        default="42,43,44",
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
            "multi_agent_integration.json"
        ),
    )

    args = (
        parser.parse_args()
    )

    seeds = parse_seeds(
        args.seeds
    )

    output = {
        "contract": {
            "agents":
                EXPECTED_AGENTS,

            "actions":
                ACTION_CASES,

            "pad_modes": [
                False,
                True,
            ],

            "probe_host_scores":
                (
                    "synthetic deterministic "
                    "integration-only; "
                    "NOT A4 planner evidence"
                ),

            "raw_action_indices":
                (
                    "recorded only; "
                    "never hard-coded "
                    "in production"
                ),

            "async_action_queues":
                ASYNC_ACTION_QUEUES,

            "async_expected_launches":
                ASYNC_EXPECTED_LAUNCHES,

            "async_expected_final_time":
                ASYNC_EXPECTED_FINAL_TIME,

            "async_readiness_source":
                (
                    "scheduler-local executed "
                    "action duration only; "
                    "controller internals are "
                    "oracle assertions only"
                ),

            "busy_agent_submission":
                (
                    "busy agents are omitted; "
                    "never submit explicit "
                    "Sleep as busy filler"
                ),
        },

        # Existing synchronous runs.
        "runs": [],

        # New asynchronous runs.
        "async_runs": [],
    }

    total_resolutions = 0
    total_joint_cases = 0

    total_async_launches = 0
    total_async_completions = 0

    # ========================================================
    # 3 seeds × 2 padding modes
    # ========================================================

    for pad_spaces in (
        False,
        True,
    ):
        for seed in seeds:

            print()
            print(
                "=" * 80
            )

            print(
                "[RUN]",
                f"seed={seed}",
                f"pad_spaces="
                f"{pad_spaces}",
            )

            # =================================================
            # Synchronous A3.6 cases
            # =================================================

            run = run_one(
                seed=seed,
                steps=int(
                    args.steps
                ),
                pad_spaces=pad_spaces,
            )

            output[
                "runs"
            ].append(
                run
            )

            for agent in (
                EXPECTED_AGENTS
            ):
                meta = (
                    run[
                        "layout"
                    ][
                        agent
                    ]
                )

                print(
                    agent,
                    "space=",
                    meta[
                        "action_space_n"
                    ],
                    "valid=",
                    meta[
                        "n_valid_mask"
                    ],
                    "hosts=",
                    meta[
                        "wrapper_host_count"
                    ],
                    "subnets=",
                    meta[
                        "subnets"
                    ],
                )

            for action_id in (
                "0",
                "1",
                "2",
                "3",
            ):
                action_result = (
                    run[
                        "actions"
                    ][
                        action_id
                    ]
                )

                print(
                    "  [PASS]",
                    action_result[
                        "action_name"
                    ],
                    "family=",
                    action_result[
                        "family"
                    ],
                    "decision_dt=",
                    action_result[
                        "decision_dt"
                    ],
                )

                total_resolutions += len(
                    action_result[
                        "resolutions"
                    ]
                )

                total_joint_cases += 1

            # =================================================
            # Asynchronous A3.6 extension
            # =================================================

            async_run = (
                run_async_one(
                    seed=seed,
                    steps=int(
                        args.steps
                    ),
                    pad_spaces=(
                        pad_spaces
                    ),
                )
            )

            output[
                "async_runs"
            ].append(
                async_run
            )

            total_async_launches += len(
                async_run[
                    "launches"
                ]
            )

            total_async_completions += len(
                async_run[
                    "completions"
                ]
            )

            print(
                "  [ASYNC PASS]",
                "launches=",
                async_run[
                    "launch_count"
                ],
                "completions=",
                async_run[
                    "completion_count"
                ],
                "final_time=",
                async_run[
                    "final_scheduler_time"
                ],
            )

            print(
                "    launch schedule:",
                async_run[
                    "actual_launch_schedule"
                ],
            )

    # ========================================================
    # Global synchronous assertions
    # ========================================================

    expected_sync_runs = (
        len(seeds)
        * 2
    )

    if (
        len(
            output[
                "runs"
            ]
        )
        != expected_sync_runs
    ):
        raise RuntimeError(
            "unexpected synchronous "
            "run count"
        )

    # 3 seeds × 2 pad modes ×
    # 4 actions × 5 agents
    expected_resolutions = (
        len(seeds)
        * 2
        * 4
        * 5
    )

    if (
        total_resolutions
        != expected_resolutions
    ):
        raise RuntimeError(
            "unexpected resolution count: "
            f"{total_resolutions}, "
            f"expected="
            f"{expected_resolutions}"
        )

    expected_joint_cases = (
        len(seeds)
        * 2
        * 4
    )

    if (
        total_joint_cases
        != expected_joint_cases
    ):
        raise RuntimeError(
            "unexpected joint "
            "action case count: "
            f"{total_joint_cases}, "
            f"expected="
            f"{expected_joint_cases}"
        )

    # ========================================================
    # Global asynchronous assertions
    # ========================================================

    expected_async_runs = (
        len(seeds)
        * 2
    )

    if (
        len(
            output[
                "async_runs"
            ]
        )
        != expected_async_runs
    ):
        raise RuntimeError(
            "unexpected async "
            "run count: "
            f"{len(output['async_runs'])}, "
            f"expected="
            f"{expected_async_runs}"
        )

    # 每个 async run：
    # 5 agents × 2 launches = 10.
    expected_async_launches = (
        expected_async_runs
        * len(
            EXPECTED_AGENTS
        )
        * 2
    )

    if (
        total_async_launches
        != expected_async_launches
    ):
        raise RuntimeError(
            "unexpected async "
            "launch count: "
            f"{total_async_launches}, "
            f"expected="
            f"{expected_async_launches}"
        )

    if (
        total_async_completions
        != expected_async_launches
    ):
        raise RuntimeError(
            "unexpected async "
            "completion count: "
            f"{total_async_completions}, "
            f"expected="
            f"{expected_async_launches}"
        )

    # ========================================================
    # Summary
    # ========================================================

    output[
        "summary"
    ] = {
        "runs":
            len(
                output[
                    "runs"
                ]
            ),

        "total_adapter_resolutions":
            int(
                total_resolutions
            ),

        "total_joint_action_cases":
            int(
                total_joint_cases
            ),

        "sync_all_pass":
            True,

        "async_runs":
            len(
                output[
                    "async_runs"
                ]
            ),

        "async_launches":
            int(
                total_async_launches
            ),

        "async_completions":
            int(
                total_async_completions
            ),

        "async_all_pass":
            True,

        "all_pass":
            True,
    }

    print()
    print(
        "=" * 80
    )

    print(
        "[A3.6 SUMMARY]"
    )

    print(
        "runs:",
        output[
            "summary"
        ][
            "runs"
        ],
    )

    print(
        "adapter resolutions:",
        total_resolutions,
    )

    print(
        "joint action cases:",
        total_joint_cases,
    )

    print(
        "sync_all_pass: True"
    )

    print(
        "async runs:",
        output[
            "summary"
        ][
            "async_runs"
        ],
    )

    print(
        "async launches:",
        total_async_launches,
    )

    print(
        "async completions:",
        total_async_completions,
    )

    print(
        "async_all_pass: True"
    )

    print(
        "all_pass: True"
    )

    # ========================================================
    # Save probe-only JSON
    # ========================================================

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
    ) as handle:
        json.dump(
            output,
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(
        "[OK] multi-agent integration "
        "probe saved:"
    )

    print(
        out_path
    )


if __name__ == "__main__":
    main()