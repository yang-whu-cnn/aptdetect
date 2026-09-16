from __future__ import annotations

import argparse
import json
import os
import sys

from pathlib import Path
from typing import Any


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


EXPECTED_AGENTS = [
    "blue_agent_0",
    "blue_agent_1",
    "blue_agent_2",
    "blue_agent_3",
    "blue_agent_4",
]

# A3.1 已真实确认。
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

    env.reset(seed=int(seed))

    return env


def blue_agents(env) -> list[str]:
    agents = sorted(
        str(agent)
        for agent in env.action_spaces().keys()
        if "blue_agent_" in str(agent)
    )

    if agents != EXPECTED_AGENTS:
        raise RuntimeError(
            "unexpected Blue agents: "
            f"{agents}"
        )

    return agents


def independent_valid_candidates(
    env,
    agent: str,
    family: str,
) -> list[dict[str, Any]]:
    """
    独立于 production resolver 的 probe oracle。

    先 mask=True，
    再精确解析 family。

    这样不能出现“用被测 resolver
    生成自己的 expected result”的循环验证。
    """

    labels = list(
        env.action_labels(agent)
    )

    mask = list(
        env.action_mask(agent)
    )

    if len(labels) != len(mask):
        raise RuntimeError(
            f"{agent}: labels/mask mismatch"
        )

    result = []

    for index, (label, valid) in enumerate(
        zip(labels, mask)
    ):
        if not bool(valid):
            continue

        text = str(label).strip()

        if family == "Sleep":
            if text != "Sleep":
                continue

            target = None

        else:
            prefix = family + " "

            if not text.startswith(prefix):
                continue

            target = text[
                len(prefix):
            ].strip()

            if not target:
                continue

        result.append(
            {
                "index": int(index),
                "label": text,
                "target_host": target,
            }
        )

    return result


def synthetic_observable_scores(
    candidates: list[dict[str, Any]],
) -> tuple[
    dict[str, float],
    str,
]:
    """
    A3.6 probe-only synthetic evidence。

    不代表 A4 正式 host evidence。

    将候选 host 排序后赋予严格递增 score，
    因此 expected target 唯一且 deterministic。
    """

    hosts = sorted(
        {
            str(item["target_host"])
            for item in candidates
            if item["target_host"] is not None
        }
    )

    if not hosts:
        raise RuntimeError(
            "no host candidates"
        )

    scores = {
        host: float(index + 1)
        for index, host in enumerate(hosts)
    }

    expected_target = hosts[-1]

    return (
        scores,
        expected_target,
    )


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

    action = item["action"]

    return {
        "action_class":
            type(action).__name__,

        "action_repr":
            str(action),

        "duration":
            int(action.duration),

        "remaining_ticks":
            int(
                item["remaining_ticks"]
            ),
    }


def controller_executed(
    controller,
    agent: str,
) -> list[dict[str, Any]]:
    actions = (
        controller
        .action
        .get(agent, [])
    )

    return [
        {
            "action_class":
                type(action).__name__,

            "action_repr":
                str(action),

            "duration":
                int(action.duration),
        }
        for action in actions
    ]


def validate_layout(
    env,
    pad_spaces: bool,
):
    agents = blue_agents(env)

    expected_spaces = (
        EXPECTED_PADDED_SPACE
        if pad_spaces
        else EXPECTED_UNPADDED_SPACE
    )

    result = {}

    for agent in agents:
        labels = list(
            env.action_labels(agent)
        )

        mask = list(
            env.action_mask(agent)
        )

        hosts = [
            str(v)
            for v in env.hosts(agent)
        ]

        subnets = [
            str(v)
            for v in env.subnets(agent)
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

        if n != expected_spaces[agent]:
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

        for case in ACTION_CASES.values():
            family = case["family"]

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

        result[agent] = {
            "action_space_n": n,
            "n_valid_mask":
                int(sum(bool(v) for v in mask)),
            "wrapper_host_count":
                len(hosts),
            "subnets":
                subnets,
            "family_valid_counts":
                family_valid_counts,
        }

    return result


def resolve_joint_actions(
    env,
    action_id: int,
):
    adapter = CybORGActionAdapter()

    case = ACTION_CASES[
        action_id
    ]

    family = str(
        case["family"]
    )

    joint_actions = {}
    resolutions = {}

    for agent in blue_agents(env):

        candidates = (
            independent_valid_candidates(
                env,
                agent,
                family,
            )
        )

        if not candidates:
            raise RuntimeError(
                f"{agent}: no valid "
                f"{family} candidate"
            )

        if family == "Sleep":
            if len(candidates) != 1:
                raise RuntimeError(
                    f"{agent}: expected exactly "
                    "one valid real Sleep, "
                    f"got {candidates}"
                )

            scores = None
            expected_index = int(
                candidates[0]["index"]
            )
            expected_target = None

        else:
            (
                scores,
                expected_target,
            ) = synthetic_observable_scores(
                candidates
            )

            expected_candidates = [
                item
                for item in candidates
                if (
                    item["target_host"]
                    == expected_target
                )
            ]

            if len(expected_candidates) != 1:
                raise RuntimeError(
                    f"{agent}: expected one "
                    f"{family} action for "
                    f"{expected_target}"
                )

            expected_index = int(
                expected_candidates[0][
                    "index"
                ]
            )

        resolution = adapter.resolve(
            env=env,
            agent_name=agent,
            action_id=int(action_id),
            observable_host_scores=scores,
        )

        if (
            resolution.requested_action_id
            != action_id
        ):
            raise RuntimeError(
                f"{agent}: requested id mismatch"
            )

        if (
            resolution.requested_cyborg_family
            != family
        ):
            raise RuntimeError(
                f"{agent}: requested family "
                "mismatch"
            )

        if resolution.fallback:
            raise RuntimeError(
                f"{agent}: unexpected fallback "
                f"for {family}: "
                f"{resolution.fallback_reason}"
            )

        if (
            resolution.executed_index
            != expected_index
        ):
            raise RuntimeError(
                f"{agent}: adapter chose "
                f"index={resolution.executed_index}, "
                f"expected={expected_index}"
            )

        if (
            resolution.executed_action_family
            != family
        ):
            raise RuntimeError(
                f"{agent}: executed family "
                "mismatch"
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

        labels = list(
            env.action_labels(agent)
        )

        mask = list(
            env.action_mask(agent)
        )

        actions = list(
            env.actions(agent)
        )

        idx = int(
            resolution.executed_index
        )

        if not bool(mask[idx]):
            raise RuntimeError(
                f"{agent}: adapter selected "
                "mask=False slot"
            )

        if (
            str(labels[idx]).strip()
            != resolution.executed_label
        ):
            raise RuntimeError(
                f"{agent}: executed label "
                "does not match wrapper"
            )

        action_obj = actions[idx]

        if (
            type(action_obj).__name__
            != family
        ):
            raise RuntimeError(
                f"{agent}: underlying "
                f"action class="
                f"{type(action_obj).__name__}, "
                f"expected={family}"
            )

        if family != "Sleep":
            object_target = getattr(
                action_obj,
                "hostname",
                None,
            )

            if (
                str(object_target)
                != expected_target
            ):
                raise RuntimeError(
                    f"{agent}: action object "
                    "target mismatch"
                )

        joint_actions[
            agent
        ] = idx

        resolutions[
            agent
        ] = {
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
                idx,

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

            "target_host":
                resolution.target_host,

            "fallback":
                bool(
                    resolution.fallback
                ),

            "synthetic_scores":
                scores,

            "expected_target":
                expected_target,
        }

    return (
        joint_actions,
        resolutions,
    )


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
            item["action_class"]
            for item in executed
        ]

        if local_tick < duration:
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

            if executed_classes != [
                "Sleep"
            ]:
                raise RuntimeError(
                    f"{agent}: busy tick "
                    "must execute Sleep; "
                    f"got {executed_classes}"
                )

            decision_available = False

        else:
            if progress is not None:
                raise RuntimeError(
                    f"{agent}: action remains "
                    "busy after completion"
                )

            if executed_classes != [
                family
            ]:
                raise RuntimeError(
                    f"{agent}: completion tick "
                    f"executed={executed_classes}, "
                    f"expected={[family]}"
                )

            decision_available = True

        result[agent] = {
            "progress":
                progress,

            "controller_executed":
                executed,

            "decision_available_after":
                decision_available,
        }

    return result


def run_action_case(
    *,
    seed: int,
    steps: int,
    pad_spaces: bool,
    action_id: int,
):
    """
    每个 family 使用一个全新 CC4 env，
    防止 Analyse/Remove/Restore 相互污染。
    """

    env = make_env(
        seed=seed,
        steps=steps,
        pad_spaces=pad_spaces,
    )

    agents = blue_agents(env)

    case = ACTION_CASES[
        action_id
    ]

    family = str(
        case["family"]
    )

    duration = int(
        case["duration"]
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

    # tick 1：五个 Blue agent
    # 同时提交同一 high-level family
    env.step(
        actions=joint_actions
    )

    tick_records.append(
        {
            "local_tick": 1,
            "global_tick":
                int(
                    controller.step_count
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

    # multi-tick 的后续 busy ticks。
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
                    int(local_tick),

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

    if decision_dt != duration:
        raise RuntimeError(
            f"{family}: "
            f"decision_dt={decision_dt}, "
            f"expected={duration}"
        )

    return {
        "action_id":
            int(action_id),

        "action_name":
            str(case["name"]),

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
            str(action_id)
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
            bool(pad_spaces),

        "layout":
            layout,

        "actions":
            actions,
    }


def parse_seeds(
    value: str,
) -> list[int]:
    result = []

    for raw in str(value).split(","):
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


def main():
    parser = argparse.ArgumentParser(
        description=(
            "A3.6 real CC4 five-Blue-agent "
            "four-action integration probe."
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

    args = parser.parse_args()

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
        },

        "runs": [],
    }

    total_resolutions = 0
    total_joint_cases = 0

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
                f"pad_spaces={pad_spaces}",
            )

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

            for agent in EXPECTED_AGENTS:
                meta = run[
                    "layout"
                ][
                    agent
                ]

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
            total_resolutions,

        "total_joint_action_cases":
            total_joint_cases,

        "all_pass":
            True,
    }

    # 3 seeds × 2 pad modes ×
    # 4 actions × 5 agents
    if total_resolutions != (
        len(seeds)
        * 2
        * 4
        * 5
    ):
        raise RuntimeError(
            "unexpected resolution count: "
            f"{total_resolutions}"
        )

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
        "all_pass: True"
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
            output,
            f,
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