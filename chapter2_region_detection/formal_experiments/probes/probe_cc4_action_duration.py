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

from CybORG.Simulator.Scenarios import (  # type: ignore
    EnterpriseScenarioGenerator,
)

from CybORG.Agents import (  # type: ignore
    SleepAgent,
    FiniteStateRedAgent,
    EnterpriseGreenAgent,
)

from CybORG.Agents.Wrappers.BlueFixedActionWrapper import (  # type: ignore
    BlueFixedActionWrapper,
)


EXPECTED_DURATIONS = {
    "Sleep": 1,
    "Analyse": 2,
    "Remove": 3,
    "Restore": 5,
}


def make_env(
    seed: int,
    steps: int,
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

    return BlueFixedActionWrapper(
        cyborg,
        pad_spaces=False,
    )


def find_valid_action(
    env,
    agent: str,
    family: str,
):
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

    for idx, (
        label,
        valid,
    ) in enumerate(
        zip(
            labels,
            mask,
        )
    ):
        if not bool(
            valid
        ):
            continue

        text = str(
            label
        ).strip()

        if family == "Sleep":
            matched = (
                text == "Sleep"
            )
        else:
            matched = (
                text.startswith(
                    family + " "
                )
            )

        if matched:
            return {
                "index": int(
                    idx
                ),
                "label": text,
                "action": actions[
                    idx
                ],
            }

    raise RuntimeError(
        f"{agent}: no valid "
        f"{family} action"
    )


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
        "action_duration":
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


def executed_snapshot(
    controller,
    agent: str,
):
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
                str(
                    action
                ),
            "duration":
                int(
                    action.duration
                ),
        }
        for action in actions
    ]


def probe_family(
    seed: int,
    agent: str,
    family: str,
):
    env = make_env(
        seed=seed,
        steps=100,
    )

    env.reset(
        seed=seed
    )

    selected = find_valid_action(
        env,
        agent,
        family,
    )

    sleep = find_valid_action(
        env,
        agent,
        "Sleep",
    )

    action_obj = selected[
        "action"
    ]

    actual_duration = int(
        action_obj.duration
    )

    expected_duration = (
        EXPECTED_DURATIONS[
            family
        ]
    )

    if (
        actual_duration
        != expected_duration
    ):
        raise RuntimeError(
            f"{family}: "
            f"duration={actual_duration}, "
            f"expected={expected_duration}"
        )

    controller = (
        env.env
        .environment_controller
    )

    start_step = int(
        controller.step_count
    )

    rows = []

    for local_tick in range(
        1,
        actual_duration + 1,
    ):
        before = (
            progress_snapshot(
                controller,
                agent,
            )
        )

        # 第一个 tick 提交真正请求的动作。
        #
        # 后续 busy ticks 故意继续提交 Sleep，
        # 用于验证 controller 是否会：
        #
        # 1. 替换当前动作；
        # 2. 排队新动作；
        # 3. 忽略 busy 期间的新请求。
        if local_tick == 1:
            submitted_index = (
                selected[
                    "index"
                ]
            )

            submitted_label = (
                selected[
                    "label"
                ]
            )

        else:
            submitted_index = (
                sleep[
                    "index"
                ]
            )

            submitted_label = (
                sleep[
                    "label"
                ]
            )

        env.step(
            actions={
                agent:
                    submitted_index
            }
        )

        after = (
            progress_snapshot(
                controller,
                agent,
            )
        )

        executed = (
            executed_snapshot(
                controller,
                agent,
            )
        )

        rows.append(
            {
                "local_tick":
                    int(
                        local_tick
                    ),

                "global_step_after":
                    int(
                        controller
                        .step_count
                    ),

                "submitted_index":
                    int(
                        submitted_index
                    ),

                "submitted_label":
                    submitted_label,

                "progress_before":
                    before,

                "progress_after":
                    after,

                "controller_executed":
                    executed,

                "decision_available_after":
                    (
                        after is None
                    ),
            }
        )

    end_step = int(
        controller.step_count
    )

    decision_dt = (
        end_step
        - start_step
    )

    if (
        decision_dt
        != expected_duration
    ):
        raise RuntimeError(
            f"{family}: "
            f"decision_dt={decision_dt}, "
            f"expected="
            f"{expected_duration}"
        )

    if (
        progress_snapshot(
            controller,
            agent,
        )
        is not None
    ):
        raise RuntimeError(
            f"{family}: action still "
            "in progress after "
            f"{expected_duration} ticks"
        )

    # 再执行一次 Sleep。
    #
    # 如果这一步被正常接收，
    # 说明前一个动作完成后的下一个 tick
    # 已经确实是新的 decision epoch。
    next_decision_step_before = int(
        controller.step_count
    )

    env.step(
        actions={
            agent:
                sleep[
                    "index"
                ]
        }
    )

    next_decision_step_after = int(
        controller.step_count
    )

    post_completion = {
        "submitted":
            sleep[
                "label"
            ],

        "step_before":
            next_decision_step_before,

        "step_after":
            next_decision_step_after,

        "progress_after":
            progress_snapshot(
                controller,
                agent,
            ),

        "controller_executed":
            executed_snapshot(
                controller,
                agent,
            ),
    }

    return {
        "family":
            family,

        "selected_index":
            selected[
                "index"
            ],

        "selected_label":
            selected[
                "label"
            ],

        "action_class":
            type(
                action_obj
            ).__name__,

        "action_duration":
            actual_duration,

        "expected_duration":
            expected_duration,

        "global_tick_start":
            start_step,

        "global_tick_end":
            end_step,

        "decision_dt":
            decision_dt,

        "ticks":
            rows,

        "post_completion":
            post_completion,
    }


def main():
    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--agent",
        default="blue_agent_0",
    )

    parser.add_argument(
        "--out",
        default=(
            "outputs/cc4_probe/"
            "action_duration_"
            "decision_epoch.json"
        ),
    )

    args = parser.parse_args()

    result: dict[
        str,
        Any,
    ] = {
        "seed":
            int(
                args.seed
            ),

        "agent":
            str(
                args.agent
            ),

        "expected_durations":
            dict(
                EXPECTED_DURATIONS
            ),

        "families":
            {},
    }

    for family in (
        "Sleep",
        "Analyse",
        "Remove",
        "Restore",
    ):
        print()
        print(
            "=" * 72
        )

        print(
            f"[PROBE] {family}"
        )

        family_result = (
            probe_family(
                seed=int(
                    args.seed
                ),
                agent=str(
                    args.agent
                ),
                family=family,
            )
        )

        result[
            "families"
        ][
            family
        ] = family_result

        print(
            "label:",
            family_result[
                "selected_label"
            ],
        )

        print(
            "duration:",
            family_result[
                "action_duration"
            ],
        )

        print(
            "decision_dt:",
            family_result[
                "decision_dt"
            ],
        )

        for row in family_result[
            "ticks"
        ]:
            print(
                "tick=",
                row[
                    "local_tick"
                ],
                "submitted=",
                row[
                    "submitted_label"
                ],
                "after=",
                row[
                    "progress_after"
                ],
                "executed=",
                row[
                    "controller_executed"
                ],
                "decision_available=",
                row[
                    "decision_available_after"
                ],
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
        "[OK] duration probe saved:"
    )

    print(
        out_path
    )


if __name__ == "__main__":
    main()