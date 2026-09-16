
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


BLUE_AGENTS = [
    "blue_agent_0",
    "blue_agent_1",
    "blue_agent_2",
    "blue_agent_3",
    "blue_agent_4",
]

META_KEYS = {
    "success",
    "action",
    "message",
}


def json_safe(value: Any):
    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return value

    if isinstance(value, dict):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):
        return [
            json_safe(v)
            for v in value
        ]

    return repr(value)


def make_env(
    seed: int,
    steps: int,
):
    """
    A4.1 probe 使用：

    Blue  = SleepAgent
    Green = EnterpriseGreenAgent
    Red   = SleepAgent

    这样避免真实 Red 攻击污染 observation contract，
    只利用 Green false positive 制造真实 Monitor alert。
    """

    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=SleepAgent,
        steps=int(steps),
    )

    cyborg = CybORG(
        scenario_generator=sg,
        seed=int(seed),
    )

    env = BlueFixedActionWrapper(
        cyborg,
        pad_spaces=False,
    )

    observations, info = env.reset(
        seed=int(seed)
    )

    return (
        env,
        observations,
        info,
    )


def force_green_false_positive_rate(
    env,
):
    """
    PROBE ONLY。

    这里只修改 Green agent 的公开仿真参数，
    使真实 Green 行为产生 Monitor false-positive
    的概率变为 1。

    FormalStateEncoder 不允许访问 controller。
    """

    controller = (
        env.env
        .environment_controller
    )

    changed = []

    for (
        agent_name,
        interface,
    ) in controller.agent_interfaces.items():

        agent = getattr(
            interface,
            "agent",
            None,
        )

        if (
            "green_agent" in agent_name
            and isinstance(
                agent,
                EnterpriseGreenAgent,
            )
        ):
            agent.fp_detection_rate = 1.0
            changed.append(agent_name)

    if not changed:
        raise RuntimeError(
            "no EnterpriseGreenAgent found"
        )

    return sorted(changed)


def observation_hosts(
    observation: dict,
) -> dict[str, dict]:

    result = {}

    for key, value in observation.items():
        if str(key) in META_KEYS:
            continue

        if not isinstance(value, dict):
            continue

        result[str(key)] = value

    return result


def build_inventory_from_reset(
    observation: dict,
):
    """
    只从 Blue RESET observation 建立：

    hostname
    IP -> hostname

    不读取 controller true state。
    """

    hosts = observation_hosts(
        observation
    )

    ip_to_hostname = {}
    hostnames = []

    for key, data in hosts.items():

        system_info = data.get(
            "System info",
            {},
        )

        hostname = str(
            system_info.get(
                "Hostname",
                key,
            )
        )

        hostnames.append(
            hostname
        )

        for interface in data.get(
            "Interface",
            [],
        ):
            ip = interface.get(
                "ip_address"
            )

            if ip is not None:
                ip_to_hostname[
                    str(ip)
                ] = hostname

    return {
        "hostnames":
            sorted(set(hostnames)),

        "ip_to_hostname":
            ip_to_hostname,
    }


def canonical_hostname(
    key: str,
    inventory: dict,
) -> str:

    text = str(key)

    if text in inventory[
        "hostnames"
    ]:
        return text

    return inventory[
        "ip_to_hostname"
    ].get(
        text,
        text,
    )


def host_evidence_summary(
    data: dict,
):
    processes = data.get(
        "Processes",
        [],
    )

    files = data.get(
        "Files",
        [],
    )

    process_count = (
        len(processes)
        if isinstance(processes, list)
        else 0
    )

    file_count = (
        len(files)
        if isinstance(files, list)
        else 0
    )

    connection_count = 0

    if isinstance(
        processes,
        list,
    ):
        for process in processes:

            if not isinstance(
                process,
                dict,
            ):
                continue

            connections = (
                process.get(
                    "Connections",
                    [],
                )
            )

            if isinstance(
                connections,
                list,
            ):
                connection_count += len(
                    connections
                )

    return {
        "keys":
            sorted(
                str(k)
                for k in data.keys()
            ),

        "process_count":
            int(process_count),

        "connection_count":
            int(connection_count),

        "file_count":
            int(file_count),

        "has_process_evidence":
            bool(
                process_count > 0
            ),

        "has_connection_evidence":
            bool(
                connection_count > 0
            ),

        "has_file_evidence":
            bool(
                file_count > 0
            ),
    }


def summarize_observation(
    observation: dict,
    inventory: dict,
):
    hosts = observation_hosts(
        observation
    )

    host_result = {}

    evidence_hosts = []

    for raw_key, data in hosts.items():

        hostname = canonical_hostname(
            raw_key,
            inventory,
        )

        evidence = (
            host_evidence_summary(
                data
            )
        )

        host_result[
            hostname
        ] = evidence

        if (
            evidence[
                "process_count"
            ]
            +
            evidence[
                "connection_count"
            ]
            +
            evidence[
                "file_count"
            ]
            > 0
        ):
            evidence_hosts.append(
                hostname
            )

    return {
        "top_level_keys":
            sorted(
                str(k)
                for k
                in observation.keys()
            ),

        "success":
            repr(
                observation.get(
                    "success"
                )
            ),

        "action":
            repr(
                observation.get(
                    "action"
                )
            ),

        "message_present":
            (
                "message"
                in observation
            ),

        "hosts":
            host_result,

        "evidence_hosts":
            sorted(
                set(
                    evidence_hosts
                )
            ),
    }


def sleep_actions(
    env,
):
    adapter = (
        CybORGActionAdapter()
    )

    result = {}

    for agent in BLUE_AGENTS:

        resolution = adapter.resolve(
            env=env,
            agent_name=agent,
            action_id=0,
            observable_host_scores=None,
        )

        if resolution.fallback:
            raise RuntimeError(
                f"{agent}: Sleep fallback "
                "must never occur"
            )

        if (
            resolution
            .executed_action_family
            != "Sleep"
        ):
            raise RuntimeError(
                f"{agent}: expected Sleep"
            )

        result[
            agent
        ] = int(
            resolution.executed_index
        )

    return result


def analyse_actions(
    env,
    target_hosts: dict[str, str],
):
    adapter = (
        CybORGActionAdapter()
    )

    result = {}
    resolutions = {}

    for agent in BLUE_AGENTS:

        host = target_hosts[
            agent
        ]

        resolution = adapter.resolve(
            env=env,
            agent_name=agent,
            action_id=1,
            observable_host_scores={
                host: 1.0
            },
        )

        if resolution.fallback:
            raise RuntimeError(
                f"{agent}: Analyse "
                f"fallback for {host}"
            )

        if (
            resolution.target_host
            != host
        ):
            raise RuntimeError(
                f"{agent}: Analyse target "
                f"{resolution.target_host} "
                f"!= {host}"
            )

        result[
            agent
        ] = int(
            resolution.executed_index
        )

        resolutions[
            agent
        ] = {
            "executed_index":
                int(
                    resolution
                    .executed_index
                ),

            "executed_label":
                str(
                    resolution
                    .executed_label
                ),

            "target_host":
                str(
                    resolution
                    .target_host
                ),
        }

    return (
        result,
        resolutions,
    )


def run_seed(
    *,
    seed: int,
    steps: int,
    max_alert_ticks: int,
):
    (
        env,
        reset_obs,
        _reset_info,
    ) = make_env(
        seed=seed,
        steps=steps,
    )

    inventories = {
        agent:
            build_inventory_from_reset(
                reset_obs[
                    agent
                ]
            )
        for agent in BLUE_AGENTS
    }

    reset_summary = {
        agent:
            summarize_observation(
                reset_obs[
                    agent
                ],
                inventories[
                    agent
                ],
            )
        for agent in BLUE_AGENTS
    }

    for agent in BLUE_AGENTS:
        if not inventories[
            agent
        ][
            "hostnames"
        ]:
            raise RuntimeError(
                f"{agent}: empty "
                "observable reset inventory"
            )

    changed_green_agents = (
        force_green_false_positive_rate(
            env
        )
    )

    first_alert_tick = {
        agent: None
        for agent in BLUE_AGENTS
    }

    first_alert_host = {
        agent: None
        for agent in BLUE_AGENTS
    }

    first_alert_observation = {
        agent: None
        for agent in BLUE_AGENTS
    }

    alert_timeline = []

    # =========================================================
    # Real Monitor observations
    # =========================================================

    for tick in range(
        1,
        max_alert_ticks + 1,
    ):
        (
            obs,
            _rewards,
            _terminated,
            _truncated,
            _info,
        ) = env.step(
            actions=sleep_actions(
                env
            )
        )

        tick_summary = {}

        for agent in BLUE_AGENTS:

            summary = (
                summarize_observation(
                    obs[
                        agent
                    ],
                    inventories[
                        agent
                    ],
                )
            )

            tick_summary[
                agent
            ] = summary

            if (
                first_alert_tick[
                    agent
                ]
                is None
                and summary[
                    "evidence_hosts"
                ]
            ):
                host = (
                    summary[
                        "evidence_hosts"
                    ][0]
                )

                first_alert_tick[
                    agent
                ] = int(
                    tick
                )

                first_alert_host[
                    agent
                ] = host

                first_alert_observation[
                    agent
                ] = json_safe(
                    obs[
                        agent
                    ]
                )

        alert_timeline.append(
            {
                "tick":
                    int(tick),

                "agents":
                    tick_summary,
            }
        )

        if all(
            first_alert_tick[
                agent
            ]
            is not None
            for agent
            in BLUE_AGENTS
        ):
            break

    missing = [
        agent
        for agent in BLUE_AGENTS
        if first_alert_tick[
            agent
        ]
        is None
    ]

    if missing:
        raise RuntimeError(
            "failed to obtain real "
            "Monitor evidence for: "
            f"{missing}"
        )

    # =========================================================
    # Analyse the observable alert host.
    # No true-state host score is used.
    # =========================================================

    target_hosts = {
        agent: str(
            first_alert_host[
                agent
            ]
        )
        for agent in BLUE_AGENTS
    }

    (
        analyse_joint_actions,
        analyse_resolutions,
    ) = analyse_actions(
        env,
        target_hosts,
    )

    analyse_ticks = []

    for local_tick in (
        1,
        2,
    ):
        if local_tick == 1:
            actions = (
                analyse_joint_actions
            )
        else:
            actions = {}

        (
            obs,
            _rewards,
            _terminated,
            _truncated,
            _info,
        ) = env.step(
            actions=actions
        )

        analyse_ticks.append(
            {
                "local_tick":
                    local_tick,

                "agents": {
                    agent: {
                        "summary":
                            summarize_observation(
                                obs[
                                    agent
                                ],
                                inventories[
                                    agent
                                ],
                            ),

                        "raw":
                            json_safe(
                                obs[
                                    agent
                                ]
                            ),
                    }
                    for agent
                    in BLUE_AGENTS
                },
            }
        )

    return {
        "seed":
            int(seed),

        "green_agents_forced_fp_1":
            changed_green_agents,

        "reset": {
            agent: {
                "inventory":
                    inventories[
                        agent
                    ],

                "summary":
                    reset_summary[
                        agent
                    ],
            }
            for agent
            in BLUE_AGENTS
        },

        "monitor_probe": {
            "first_alert_tick":
                first_alert_tick,

            "first_alert_host":
                first_alert_host,

            "first_alert_raw":
                first_alert_observation,

            "timeline":
                alert_timeline,
        },

        "analyse_probe": {
            "targets":
                target_hosts,

            "resolutions":
                analyse_resolutions,

            "ticks":
                analyse_ticks,
        },
    }


def parse_seeds(
    value: str,
):
    seeds = []

    for item in str(
        value
    ).split(","):

        item = item.strip()

        if item:
            seeds.append(
                int(item)
            )

    if not seeds:
        raise ValueError(
            "at least one seed required"
        )

    return seeds


def main():
    parser = argparse.ArgumentParser(
        description=(
            "A4.1 probe of the raw "
            "observable CC4 Blue "
            "observation contract."
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
        "--max-alert-ticks",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--out",
        default=(
            "outputs/cc4_probe/"
            "observable_state_contract.json"
        ),
    )

    args = parser.parse_args()

    seeds = parse_seeds(
        args.seeds
    )

    output = {
        "contract": {
            "planner_observation_source":
                "BlueFixedActionWrapper raw Blue observation",

            "reset_inventory_source":
                "reset observation only",

            "host_evidence_source": [
                "Processes",
                "Processes[*].Connections",
                "Files",
            ],

            "forbidden_planner_sources": [
                "controller true state",
                "red sessions",
                "true compromise labels",
                "future information",
                "A3 synthetic host scores",
            ],

            "green_fp_rate_override":
                (
                    "probe-only mechanism "
                    "to generate real "
                    "Monitor observations"
                ),
        },

        "runs": [],
    }

    for seed in seeds:
        print()
        print(
            "=" * 80
        )

        print(
            "[A4.1 RUN]",
            f"seed={seed}",
        )

        run = run_seed(
            seed=seed,
            steps=int(
                args.steps
            ),
            max_alert_ticks=int(
                args.max_alert_ticks
            ),
        )

        output[
            "runs"
        ].append(
            run
        )

        for agent in BLUE_AGENTS:

            inventory_count = len(
                run[
                    "reset"
                ][
                    agent
                ][
                    "inventory"
                ][
                    "hostnames"
                ]
            )

            alert_tick = (
                run[
                    "monitor_probe"
                ][
                    "first_alert_tick"
                ][
                    agent
                ]
            )

            alert_host = (
                run[
                    "monitor_probe"
                ][
                    "first_alert_host"
                ][
                    agent
                ]
            )

            print(
                agent,
                "inventory=",
                inventory_count,
                "first_alert_tick=",
                alert_tick,
                "first_alert_host=",
                alert_host,
            )

        print(
            "Analyse targets:",
            run[
                "analyse_probe"
            ][
                "targets"
            ],
        )

    output[
        "summary"
    ] = {
        "runs":
            len(
                output[
                    "runs"
                ]
            ),

        "agents_per_run":
            len(
                BLUE_AGENTS
            ),

        "all_agents_observed_monitor_evidence":
            True,

        "analyse_used_only_observable_targets":
            True,

        "hidden_truth_used_for_planner":
            False,
    }

    print()
    print(
        "=" * 80
    )

    print(
        "[A4.1 SUMMARY]"
    )

    print(
        "runs:",
        len(
            output[
                "runs"
            ]
        ),
    )

    print(
        "agents_per_run:",
        len(
            BLUE_AGENTS
        ),
    )

    print(
        "all_agents_observed_monitor_evidence: True"
    )

    print(
        "analyse_used_only_observable_targets: True"
    )

    print(
        "hidden_truth_used_for_planner: False"
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
        "[OK] observable-state "
        "contract probe saved:"
    )

    print(
        out_path
    )


if __name__ == "__main__":
    main()