from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List


THIS = Path(__file__).resolve()
# probe 文件位于：
# chapter2_region_detection/
#   formal_experiments/
#     probes/
#       probe_cc4_four_action_contract.py
#
# 因此 parents[2] 才是 chapter2_region_detection。
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


TARGET_FAMILIES = (
    "Sleep",
    "Analyse",
    "Remove",
    "Restore",
)


def _safe_repr(
    value: Any,
) -> str:
    try:
        return repr(
            value
        )
    except Exception:
        return (
            f"<unreprable "
            f"{type(value).__name__}>"
        )


def _json_safe(
    value: Any,
):
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

    if isinstance(
        value,
        dict,
    ):
        return {
            str(k): _json_safe(v)
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
            _json_safe(v)
            for v in value
        ]

    return _safe_repr(
        value
    )


def _strip_invalid_prefix(
    label: str,
) -> str:
    label = str(
        label
    ).strip()

    prefix = "[Invalid] "

    if label.startswith(
        prefix
    ):
        return label[
            len(prefix):
        ]

    return label


def _family_from_label(
    label: str,
) -> str:
    clean = (
        _strip_invalid_prefix(
            label
        )
    )

    for family in TARGET_FAMILIES:
        if family in clean:
            return family

    return "other"


def _call_optional(
    env,
    method_name: str,
    agent: str,
):
    method = getattr(
        env,
        method_name,
        None,
    )

    if method is None:
        return {
            "available": False,
            "value": None,
            "error": None,
        }

    try:
        value = method(
            agent
        )

        return {
            "available": True,
            "value": _json_safe(
                value
            ),
            "error": None,
        }

    except Exception as exc:
        return {
            "available": True,
            "value": None,
            "error": (
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
        }


def make_env(
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

    return env


def get_blue_agents(
    env,
) -> List[str]:
    spaces = (
        env.action_spaces()
    )

    agents = [
        str(name)
        for name in spaces.keys()
        if "blue_agent" in str(
            name
        )
    ]

    return sorted(
        agents
    )


def probe_agent(
    env,
    agent: str,
) -> Dict[str, Any]:

    labels = list(
        env.action_labels(
            agent
        )
    )

    mask = [
        bool(v)
        for v in list(
            env.action_mask(
                agent
            )
        )
    ]

    if len(labels) != len(mask):
        raise RuntimeError(
            f"{agent}: "
            f"labels={len(labels)} "
            f"mask={len(mask)}"
        )

    spaces = (
        env.action_spaces()
    )

    space = spaces[
        agent
    ]

    space_n = getattr(
        space,
        "n",
        None,
    )

    if (
        space_n is not None
        and int(space_n)
        != len(labels)
    ):
        raise RuntimeError(
            f"{agent}: "
            f"space.n={space_n} "
            f"labels={len(labels)}"
        )

    actions_result = (
        _call_optional(
            env,
            "actions",
            agent,
        )
    )

    hosts_result = (
        _call_optional(
            env,
            "hosts",
            agent,
        )
    )

    subnets_result = (
        _call_optional(
            env,
            "subnets",
            agent,
        )
    )

    raw_actions = None

    if (
        actions_result[
            "available"
        ]
        and actions_result[
            "error"
        ] is None
    ):
        try:
            raw_actions = list(
                env.actions(
                    agent
                )
            )
        except Exception:
            raw_actions = None

    rows = []

    family_counts = {
        family: {
            "total": 0,
            "valid": 0,
            "indices": [],
            "valid_indices": [],
        }
        for family in TARGET_FAMILIES
    }

    for idx, (
        label,
        valid,
    ) in enumerate(
        zip(
            labels,
            mask,
        )
    ):
        label_str = str(
            label
        )

        family = (
            _family_from_label(
                label_str
            )
        )

        action_repr = None
        action_class = None

        if (
            raw_actions is not None
            and idx < len(
                raw_actions
            )
        ):
            action_obj = (
                raw_actions[idx]
            )

            action_repr = _safe_repr(
                action_obj
            )

            action_class = (
                type(
                    action_obj
                ).__name__
            )

        row = {
            "index": int(
                idx
            ),
            "label": label_str,
            "clean_label":
                _strip_invalid_prefix(
                    label_str
                ),
            "family": family,
            "valid": bool(
                valid
            ),
            "action_class":
                action_class,
            "action_repr":
                action_repr,
        }

        rows.append(
            row
        )

        if family in family_counts:
            family_counts[
                family
            ][
                "total"
            ] += 1

            family_counts[
                family
            ][
                "indices"
            ].append(
                int(idx)
            )

            if valid:
                family_counts[
                    family
                ][
                    "valid"
                ] += 1

                family_counts[
                    family
                ][
                    "valid_indices"
                ].append(
                    int(idx)
                )

    interesting = [
        row
        for row in rows
        if row[
            "family"
        ] in TARGET_FAMILIES
    ]

    return {
        "agent": agent,
        "action_space_repr":
            _safe_repr(
                space
            ),
        "n_labels": len(
            labels
        ),
        "n_valid": int(
            sum(mask)
        ),
        "family_counts":
            family_counts,
        "hosts":
            hosts_result,
        "subnets":
            subnets_result,
        "actions_api":
            actions_result,
        "interesting_actions":
            interesting,
    }


def print_agent(
    result: Dict[str, Any],
):
    print()
    print(
        "=" * 88
    )

    print(
        f"[AGENT] "
        f"{result['agent']}"
    )

    print(
        f"action_space="
        f"{result['action_space_repr']}"
    )

    print(
        f"labels="
        f"{result['n_labels']} "
        f"valid="
        f"{result['n_valid']}"
    )

    print(
        "[FOUR ACTION FAMILIES]"
    )

    for family in TARGET_FAMILIES:
        info = result[
            "family_counts"
        ][
            family
        ]

        print(
            f"  {family:<10} "
            f"total={info['total']:<4} "
            f"valid={info['valid']:<4} "
            f"valid_idx="
            f"{info['valid_indices']}"
        )

    print(
        "[HOSTS API]"
    )

    print(
        "  available=",
        result[
            "hosts"
        ][
            "available"
        ],
    )

    print(
        "  value=",
        result[
            "hosts"
        ][
            "value"
        ],
    )

    print(
        "  error=",
        result[
            "hosts"
        ][
            "error"
        ],
    )

    print(
        "[INTERESTING ACTIONS]"
    )

    for row in result[
        "interesting_actions"
    ]:
        if not row[
            "valid"
        ]:
            continue

        print(
            f"  idx="
            f"{row['index']:<4} "
            f"family="
            f"{row['family']:<8} "
            f"class="
            f"{str(row['action_class']):<18} "
            f"label="
            f"{row['label']}"
        )


def run_probe(
    seed: int,
    steps: int,
    pad_spaces: bool,
):
    env = make_env(
        seed=seed,
        steps=steps,
        pad_spaces=pad_spaces,
    )

    reset_out = env.reset(
        seed=int(
            seed
        )
    )

    reset_type = (
        type(
            reset_out
        ).__name__
    )

    agents = get_blue_agents(
        env
    )

    print()
    print(
        "#" * 88
    )

    print(
        f"[RUN] "
        f"seed={seed} "
        f"pad_spaces={pad_spaces} "
        f"reset_type={reset_type}"
    )

    print(
        f"blue_agents="
        f"{agents}"
    )

    results = {}

    for agent in agents:
        agent_result = (
            probe_agent(
                env,
                agent,
            )
        )

        results[
            agent
        ] = agent_result

        print_agent(
            agent_result
        )

    return {
        "seed": int(
            seed
        ),
        "pad_spaces":
            bool(
                pad_spaces
            ),
        "blue_agents":
            agents,
        "reset_type":
            reset_type,
        "agents":
            results,
    }


def parse_seeds(
    value: str,
) -> List[int]:
    seeds = []

    for item in str(
        value
    ).split(
        ","
    ):
        item = item.strip()

        if not item:
            continue

        seeds.append(
            int(
                item
            )
        )

    if not seeds:
        raise ValueError(
            "at least one seed required"
        )

    return seeds


def main():
    parser = (
        argparse.ArgumentParser(
            description=(
                "Probe CC4 four-action "
                "BlueFixedActionWrapper contract."
            )
        )
    )

    parser.add_argument(
        "--seeds",
        default="42,43,44",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--out",
        default=(
            "outputs/cc4_probe/"
            "four_action_contract.json"
        ),
    )

    args = parser.parse_args()

    seeds = parse_seeds(
        args.seeds
    )

    print(
        "[CC4 FOUR-ACTION CONTRACT PROBE]"
    )

    print(
        "CYBORG_ROOT=",
        CYBORG_ROOT,
    )

    print(
        "seeds=",
        seeds,
    )

    output = {
        "cyborg_root":
            CYBORG_ROOT,
        "target_families":
            list(
                TARGET_FAMILIES
            ),
        "runs": [],
    }

    for pad_spaces in (
        False,
        True,
    ):
        for seed in seeds:
            result = run_probe(
                seed=seed,
                steps=args.steps,
                pad_spaces=pad_spaces,
            )

            output[
                "runs"
            ].append(
                result
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
        "[OK] probe saved:"
    )

    print(
        out_path
    )


if __name__ == "__main__":
    main()