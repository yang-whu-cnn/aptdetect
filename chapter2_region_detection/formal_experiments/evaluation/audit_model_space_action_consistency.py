from __future__ import annotations

import argparse
import json

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from formal_experiments.evaluation.validate_bootstrap_world_model import (
    chained,
    load_replay_jsonl,
    rmse,
    spearman,
    validate_frozen_split,
)

from formal_experiments.evaluation.validate_response_reward_predictor import (
    GAMMA_TICK,
    discounted_sum,
    pearson,
)

from formal_experiments.training.bootstrap_world_model import (
    BootstrapProbabilisticWorldModel,
    EXECUTED_FAMILY_TO_ACTION_ID,
)

from formal_experiments.training.response_reward_predictor import (
    ResponseRewardPredictor,
    build_response_reward_dataset,
)

from shared.action_contract import (
    ACTION_CONTRACTS,
    N_ACTIONS,
)

from shared.formal_state import (
    FORMAL_STATE_DIM,
    FORMAL_STATE_FEATURE_NAMES,
)


HORIZON = 4
TARGET_THRESHOLD = 0.5


ANY_TARGET_INDEX = (
    FORMAL_STATE_FEATURE_NAMES.index(
        "any_valid_observable_target"
    )
)


ACTION_DURATION = {
    int(
        item.action_id
    ):
        int(
            item.duration_ticks
        )

    for item
    in ACTION_CONTRACTS
}


@dataclass(frozen=True)
class RequestedRolloutWindow:
    episode_seed: int
    agent_name: str
    start_decision_index: int

    initial_state: np.ndarray

    requested_actions: np.ndarray
    actual_actions: np.ndarray

    actual_durations: np.ndarray
    rewards: np.ndarray

    target_state: np.ndarray


# ============================================================
# Model-space canonical action
# ============================================================

def canonicalize_requested_action(
    state,
    requested_action_id: int,
    *,
    threshold: float = TARGET_THRESHOLD,
) -> int:

    state = np.asarray(
        state,
        dtype=np.float32,
    )

    if state.shape != (
        FORMAL_STATE_DIM,
    ):
        raise ValueError(
            "state shape mismatch"
        )

    if not np.all(
        np.isfinite(
            state
        )
    ):
        raise ValueError(
            "state contains "
            "non-finite values"
        )

    requested_action_id = int(
        requested_action_id
    )

    if not (
        0
        <= requested_action_id
        < N_ACTIONS
    ):
        raise ValueError(
            "invalid requested action"
        )

    # no_op always Sleep
    if requested_action_id == 0:
        return 0

    available = (
        float(
            state[
                ANY_TARGET_INDEX
            ]
        )
        >= float(
            threshold
        )
    )

    return (
        requested_action_id
        if available
        else 0
    )


def canonicalize_requested_tensor(
    states: torch.Tensor,
    requested_actions: torch.Tensor,
    *,
    threshold: float = TARGET_THRESHOLD,
) -> torch.Tensor:

    if (
        states.ndim != 2
        or states.shape[
            1
        ]
        != FORMAL_STATE_DIM
    ):
        raise ValueError(
            "states must have "
            "shape (N,D)"
        )

    if (
        requested_actions.ndim != 1
        or requested_actions.shape[
            0
        ]
        != states.shape[
            0
        ]
    ):
        raise ValueError(
            "requested action "
            "shape mismatch"
        )

    requested_actions = (
        requested_actions.to(
            dtype=torch.long
        )
    )

    if requested_actions.numel():
        if (
            int(
                requested_actions.min()
            )
            < 0

            or int(
                requested_actions.max()
            )
            >= N_ACTIONS
        ):
            raise ValueError(
                "invalid requested action"
            )

    available = (
        states[
            :,
            ANY_TARGET_INDEX
        ]
        >= float(
            threshold
        )
    )

    targeted = (
        requested_actions
        != 0
    )

    return torch.where(
        targeted
        & (~available),

        torch.zeros_like(
            requested_actions
        ),

        requested_actions,
    )


# ============================================================
# Replay action helper
# ============================================================

def actual_executed_action_id(
    transition,
) -> int:

    family = str(
        transition
        .executed_action_family
    )

    if (
        family
        not in
        EXECUTED_FAMILY_TO_ACTION_ID
    ):
        raise ValueError(
            "unsupported family: "
            f"{family}"
        )

    return int(
        EXECUTED_FAMILY_TO_ACTION_ID[
            family
        ]
    )


# ============================================================
# Exact true-state audit
# ============================================================

def audit_true_state_mapping(
    transitions,
):
    total = 0
    correct = 0

    targeted_total = 0
    targeted_correct = 0

    feature_one_fallback = 0
    feature_zero_nonfallback = 0

    nonbinary_feature_count = 0

    per_requested = {
        str(
            action_id
        ): {
            "count": 0,
            "correct": 0,
            "fallback": 0,
        }

        for action_id
        in range(
            N_ACTIONS
        )
    }

    for item in transitions:

        requested = int(
            item.requested_action_id
        )

        actual = (
            actual_executed_action_id(
                item
            )
        )

        feature = float(
            item.state[
                ANY_TARGET_INDEX
            ]
        )

        if not (
            np.isclose(
                feature,
                0.0,
                atol=1e-6,
            )
            or
            np.isclose(
                feature,
                1.0,
                atol=1e-6,
            )
        ):
            nonbinary_feature_count += 1

        predicted = (
            canonicalize_requested_action(
                item.state,
                requested,
            )
        )

        # ----------------------------------------------------
        # Adapter semantics
        # ----------------------------------------------------

        if requested == 0:

            if (
                actual != 0
                or bool(
                    item.fallback
                )
            ):
                raise RuntimeError(
                    "requested no_op "
                    "violated Sleep contract"
                )

        elif item.fallback:

            if actual != 0:
                raise RuntimeError(
                    "fallback must "
                    "execute Sleep"
                )

            if (
                item.fallback_reason
                !=
                "no_valid_observable_target"
            ):
                raise RuntimeError(
                    "unexpected "
                    "fallback reason"
                )

        else:

            if actual != requested:
                raise RuntimeError(
                    "non-fallback targeted "
                    "action changed family"
                )

        # ----------------------------------------------------
        # Representation audit
        # ----------------------------------------------------

        total += 1

        per_requested[
            str(
                requested
            )
        ][
            "count"
        ] += 1

        if item.fallback:
            per_requested[
                str(
                    requested
                )
            ][
                "fallback"
            ] += 1

        if predicted == actual:
            correct += 1

            per_requested[
                str(
                    requested
                )
            ][
                "correct"
            ] += 1

        if requested != 0:

            targeted_total += 1

            if predicted == actual:
                targeted_correct += 1

            if (
                feature
                >= TARGET_THRESHOLD
                and item.fallback
            ):
                feature_one_fallback += 1

            if (
                feature
                < TARGET_THRESHOLD
                and not item.fallback
            ):
                feature_zero_nonfallback += 1

    if total == 0:
        raise ValueError(
            "empty transitions"
        )

    return {
        "count":
            int(
                total
            ),

        "accuracy":
            float(
                correct
                /
                total
            ),

        "targeted_count":
            int(
                targeted_total
            ),

        "targeted_accuracy":
            (
                1.0
                if targeted_total == 0
                else float(
                    targeted_correct
                    /
                    targeted_total
                )
            ),

        "feature_one_but_fallback":
            int(
                feature_one_fallback
            ),

        "feature_zero_but_nonfallback":
            int(
                feature_zero_nonfallback
            ),

        "nonbinary_feature_count":
            int(
                nonbinary_feature_count
            ),

        "per_requested_action":
            per_requested,
    }


# ============================================================
# H4 requested-plan windows
# ============================================================

def build_requested_windows(
    transitions,
    *,
    horizon: int = HORIZON,
):
    groups = {}

    for item in transitions:

        key = (
            int(
                item.episode_seed
            ),

            str(
                item.agent_name
            ),
        )

        groups.setdefault(
            key,
            [],
        ).append(
            item
        )

    windows = []

    for (
        seed,
        agent,
    ), sequence in sorted(
        groups.items()
    ):

        sequence.sort(
            key=lambda item:
                item.decision_index
        )

        for start in range(
            len(
                sequence
            )
            - horizon
            + 1
        ):

            block = sequence[
                start:
                start + horizon
            ]

            if not all(
                item.action_completed
                for item
                in block
            ):
                continue

            if not all(
                chained(
                    block[
                        index
                    ],

                    block[
                        index + 1
                    ],
                )

                for index
                in range(
                    horizon - 1
                )
            ):
                continue

            windows.append(
                RequestedRolloutWindow(
                    episode_seed=seed,

                    agent_name=agent,

                    start_decision_index=(
                        block[
                            0
                        ]
                        .decision_index
                    ),

                    initial_state=(
                        block[
                            0
                        ]
                        .state
                        .copy()
                    ),

                    requested_actions=np.asarray(
                        [
                            int(
                                item
                                .requested_action_id
                            )

                            for item
                            in block
                        ],

                        dtype=np.int64,
                    ),

                    actual_actions=np.asarray(
                        [
                            actual_executed_action_id(
                                item
                            )

                            for item
                            in block
                        ],

                        dtype=np.int64,
                    ),

                    actual_durations=np.asarray(
                        [
                            int(
                                item
                                .executed_duration_ticks
                            )

                            for item
                            in block
                        ],

                        dtype=np.int64,
                    ),

                    rewards=np.asarray(
                        [
                            float(
                                item
                                .response_reward
                            )

                            for item
                            in block
                        ],

                        dtype=np.float32,
                    ),

                    target_state=(
                        block[
                            -1
                        ]
                        .next_state
                        .copy()
                    ),
                )
            )

    if not windows:
        raise ValueError(
            "no requested-action "
            "H4 windows"
        )

    return tuple(
        windows
    )


# ============================================================
# Return helpers
# ============================================================

def true_returns(
    windows,
):
    return np.asarray(
        [
            discounted_sum(
                item.rewards,

                item.actual_durations,

                GAMMA_TICK,
            )

            for item
            in windows
        ],

        dtype=np.float64,
    )


def constant_return_baseline(
    windows,
    reward_mean: float,
):
    return np.asarray(
        [
            discounted_sum(
                np.full(
                    HORIZON,

                    float(
                        reward_mean
                    ),

                    dtype=np.float64,
                ),

                item.actual_durations,

                GAMMA_TICK,
            )

            for item
            in windows
        ],

        dtype=np.float64,
    )


# ============================================================
# Integrated requested-plan rollout
# ============================================================

def integrated_rollout_metrics(
    *,
    world_model,
    reward_predictor,
    windows,
    train_reward_mean: float,
):
    if (
        world_model.config
        .target_mode
        != "absolute"
    ):
        raise ValueError(
            "A4.6 requires selected "
            "absolute WM"
        )

    initial = np.stack(
        [
            item.initial_state
            for item
            in windows
        ],
        axis=0,
    ).astype(
        np.float32
    )

    requested = np.stack(
        [
            item.requested_actions
            for item
            in windows
        ],
        axis=0,
    ).astype(
        np.int64
    )

    actual = np.stack(
        [
            item.actual_actions
            for item
            in windows
        ],
        axis=0,
    ).astype(
        np.int64
    )

    target_state = np.stack(
        [
            item.target_state
            for item
            in windows
        ],
        axis=0,
    ).astype(
        np.float32
    )

    true_value = true_returns(
        windows
    )

    member_final_states = []
    member_returns = []

    targeted_matches = 0
    targeted_total = 0

    for member_index in range(
        len(
            world_model.models
        )
    ):
        current = torch.as_tensor(
            initial,

            dtype=torch.float32,

            device=(
                world_model.device
            ),
        )

        cumulative_return = (
            np.zeros(
                len(
                    windows
                ),

                dtype=np.float64,
            )
        )

        elapsed_ticks = (
            np.zeros(
                len(
                    windows
                ),

                dtype=np.int64,
            )
        )

        for step in range(
            HORIZON
        ):
            requested_t = (
                torch.as_tensor(
                    requested[
                        :,
                        step
                    ],

                    dtype=torch.long,

                    device=(
                        world_model.device
                    ),
                )
            )

            canonical_t = (
                canonicalize_requested_tensor(
                    current,
                    requested_t,
                )
            )

            canonical_np = (
                canonical_t
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.int64
                )
            )

            targeted_mask = (
                requested[
                    :,
                    step
                ]
                != 0
            )

            targeted_matches += int(
                np.sum(
                    canonical_np[
                        targeted_mask
                    ]
                    ==
                    actual[
                        targeted_mask,
                        step,
                    ]
                )
            )

            targeted_total += int(
                np.sum(
                    targeted_mask
                )
            )

            next_state = (
                world_model
                .predict_member_mean_tensor(
                    member_index,
                    current,
                    canonical_t,
                )
            )

            predicted_reward = (
                reward_predictor
                .predict_tensor(
                    current,
                    canonical_t,
                    next_state,
                )
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float64
                )
            )

            weights = (
                GAMMA_TICK
                **
                elapsed_ticks.astype(
                    np.float64
                )
            )

            cumulative_return += (
                weights
                *
                predicted_reward
            )

            predicted_durations = (
                np.asarray(
                    [
                        ACTION_DURATION[
                            int(
                                action_id
                            )
                        ]

                        for action_id
                        in canonical_np
                    ],

                    dtype=np.int64,
                )
            )

            elapsed_ticks += (
                predicted_durations
            )

            current = (
                next_state
            )

        member_final_states.append(
            current
            .detach()
            .cpu()
            .numpy()
        )

        member_returns.append(
            cumulative_return
        )

    member_final_states = np.stack(
        member_final_states,
        axis=0,
    )

    member_returns = np.stack(
        member_returns,
        axis=0,
    )

    predicted_state = np.mean(
        member_final_states,
        axis=0,
    )

    predicted_value = np.mean(
        member_returns,
        axis=0,
    )

    return_uncertainty = np.std(
        member_returns,
        axis=0,
        ddof=0,
    )

    persistence_rmse = rmse(
        initial,
        target_state,
    )

    baseline = (
        constant_return_baseline(
            windows,
            train_reward_mean,
        )
    )

    seeds = np.asarray(
        [
            item.episode_seed
            for item
            in windows
        ],

        dtype=np.int64,
    )

    per_episode = {}

    positive_count = 0

    for seed in sorted(
        set(
            seeds.tolist()
        )
    ):
        mask = (
            seeds
            == seed
        )

        rho = spearman(
            predicted_value[
                mask
            ],

            true_value[
                mask
            ],
        )

        if rho > 0.0:
            positive_count += 1

        per_episode[
            str(
                seed
            )
        ] = {
            "count":
                int(
                    mask.sum()
                ),

            "state_rmse":
                rmse(
                    predicted_state[
                        mask
                    ],

                    target_state[
                        mask
                    ],
                ),

            "value_rmse":
                rmse(
                    predicted_value[
                        mask
                    ],

                    true_value[
                        mask
                    ],
                ),

            "value_spearman":
                float(
                    rho
                ),
        }

    return {
        "window_count":
            len(
                windows
            ),

        "h4_state_rmse":
            rmse(
                predicted_state,
                target_state,
            ),

        "persistence_state_rmse":
            persistence_rmse,

        "h4_value_rmse":
            rmse(
                predicted_value,
                true_value,
            ),

        "constant_value_baseline_rmse":
            rmse(
                baseline,
                true_value,
            ),

        "h4_value_pearson":
            pearson(
                predicted_value,
                true_value,
            ),

        "h4_value_spearman":
            spearman(
                predicted_value,
                true_value,
            ),

        "positive_episode_value_spearman_count":
            int(
                positive_count
            ),

        "targeted_member_step_count":
            int(
                targeted_total
            ),

        "targeted_member_step_match_rate":
            (
                1.0
                if targeted_total == 0
                else float(
                    targeted_matches
                    /
                    targeted_total
                )
            ),

        "return_uncertainty_true_error_spearman":
            spearman(
                return_uncertainty,

                np.abs(
                    predicted_value
                    -
                    true_value
                ),
            ),

        "per_episode":
            per_episode,
    }


# ============================================================
# Hard Gate
# ============================================================

def quality_gate(
    *,
    train_mapping,
    validation_mapping,
    integrated,
):
    result = {
        "train_true_state_mapping_exact":
            bool(
                train_mapping[
                    "accuracy"
                ]
                == 1.0
            ),

        "validation_true_state_mapping_exact":
            bool(
                validation_mapping[
                    "accuracy"
                ]
                == 1.0
            ),

        "no_train_feature_one_fallback":
            (
                train_mapping[
                    "feature_one_but_fallback"
                ]
                == 0
            ),

        "no_validation_feature_one_fallback":
            (
                validation_mapping[
                    "feature_one_but_fallback"
                ]
                == 0
            ),

        "no_train_feature_zero_nonfallback":
            (
                train_mapping[
                    "feature_zero_but_nonfallback"
                ]
                == 0
            ),

        "no_validation_feature_zero_nonfallback":
            (
                validation_mapping[
                    "feature_zero_but_nonfallback"
                ]
                == 0
            ),

        "h4_state_beats_persistence":
            (
                integrated[
                    "h4_state_rmse"
                ]
                <
                integrated[
                    "persistence_state_rmse"
                ]
            ),

        "h4_value_beats_constant":
            (
                integrated[
                    "h4_value_rmse"
                ]
                <
                integrated[
                    "constant_value_baseline_rmse"
                ]
            ),

        "h4_value_spearman_gt_0_3":
            (
                integrated[
                    "h4_value_spearman"
                ]
                > 0.3
            ),

        "positive_episode_value_spearman_count":
            int(
                integrated[
                    "positive_episode_value_spearman_count"
                ]
            ),
    }

    result[
        "stable_episode_value_relation"
    ] = (
        result[
            "positive_episode_value_spearman_count"
        ]
        >= 5
    )

    result[
        "pass"
    ] = all(
        [
            result[
                "train_true_state_mapping_exact"
            ],

            result[
                "validation_true_state_mapping_exact"
            ],

            result[
                "no_train_feature_one_fallback"
            ],

            result[
                "no_validation_feature_one_fallback"
            ],

            result[
                "no_train_feature_zero_nonfallback"
            ],

            result[
                "no_validation_feature_zero_nonfallback"
            ],

            result[
                "h4_state_beats_persistence"
            ],

            result[
                "h4_value_beats_constant"
            ],

            result[
                "h4_value_spearman_gt_0_3"
            ],

            result[
                "stable_episode_value_relation"
            ],
        ]
    )

    return result


# ============================================================
# Path
# ============================================================

def resolve_path(
    value: str,
) -> Path:

    path = Path(
        value
    )

    if path.is_absolute():
        return path

    return (
        Path.cwd()
        /
        path
    ).resolve()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "A4.6a model-space "
            "requested/executed "
            "action audit"
        )
    )

    parser.add_argument(
        "--train-replay",

        default=(
            "outputs/formal_replay/"
            "train.jsonl"
        ),
    )

    parser.add_argument(
        "--validation-replay",

        default=(
            "outputs/formal_replay/"
            "validation.jsonl"
        ),
    )

    parser.add_argument(
        "--world-model",

        default=(
            "outputs/world_model/"
            "a4_5b/"
            "world_model_absolute.pt"
        ),
    )

    parser.add_argument(
        "--reward-model",

        default=(
            "outputs/world_model/"
            "a4_5c/"
            "response_reward_predictor.pt"
        ),
    )

    parser.add_argument(
        "--out",

        default=(
            "outputs/world_model/"
            "a4_6a/"
            "action_consistency_report.json"
        ),
    )

    parser.add_argument(
        "--device",
        default="cpu",
    )

    args = (
        parser.parse_args()
    )

    train = load_replay_jsonl(
        resolve_path(
            args.train_replay
        )
    )

    validation = (
        load_replay_jsonl(
            resolve_path(
                args.validation_replay
            )
        )
    )

    validate_frozen_split(
        train,
        "train",
    )

    validate_frozen_split(
        validation,
        "validation",
    )

    train_mapping = (
        audit_true_state_mapping(
            train
        )
    )

    validation_mapping = (
        audit_true_state_mapping(
            validation
        )
    )

    world_model = (
        BootstrapProbabilisticWorldModel
        .load_checkpoint(
            resolve_path(
                args.world_model
            ),

            device=args.device,
        )
    )

    reward_predictor = (
        ResponseRewardPredictor
        .load_checkpoint(
            resolve_path(
                args.reward_model
            ),

            device=args.device,
        )
    )

    windows = (
        build_requested_windows(
            validation,

            horizon=HORIZON,
        )
    )

    train_reward_dataset = (
        build_response_reward_dataset(
            train
        )
    )

    train_reward_mean = float(
        np.mean(
            train_reward_dataset
            .rewards
        )
    )

    integrated = (
        integrated_rollout_metrics(
            world_model=(
                world_model
            ),

            reward_predictor=(
                reward_predictor
            ),

            windows=windows,

            train_reward_mean=(
                train_reward_mean
            ),
        )
    )

    gate = quality_gate(
        train_mapping=(
            train_mapping
        ),

        validation_mapping=(
            validation_mapping
        ),

        integrated=(
            integrated
        ),
    )

    report = {
        "contract": {
            "state_dim":
                FORMAL_STATE_DIM,

            "n_actions":
                N_ACTIONS,

            "any_valid_observable_target_index":
                ANY_TARGET_INDEX,

            "target_threshold":
                TARGET_THRESHOLD,

            "horizon":
                HORIZON,

            "gamma_tick":
                GAMMA_TICK,

            "canonical_rule":
                (
                    "no_op->Sleep; "
                    "targeted + no valid "
                    "observable target -> Sleep; "
                    "otherwise keep requested"
                ),

            "hidden_inputs":
                False,

            "calibration_test_seeds_used":
                False,
        },

        "train_true_state_mapping":
            train_mapping,

        "validation_true_state_mapping":
            validation_mapping,

        "integrated_requested_plan_h4":
            integrated,

        "quality_gate":
            gate,
    }

    out_path = resolve_path(
        args.out
    )

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),

        encoding="utf-8",
    )

    print()
    print(
        "=" * 80
    )

    print(
        "[A4.6a SUMMARY]"
    )

    print(
        "train mapping accuracy:",
        train_mapping[
            "accuracy"
        ],
    )

    print(
        "validation mapping accuracy:",
        validation_mapping[
            "accuracy"
        ],
    )

    print(
        "train feature=1 fallback:",
        train_mapping[
            "feature_one_but_fallback"
        ],
    )

    print(
        "validation feature=1 fallback:",
        validation_mapping[
            "feature_one_but_fallback"
        ],
    )

    print(
        "train feature=0 nonfallback:",
        train_mapping[
            "feature_zero_but_nonfallback"
        ],
    )

    print(
        "validation feature=0 nonfallback:",
        validation_mapping[
            "feature_zero_but_nonfallback"
        ],
    )

    print(
        "integrated H4 state RMSE:",
        integrated[
            "h4_state_rmse"
        ],
    )

    print(
        "persistence state RMSE:",
        integrated[
            "persistence_state_rmse"
        ],
    )

    print(
        "integrated H4 value RMSE:",
        integrated[
            "h4_value_rmse"
        ],
    )

    print(
        "constant value baseline RMSE:",
        integrated[
            "constant_value_baseline_rmse"
        ],
    )

    print(
        "integrated H4 value Spearman:",
        integrated[
            "h4_value_spearman"
        ],
    )

    print(
        "positive episode value Spearman:",
        integrated[
            "positive_episode_value_spearman_count"
        ],
    )

    print(
        "targeted member-step match rate:",
        integrated[
            "targeted_member_step_match_rate"
        ],
    )

    print(
        "quality_gate:",
        gate,
    )

    print(
        "[OK] report:",
        out_path,
    )

    if not gate[
        "pass"
    ]:
        print(
            "[A4.6a GATE NOT PASSED] "
            "Do not freeze Step-4 "
            "model-space action adapter."
        )


if __name__ == "__main__":
    main()