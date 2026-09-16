from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from math import pi
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from formal_experiments.data_collection.collect_cc4_formal_replay import (
    DEFAULT_SPLIT_SEEDS,
)
from formal_experiments.data_collection.decision_replay import (
    DecisionEpochTransition,
)
from formal_experiments.training.bootstrap_world_model import (
    BootstrapProbabilisticWorldModel,
    BootstrapWorldModelConfig,
    EXECUTED_FAMILY_TO_ACTION_ID,
    build_world_model_dataset,
)
from shared.formal_state import (
    FORMAL_STATE_DIM,
    FORMAL_STATE_FEATURE_NAMES,
)


TARGET_MODES = (
    "absolute",
    "delta",
)

ROLLOUT_HORIZONS = (
    2,
    4,
)


@dataclass(frozen=True)
class RolloutWindow:
    episode_seed: int
    agent_name: str
    start_decision_index: int
    initial_state: np.ndarray
    actions: np.ndarray
    target_state: np.ndarray


def transition_from_json(
    data: dict[str, Any],
) -> DecisionEpochTransition:
    data = dict(data)

    data["state"] = np.asarray(
        data["state"],
        dtype=np.float32,
    )

    data["next_state"] = np.asarray(
        data["next_state"],
        dtype=np.float32,
    )

    data["incident_event_ids"] = tuple(
        data.get(
            "incident_event_ids",
            (),
        )
    )

    data["incident_host_ids"] = tuple(
        data.get(
            "incident_host_ids",
            (),
        )
    )

    for name in (
        "state",
        "next_state",
    ):
        value = data[name]

        if (
            value.shape
            != (FORMAL_STATE_DIM,)
            or not np.all(
                np.isfinite(value)
            )
        ):
            raise ValueError(
                f"{name} must be finite "
                f"shape=({FORMAL_STATE_DIM},)"
            )

    return DecisionEpochTransition(
        **data
    )


def load_replay_jsonl(
    path: str | Path,
) -> tuple[
    DecisionEpochTransition,
    ...,
]:
    path = Path(path)

    items = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        for line_no, raw in enumerate(
            handle,
            start=1,
        ):
            raw = raw.strip()

            if not raw:
                continue

            try:
                items.append(
                    transition_from_json(
                        json.loads(raw)
                    )
                )

            except Exception as exc:
                raise ValueError(
                    f"{path}: invalid "
                    f"transition at "
                    f"line {line_no}"
                ) from exc

    if not items:
        raise ValueError(
            f"{path}: empty replay"
        )

    return tuple(items)


def validate_frozen_split(
    transitions: Sequence[
        DecisionEpochTransition
    ],
    split: str,
) -> None:
    if split not in (
        "train",
        "validation",
    ):
        raise ValueError(
            "split must be "
            "train or validation"
        )

    actual = {
        int(
            item.episode_seed
        )
        for item
        in transitions
    }

    expected = set(
        DEFAULT_SPLIT_SEEDS[
            split
        ]
    )

    if actual != expected:
        raise ValueError(
            f"{split} seed mismatch: "
            f"actual={sorted(actual)}, "
            f"expected={sorted(expected)}"
        )


def complete(
    transitions: Sequence[
        DecisionEpochTransition
    ],
) -> tuple[
    DecisionEpochTransition,
    ...,
]:
    result = tuple(
        item
        for item
        in transitions
        if item.action_completed
    )

    if not result:
        raise ValueError(
            "no completed transitions"
        )

    return result


def executed_action_id(
    transition: DecisionEpochTransition,
) -> int:
    try:
        return int(
            EXECUTED_FAMILY_TO_ACTION_ID[
                transition
                .executed_action_family
            ]
        )

    except KeyError as exc:
        raise ValueError(
            "unsupported family: "
            f"{transition.executed_action_family}"
        ) from exc


def chained(
    left: DecisionEpochTransition,
    right: DecisionEpochTransition,
) -> bool:
    return bool(
        left.episode_seed
        == right.episode_seed

        and left.agent_name
        == right.agent_name

        and right.decision_index
        == left.decision_index + 1

        and right.global_tick_start
        == left.global_tick_end

        and np.allclose(
            left.next_state,
            right.state,
            rtol=0.0,
            atol=1e-6,
        )
    )


def build_rollout_windows(
    transitions: Sequence[
        DecisionEpochTransition
    ],
    horizon: int,
) -> tuple[
    RolloutWindow,
    ...,
]:
    if horizon <= 0:
        raise ValueError(
            "horizon must be > 0"
        )

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
            len(sequence)
            - horizon
            + 1
        ):
            block = sequence[
                start:
                start + horizon
            ]

            if not all(
                item.action_completed
                for item in block
            ):
                continue

            if not all(
                chained(
                    block[index],
                    block[index + 1],
                )
                for index
                in range(
                    horizon - 1
                )
            ):
                continue

            windows.append(
                RolloutWindow(
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
                    actions=np.asarray(
                        [
                            executed_action_id(
                                item
                            )
                            for item
                            in block
                        ],
                        dtype=np.int64,
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
            f"no H={horizon} "
            "rollout windows"
        )

    return tuple(windows)


def rmse(
    predicted,
    target,
) -> float:
    predicted = np.asarray(
        predicted,
        dtype=np.float64,
    )

    target = np.asarray(
        target,
        dtype=np.float64,
    )

    return float(
        np.sqrt(
            np.mean(
                (
                    predicted
                    - target
                )
                ** 2
            )
        )
    )


def mae(
    predicted,
    target,
) -> float:
    predicted = np.asarray(
        predicted,
        dtype=np.float64,
    )

    target = np.asarray(
        target,
        dtype=np.float64,
    )

    return float(
        np.mean(
            np.abs(
                predicted
                - target
            )
        )
    )


def sample_rmse(
    predicted,
    target,
) -> np.ndarray:
    predicted = np.asarray(
        predicted,
        dtype=np.float64,
    )

    target = np.asarray(
        target,
        dtype=np.float64,
    )

    return np.sqrt(
        np.mean(
            (
                predicted
                - target
            )
            ** 2,
            axis=1,
        )
    )


def mixture_nll_per_dim(
    target,
    member_means,
    member_variances,
    eps: float = 1e-8,
) -> float:
    target = np.asarray(
        target,
        dtype=np.float64,
    )

    means = np.asarray(
        member_means,
        dtype=np.float64,
    )

    variances = np.maximum(
        np.asarray(
            member_variances,
            dtype=np.float64,
        ),
        eps,
    )

    if (
        means.shape
        != variances.shape
        or means.ndim != 3
        or target.shape
        != means.shape[1:]
    ):
        raise ValueError(
            "mixture shapes must be "
            "target[N,D], "
            "means/vars[M,N,D]"
        )

    residual = (
        target[
            None,
            :,
            :,
        ]
        - means
    )

    log_probability = (
        -0.5
        * np.sum(
            np.log(
                2.0
                * pi
                * variances
            )
            +
            residual ** 2
            / variances,
            axis=2,
        )
    )

    maximum = np.max(
        log_probability,
        axis=0,
        keepdims=True,
    )

    log_mixture = (
        maximum[
            0
        ]
        +
        np.log(
            np.mean(
                np.exp(
                    log_probability
                    - maximum
                ),
                axis=0,
            )
        )
    )

    return float(
        -np.mean(
            log_mixture
        )
        /
        target.shape[
            1
        ]
    )


def average_ranks(
    values,
) -> np.ndarray:
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    order = np.argsort(
        values,
        kind="mergesort",
    )

    ranks = np.empty(
        values.size,
        dtype=np.float64,
    )

    start = 0

    while start < values.size:
        end = start + 1

        while (
            end < values.size
            and values[
                order[end]
            ]
            ==
            values[
                order[start]
            ]
        ):
            end += 1

        ranks[
            order[
                start:end
            ]
        ] = (
            start
            + 1
            + end
        ) / 2.0

        start = end

    return ranks


def spearman(
    x,
    y,
) -> float:
    rank_x = average_ranks(
        x
    )

    rank_y = average_ranks(
        y
    )

    if (
        rank_x.size < 2
        or np.std(
            rank_x
        )
        == 0.0
        or np.std(
            rank_y
        )
        == 0.0
    ):
        return 0.0

    return float(
        np.corrcoef(
            rank_x,
            rank_y,
        )[0, 1]
    )


def binary_auc(
    labels,
    scores,
) -> float | None:
    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    positive = (
        labels == 1
    )

    n_positive = int(
        positive.sum()
    )

    n_negative = int(
        (
            labels == 0
        ).sum()
    )

    if (
        n_positive == 0
        or n_negative == 0
    ):
        return None

    ranks = average_ranks(
        scores
    )

    return float(
        (
            ranks[
                positive
            ].sum()
            -
            n_positive
            * (
                n_positive + 1
            )
            / 2.0
        )
        /
        (
            n_positive
            * n_negative
        )
    )


def quantile_table(
    uncertainty,
    error,
    bins: int = 4,
):
    uncertainty = np.asarray(
        uncertainty,
        dtype=np.float64,
    )

    error = np.asarray(
        error,
        dtype=np.float64,
    )

    groups = np.array_split(
        np.argsort(
            uncertainty,
            kind="mergesort",
        ),
        bins,
    )

    return [
        {
            "bin":
                index + 1,

            "count":
                int(
                    group.size
                ),

            "mean_uncertainty":
                float(
                    np.mean(
                        uncertainty[
                            group
                        ]
                    )
                ),

            "mean_error":
                float(
                    np.mean(
                        error[
                            group
                        ]
                    )
                ),
        }
        for index, group
        in enumerate(groups)
    ]


def grouped_metrics(
    predicted,
    target,
    labels,
):
    labels = np.asarray(
        labels,
        dtype=object,
    )

    return {
        str(label): {
            "count":
                int(
                    np.sum(
                        labels
                        == label
                    )
                ),

            "rmse":
                rmse(
                    predicted[
                        labels
                        == label
                    ],
                    target[
                        labels
                        == label
                    ],
                ),

            "mae":
                mae(
                    predicted[
                        labels
                        == label
                    ],
                    target[
                        labels
                        == label
                    ],
                ),
        }
        for label
        in sorted(
            set(
                labels.tolist()
            )
        )
    }


def one_step_metrics(
    model: (
        BootstrapProbabilisticWorldModel
    ),
    transitions,
):
    items = complete(
        transitions
    )

    dataset = (
        build_world_model_dataset(
            items
        )
    )

    result = (
        model.predict_ensemble(
            dataset.states,
            dataset.actions,
        )
    )

    predicted = result[
        "mean"
    ]

    target = (
        dataset.next_states
    )

    errors = sample_rmse(
        predicted,
        target,
    )

    uncertainty = np.mean(
        np.sqrt(
            np.maximum(
                result[
                    "epistemic_variance"
                ],
                0.0,
            )
        ),
        axis=1,
    )

    feature_rmse = np.sqrt(
        np.mean(
            (
                predicted
                - target
            )
            ** 2,
            axis=0,
        )
    )

    feature_mae = np.mean(
        np.abs(
            predicted
            - target
        ),
        axis=0,
    )

    return {
        "count":
            dataset.n_samples,

        "rmse":
            rmse(
                predicted,
                target,
            ),

        "mae":
            mae(
                predicted,
                target,
            ),

        "mixture_gaussian_nll_per_dim":
            mixture_nll_per_dim(
                target,
                result[
                    "member_means"
                ],
                result[
                    "member_variances"
                ],
            ),

        "persistence_rmse":
            rmse(
                dataset.states,
                target,
            ),

        "persistence_mae":
            mae(
                dataset.states,
                target,
            ),

        "epistemic_error_spearman":
            spearman(
                uncertainty,
                errors,
            ),

        "per_feature": {
            name: {
                "rmse":
                    float(
                        feature_rmse[
                            index
                        ]
                    ),

                "mae":
                    float(
                        feature_mae[
                            index
                        ]
                    ),
            }
            for index, name
            in enumerate(
                FORMAL_STATE_FEATURE_NAMES
            )
        },

        "per_executed_action":
            grouped_metrics(
                predicted,
                target,
                [
                    item
                    .executed_action_family
                    for item in items
                ],
            ),

        "per_agent":
            grouped_metrics(
                predicted,
                target,
                [
                    item.agent_name
                    for item in items
                ],
            ),
    }


def rollout_metrics(
    model,
    windows,
):
    initial = np.stack(
        [
            item.initial_state
            for item in windows
        ]
    ).astype(
        np.float32
    )

    actions = np.stack(
        [
            item.actions
            for item in windows
        ]
    ).astype(
        np.int64
    )

    target = np.stack(
        [
            item.target_state
            for item in windows
        ]
    ).astype(
        np.float32
    )

    horizon = (
        actions.shape[
            1
        ]
    )

    member_final = []

    for member_index in range(
        len(
            model.models
        )
    ):
        current = torch.as_tensor(
            initial,
            dtype=torch.float32,
            device=model.device,
        )

        for step in range(
            horizon
        ):
            action_tensor = (
                torch.as_tensor(
                    actions[
                        :,
                        step,
                    ],
                    dtype=torch.long,
                    device=model.device,
                )
            )

            current = (
                model
                .predict_member_mean_tensor(
                    member_index,
                    current,
                    action_tensor,
                )
            )

        member_final.append(
            current
            .detach()
            .cpu()
            .numpy()
        )

    member_final = np.stack(
        member_final,
        axis=0,
    )

    predicted = np.mean(
        member_final,
        axis=0,
    )

    errors = sample_rmse(
        predicted,
        target,
    )

    uncertainty = np.mean(
        np.std(
            member_final,
            axis=0,
            ddof=0,
        ),
        axis=1,
    )

    seeds = np.asarray(
        [
            item.episode_seed
            for item in windows
        ],
        dtype=np.int64,
    )

    per_episode_rmse = {}
    per_episode_spearman = {}

    for seed in sorted(
        set(
            seeds.tolist()
        )
    ):
        mask = (
            seeds == seed
        )

        per_episode_rmse[
            str(seed)
        ] = rmse(
            predicted[
                mask
            ],
            target[
                mask
            ],
        )

        per_episode_spearman[
            str(seed)
        ] = spearman(
            uncertainty[
                mask
            ],
            errors[
                mask
            ],
        )

    threshold = float(
        np.quantile(
            errors,
            0.75,
        )
    )

    high_error = (
        errors
        >= threshold
    ).astype(
        np.int64
    )

    return {
        "horizon":
            int(
                horizon
            ),

        "window_count":
            len(
                windows
            ),

        "rmse":
            rmse(
                predicted,
                target,
            ),

        "mae":
            mae(
                predicted,
                target,
            ),

        "persistence_rmse":
            rmse(
                initial,
                target,
            ),

        "persistence_mae":
            mae(
                initial,
                target,
            ),

        "epistemic_error_spearman":
            spearman(
                uncertainty,
                errors,
            ),

        "per_episode_rmse":
            per_episode_rmse,

        "per_episode_epistemic_error_spearman":
            per_episode_spearman,

        "positive_episode_spearman_count":
            int(
                sum(
                    value > 0.0
                    for value
                    in per_episode_spearman
                    .values()
                )
            ),

        "high_error_threshold":
            threshold,

        "high_error_auroc":
            binary_auc(
                high_error,
                uncertainty,
            ),

        "uncertainty_quantiles":
            quantile_table(
                uncertainty,
                errors,
            ),
    }


def evaluate_mode(
    target_mode,
    train_transitions,
    validation_transitions,
    out_dir,
    device,
):
    train_dataset = (
        build_world_model_dataset(
            train_transitions
        )
    )

    config = (
        BootstrapWorldModelConfig(
            target_mode=target_mode
        )
    )

    model = (
        BootstrapProbabilisticWorldModel(
            config=config,
            device=device,
        )
    )

    print(
        f"\n[A4.5b TRAIN] "
        f"mode={target_mode} "
        f"samples="
        f"{train_dataset.n_samples}"
    )

    training = model.fit(
        train_dataset
    )

    checkpoint = (
        out_dir
        /
        f"world_model_"
        f"{target_mode}.pt"
    )

    model.save_checkpoint(
        checkpoint
    )

    result = {
        "target_mode":
            target_mode,

        "checkpoint":
            str(
                checkpoint
            ),

        "training": {
            "n_samples":
                int(
                    training.n_samples
                ),

            "member_final_losses":
                [
                    float(value)
                    for value
                    in training
                    .member_final_losses
                ],

            "bootstrap_unique_fractions":
                [
                    float(value)
                    for value
                    in training
                    .bootstrap_unique_fractions
                ],
        },

        "one_step":
            one_step_metrics(
                model,
                validation_transitions,
            ),
    }

    for horizon in (
        ROLLOUT_HORIZONS
    ):
        result[
            f"rollout_h{horizon}"
        ] = rollout_metrics(
            model,
            build_rollout_windows(
                validation_transitions,
                horizon,
            ),
        )

    return result


def select_target_mode(
    absolute,
    delta,
):
    absolute_h4 = float(
        absolute[
            "rollout_h4"
        ][
            "rmse"
        ]
    )

    delta_h4 = float(
        delta[
            "rollout_h4"
        ][
            "rmse"
        ]
    )

    improvement = (
        0.0
        if absolute_h4 <= 0
        else (
            absolute_h4
            - delta_h4
        )
        / absolute_h4
    )

    absolute_episode = (
        absolute[
            "rollout_h4"
        ][
            "per_episode_rmse"
        ]
    )

    delta_episode = (
        delta[
            "rollout_h4"
        ][
            "per_episode_rmse"
        ]
    )

    common = sorted(
        set(
            absolute_episode
        )
        &
        set(
            delta_episode
        )
    )

    delta_wins = sum(
        float(
            delta_episode[
                seed
            ]
        )
        <
        float(
            absolute_episode[
                seed
            ]
        )
        for seed in common
    )

    selected = (
        "delta"
        if (
            improvement >= 0.02
            and delta_wins >= 6
        )
        else "absolute"
    )

    return {
        "selected_target_mode":
            selected,

        "absolute_h4_rmse":
            absolute_h4,

        "delta_h4_rmse":
            delta_h4,

        "delta_relative_h4_improvement":
            float(
                improvement
            ),

        "delta_episode_h4_wins":
            int(
                delta_wins
            ),

        "common_validation_episode_count":
            len(
                common
            ),

        "rule":
            (
                "delta requires >=2% "
                "aggregate H4 improvement "
                "and >=6/8 episode wins"
            ),
    }


def quality_gate(
    metrics,
):
    one_step = metrics[
        "one_step"
    ]

    h4 = metrics[
        "rollout_h4"
    ]

    result = {
        "one_step_beats_persistence":
            float(
                one_step[
                    "rmse"
                ]
            )
            <
            float(
                one_step[
                    "persistence_rmse"
                ]
            ),

        "h4_beats_persistence":
            float(
                h4[
                    "rmse"
                ]
            )
            <
            float(
                h4[
                    "persistence_rmse"
                ]
            ),

        "h4_aggregate_spearman_positive":
            float(
                h4[
                    "epistemic_error_spearman"
                ]
            )
            > 0.0,

        "h4_positive_episode_spearman_count":
            int(
                h4[
                    "positive_episode_spearman_count"
                ]
            ),
    }

    result[
        "h4_stable_positive_spearman"
    ] = (
        result[
            "h4_positive_episode_spearman_count"
        ]
        >= 5
    )

    result[
        "pass"
    ] = bool(
        result[
            "one_step_beats_persistence"
        ]
        and result[
            "h4_beats_persistence"
        ]
        and result[
            "h4_aggregate_spearman_positive"
        ]
        and result[
            "h4_stable_positive_spearman"
        ]
    )

    return result


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


def main():
    parser = argparse.ArgumentParser(
        description=(
            "A4.5b held-out "
            "world-model validation"
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
        "--out-dir",
        default=(
            "outputs/world_model/"
            "a4_5b"
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

    train_seeds = {
        item.episode_seed
        for item in train
    }

    validation_seeds = {
        item.episode_seed
        for item
        in validation
    }

    if (
        train_seeds
        &
        validation_seeds
    ):
        raise RuntimeError(
            "train/validation seed leakage"
        )

    out_dir = resolve_path(
        args.out_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "contract": {
            "state_dim":
                FORMAL_STATE_DIM,

            "ensemble_size":
                5,

            "hidden_dim":
                128,

            "epochs":
                50,

            "target_modes":
                list(
                    TARGET_MODES
                ),

            "rollout_horizons":
                list(
                    ROLLOUT_HORIZONS
                ),

            "selection_primary":
                (
                    "held-out H4 "
                    "final-state RMSE"
                ),

            "calibration_test_seeds_used":
                False,
        },

        "train_replay_count":
            len(
                train
            ),

        "validation_replay_count":
            len(
                validation
            ),

        "modes":
            {},
    }

    for mode in (
        TARGET_MODES
    ):
        report[
            "modes"
        ][
            mode
        ] = evaluate_mode(
            mode,
            train,
            validation,
            out_dir,
            args.device,
        )

        metrics = (
            report[
                "modes"
            ][
                mode
            ]
        )

        print(
            f"[A4.5b MODE] {mode} "
            f"one="
            f"{metrics['one_step']['rmse']:.6f} "
            f"one_persist="
            f"{metrics['one_step']['persistence_rmse']:.6f} "
            f"h2="
            f"{metrics['rollout_h2']['rmse']:.6f} "
            f"h4="
            f"{metrics['rollout_h4']['rmse']:.6f} "
            f"h4_persist="
            f"{metrics['rollout_h4']['persistence_rmse']:.6f} "
            f"h4_spearman="
            f"{metrics['rollout_h4']['epistemic_error_spearman']:.6f}"
        )

    selection = (
        select_target_mode(
            report[
                "modes"
            ][
                "absolute"
            ],
            report[
                "modes"
            ][
                "delta"
            ],
        )
    )

    selected = (
        selection[
            "selected_target_mode"
        ]
    )

    gate = quality_gate(
        report[
            "modes"
        ][
            selected
        ]
    )

    report[
        "selection"
    ] = selection

    report[
        "quality_gate"
    ] = gate

    report_path = (
        out_dir
        /
        "a4_5b_report.json"
    )

    report_path.write_text(
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
        "[A4.5b SUMMARY]"
    )

    print(
        "selected_target_mode:",
        selected,
    )

    print(
        "delta_relative_h4_improvement:",
        selection[
            "delta_relative_h4_improvement"
        ],
    )

    print(
        "delta_episode_h4_wins:",
        selection[
            "delta_episode_h4_wins"
        ],
    )

    print(
        "quality_gate:",
        gate,
    )

    print(
        "[OK] report:",
        report_path,
    )

    if not gate[
        "pass"
    ]:
        print(
            "[A4.5b GATE NOT PASSED] "
            "Do not enter planner integration."
        )


if __name__ == "__main__":
    main()