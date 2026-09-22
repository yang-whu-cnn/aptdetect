from __future__ import annotations

import argparse
import json

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from formal_experiments.data_collection.decision_replay import (
    DecisionEpochTransition,
)

from formal_experiments.evaluation.validate_bootstrap_world_model import (
    chained,
    executed_action_id,
    load_replay_jsonl,
    mae,
    rmse,
    spearman,
    validate_frozen_split,
)

from formal_experiments.training.bootstrap_world_model import (
    BootstrapProbabilisticWorldModel,
)

from formal_experiments.training.response_reward_predictor import (
    ResponseRewardPredictor,
    ResponseRewardPredictorConfig,
    build_response_reward_dataset,
)

from shared.action_contract import (
    ACTION_CONTRACTS,
)


GAMMA_TICK = 0.99
HORIZON = 4


ACTION_ID_TO_DURATION = {
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
class RewardRolloutWindow:
    episode_seed: int
    agent_name: str
    start_decision_index: int

    initial_state: np.ndarray

    states: np.ndarray
    actions: np.ndarray
    next_states: np.ndarray
    durations: np.ndarray
    rewards: np.ndarray


def pearson(
    x,
    y,
) -> float:
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    y = np.asarray(
        y,
        dtype=np.float64,
    )

    if (
        x.size < 2
        or np.std(x) == 0.0
        or np.std(y) == 0.0
    ):
        return 0.0

    return float(
        np.corrcoef(
            x,
            y,
        )[0, 1]
    )


def discount_weights(
    durations,
    gamma_tick: float,
) -> np.ndarray:
    durations = np.asarray(
        durations,
        dtype=np.int64,
    )

    if (
        durations.ndim != 1
        or durations.size == 0
        or np.any(
            durations <= 0
        )
    ):
        raise ValueError(
            "durations must be "
            "positive 1-D"
        )

    gamma_tick = float(
        gamma_tick
    )

    if not (
        0.0
        < gamma_tick
        <= 1.0
    ):
        raise ValueError(
            "gamma_tick must be "
            "in (0,1]"
        )

    elapsed = 0
    weights = []

    for duration in durations:

        weights.append(
            gamma_tick ** elapsed
        )

        elapsed += int(
            duration
        )

    return np.asarray(
        weights,
        dtype=np.float64,
    )


def discounted_sum(
    rewards,
    durations,
    gamma_tick: float,
) -> float:
    rewards = np.asarray(
        rewards,
        dtype=np.float64,
    )

    weights = discount_weights(
        durations,
        gamma_tick,
    )

    if rewards.shape != weights.shape:
        raise ValueError(
            "reward/duration "
            "shape mismatch"
        )

    return float(
        np.sum(
            rewards
            * weights
        )
    )


def build_reward_rollout_windows(
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
                for item
                in block
            ):
                continue

            if not all(
                chained(
                    block[index],
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

            actions = np.asarray(
                [
                    executed_action_id(
                        item
                    )
                    for item
                    in block
                ],
                dtype=np.int64,
            )

            durations = np.asarray(
                [
                    int(
                        item
                        .executed_duration_ticks
                    )
                    for item
                    in block
                ],
                dtype=np.int64,
            )

            for (
                action_id,
                duration,
            ) in zip(
                actions,
                durations,
            ):
                if (
                    ACTION_ID_TO_DURATION[
                        int(action_id)
                    ]
                    != int(duration)
                ):
                    raise RuntimeError(
                        "action/duration "
                        "contract mismatch"
                    )

            windows.append(
                RewardRolloutWindow(
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

                    states=np.stack(
                        [
                            item.state
                            for item
                            in block
                        ],
                        axis=0,
                    ).astype(
                        np.float32
                    ),

                    actions=actions,

                    next_states=np.stack(
                        [
                            item.next_state
                            for item
                            in block
                        ],
                        axis=0,
                    ).astype(
                        np.float32
                    ),

                    durations=durations,

                    rewards=np.asarray(
                        [
                            item.response_reward
                            for item
                            in block
                        ],
                        dtype=np.float32,
                    ),
                )
            )

    if not windows:
        raise ValueError(
            "no H=4 reward windows"
        )

    return tuple(
        windows
    )


def action_mean_table(
    dataset,
) -> dict[int, float]:
    result = {}

    global_mean = float(
        np.mean(
            dataset.rewards
        )
    )

    for action_id in range(
        4
    ):
        mask = (
            dataset.actions
            == action_id
        )

        result[
            action_id
        ] = (
            float(
                np.mean(
                    dataset.rewards[
                        mask
                    ]
                )
            )
            if np.any(mask)
            else global_mean
        )

    return result


def one_step_metrics(
    predictor,
    train_dataset,
    validation_transitions,
):
    validation_dataset = (
        build_response_reward_dataset(
            validation_transitions
        )
    )

    predicted = (
        predictor.predict(
            validation_dataset.states,
            validation_dataset.actions,
            validation_dataset.next_states,
        )
        .astype(
            np.float64
        )
    )

    target = (
        validation_dataset
        .rewards
        .astype(
            np.float64
        )
    )

    train_mean = float(
        np.mean(
            train_dataset.rewards
        )
    )

    constant = np.full_like(
        target,
        train_mean,
    )

    action_means = (
        action_mean_table(
            train_dataset
        )
    )

    action_baseline = (
        np.asarray(
            [
                action_means[
                    int(action)
                ]
                for action
                in validation_dataset
                .actions
            ],
            dtype=np.float64,
        )
    )

    per_action = {}

    for action_id in range(
        4
    ):
        mask = (
            validation_dataset
            .actions
            == action_id
        )

        per_action[
            str(action_id)
        ] = {
            "count":
                int(
                    mask.sum()
                ),

            "rmse":
                (
                    None
                    if not np.any(mask)
                    else rmse(
                        predicted[
                            mask
                        ],
                        target[
                            mask
                        ],
                    )
                ),

            "mae":
                (
                    None
                    if not np.any(mask)
                    else mae(
                        predicted[
                            mask
                        ],
                        target[
                            mask
                        ],
                    )
                ),
        }

    return {
        "count":
            int(
                target.size
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

        "pearson":
            pearson(
                predicted,
                target,
            ),

        "spearman":
            spearman(
                predicted,
                target,
            ),

        "train_mean_baseline_rmse":
            rmse(
                constant,
                target,
            ),

        "train_mean_baseline_mae":
            mae(
                constant,
                target,
            ),

        "action_mean_baseline_rmse":
            rmse(
                action_baseline,
                target,
            ),

        "action_mean_baseline_mae":
            mae(
                action_baseline,
                target,
            ),

        "per_action":
            per_action,
    }


def true_window_returns(
    windows,
    gamma_tick,
):
    return np.asarray(
        [
            discounted_sum(
                item.rewards,
                item.durations,
                gamma_tick,
            )
            for item
            in windows
        ],
        dtype=np.float64,
    )


def constant_window_baseline(
    windows,
    *,
    reward_mean,
    gamma_tick,
):
    return np.asarray(
        [
            discounted_sum(
                np.full(
                    len(
                        item.actions
                    ),
                    float(
                        reward_mean
                    ),
                    dtype=np.float64,
                ),
                item.durations,
                gamma_tick,
            )
            for item
            in windows
        ],
        dtype=np.float64,
    )


def oracle_state_h4_metrics(
    predictor,
    windows,
    train_reward_mean,
    gamma_tick,
):
    predicted_returns = []
    target_returns = []

    for item in windows:

        predicted_rewards = (
            predictor.predict(
                item.states,
                item.actions,
                item.next_states,
            )
        )

        predicted_returns.append(
            discounted_sum(
                predicted_rewards,
                item.durations,
                gamma_tick,
            )
        )

        target_returns.append(
            discounted_sum(
                item.rewards,
                item.durations,
                gamma_tick,
            )
        )

    predicted_returns = (
        np.asarray(
            predicted_returns,
            dtype=np.float64,
        )
    )

    target_returns = (
        np.asarray(
            target_returns,
            dtype=np.float64,
        )
    )

    baseline = (
        constant_window_baseline(
            windows,
            reward_mean=(
                train_reward_mean
            ),
            gamma_tick=(
                gamma_tick
            ),
        )
    )

    return {
        "window_count":
            len(
                windows
            ),

        "rmse":
            rmse(
                predicted_returns,
                target_returns,
            ),

        "mae":
            mae(
                predicted_returns,
                target_returns,
            ),

        "pearson":
            pearson(
                predicted_returns,
                target_returns,
            ),

        "spearman":
            spearman(
                predicted_returns,
                target_returns,
            ),

        "constant_baseline_rmse":
            rmse(
                baseline,
                target_returns,
            ),
    }


def wm_h4_metrics(
    *,
    world_model,
    predictor,
    windows,
    train_reward_mean,
    gamma_tick,
):
    if (
        world_model.config
        .target_mode
        != "absolute"
    ):
        raise ValueError(
            "A4.5c requires selected "
            "absolute WM checkpoint"
        )

    initial_states = np.stack(
        [
            item.initial_state
            for item
            in windows
        ],
        axis=0,
    ).astype(
        np.float32
    )

    actions = np.stack(
        [
            item.actions
            for item
            in windows
        ],
        axis=0,
    ).astype(
        np.int64
    )

    durations = np.stack(
        [
            item.durations
            for item
            in windows
        ],
        axis=0,
    ).astype(
        np.int64
    )

    true_returns = (
        true_window_returns(
            windows,
            gamma_tick,
        )
    )

    member_returns = []

    for member_index in range(
        len(
            world_model.models
        )
    ):
        current = torch.as_tensor(
            initial_states,
            dtype=torch.float32,
            device=world_model.device,
        )

        cumulative = np.zeros(
            len(
                windows
            ),
            dtype=np.float64,
        )

        elapsed = np.zeros(
            len(
                windows
            ),
            dtype=np.int64,
        )

        for step in range(
            HORIZON
        ):
            action_t = (
                torch.as_tensor(
                    actions[
                        :,
                        step,
                    ],
                    dtype=torch.long,
                    device=world_model.device,
                )
            )

            next_state = (
                world_model
                .predict_member_mean_tensor(
                    member_index,
                    current,
                    action_t,
                )
            )

            predicted_reward = (
                predictor
                .predict_tensor(
                    current,
                    action_t,
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
                gamma_tick
                ** elapsed.astype(
                    np.float64
                )
            )

            cumulative += (
                weights
                * predicted_reward
            )

            elapsed += (
                durations[
                    :,
                    step
                ]
            )

            current = next_state

        member_returns.append(
            cumulative
        )

    member_returns = np.stack(
        member_returns,
        axis=0,
    )

    predicted = np.mean(
        member_returns,
        axis=0,
    )

    return_uncertainty = np.std(
        member_returns,
        axis=0,
        ddof=0,
    )

    baseline = (
        constant_window_baseline(
            windows,
            reward_mean=(
                train_reward_mean
            ),
            gamma_tick=(
                gamma_tick
            ),
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

        value = spearman(
            predicted[
                mask
            ],
            true_returns[
                mask
            ],
        )

        if value > 0.0:
            positive_count += 1

        per_episode[
            str(seed)
        ] = {
            "count":
                int(
                    mask.sum()
                ),

            "rmse":
                rmse(
                    predicted[
                        mask
                    ],
                    true_returns[
                        mask
                    ],
                ),

            "spearman":
                float(
                    value
                ),
        }

    return {
        "window_count":
            len(
                windows
            ),

        "rmse":
            rmse(
                predicted,
                true_returns,
            ),

        "mae":
            mae(
                predicted,
                true_returns,
            ),

        "pearson":
            pearson(
                predicted,
                true_returns,
            ),

        "spearman":
            spearman(
                predicted,
                true_returns,
            ),

        "constant_baseline_rmse":
            rmse(
                baseline,
                true_returns,
            ),

        "return_uncertainty_true_error_spearman":
            spearman(
                return_uncertainty,
                np.abs(
                    predicted
                    - true_returns
                ),
            ),

        "positive_episode_spearman_count":
            int(
                positive_count
            ),

        "per_episode":
            per_episode,
    }


def quality_gate(
    *,
    one_step,
    wm_h4,
):
    result = {
        "one_step_beats_train_mean":
            float(
                one_step[
                    "rmse"
                ]
            )
            <
            float(
                one_step[
                    "train_mean_baseline_rmse"
                ]
            ),

        "wm_h4_beats_constant_return":
            float(
                wm_h4[
                    "rmse"
                ]
            )
            <
            float(
                wm_h4[
                    "constant_baseline_rmse"
                ]
            ),

        "wm_h4_spearman_gt_0_3":
            float(
                wm_h4[
                    "spearman"
                ]
            )
            > 0.3,

        "wm_h4_positive_episode_spearman_count":
            int(
                wm_h4[
                    "positive_episode_spearman_count"
                ]
            ),
    }

    result[
        "wm_h4_stable_episode_relation"
    ] = (
        result[
            "wm_h4_positive_episode_spearman_count"
        ]
        >= 5
    )

    result[
        "pass"
    ] = bool(
        result[
            "one_step_beats_train_mean"
        ]
        and result[
            "wm_h4_beats_constant_return"
        ]
        and result[
            "wm_h4_spearman_gt_0_3"
        ]
        and result[
            "wm_h4_stable_episode_relation"
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
        / path
    ).resolve()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "A4.5c response-reward "
            "predictor validation"
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
        "--out-dir",
        default=(
            "outputs/world_model/"
            "a4_5c"
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

    if (
        {
            item.episode_seed
            for item
            in train
        }
        &
        {
            item.episode_seed
            for item
            in validation
        }
    ):
        raise RuntimeError(
            "train/validation leakage"
        )

    train_dataset = (
        build_response_reward_dataset(
            train
        )
    )

    predictor = (
        ResponseRewardPredictor(
            ResponseRewardPredictorConfig(),
            device=args.device,
        )
    )

    training = predictor.fit(
        train_dataset
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

    if (
        world_model.config
        .target_mode
        != "absolute"
    ):
        raise RuntimeError(
            "selected WM checkpoint "
            "must be absolute"
        )

    windows = (
        build_reward_rollout_windows(
            validation,
            horizon=HORIZON,
        )
    )

    one_step = (
        one_step_metrics(
            predictor,
            train_dataset,
            validation,
        )
    )

    train_reward_mean = float(
        np.mean(
            train_dataset.rewards
        )
    )

    oracle_h4 = (
        oracle_state_h4_metrics(
            predictor,
            windows,
            train_reward_mean,
            GAMMA_TICK,
        )
    )

    wm_h4 = (
        wm_h4_metrics(
            world_model=world_model,
            predictor=predictor,
            windows=windows,
            train_reward_mean=(
                train_reward_mean
            ),
            gamma_tick=(
                GAMMA_TICK
            ),
        )
    )

    gate = quality_gate(
        one_step=one_step,
        wm_h4=wm_h4,
    )

    out_dir = resolve_path(
        args.out_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        out_dir
        /
        "response_reward_predictor.pt"
    )

    predictor.save_checkpoint(
        checkpoint_path
    )

    report = {
        "contract": {
            "state_dim":
                27,

            "n_actions":
                4,

            "reward_model_input":
                (
                    "state + executed/canonical "
                    "action + next_state"
                ),

            "hidden_reward_inputs":
                False,

            "gamma_tick":
                GAMMA_TICK,

            "horizon":
                HORIZON,

            "world_model_target_mode":
                "absolute",

            "calibration_test_seeds_used":
                False,
        },

        "training": {
            "n_samples":
                int(
                    training.n_samples
                ),

            "final_loss":
                float(
                    training.final_loss
                ),

            "reward_mean":
                float(
                    training.reward_mean
                ),

            "reward_std":
                float(
                    training.reward_std
                ),
        },

        "one_step":
            one_step,

        "oracle_state_h4":
            oracle_h4,

        "wm_h4":
            wm_h4,

        "quality_gate":
            gate,

        "checkpoint":
            str(
                checkpoint_path
            ),
    }

    report_path = (
        out_dir
        /
        "a4_5c_report.json"
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
        "[A4.5c SUMMARY]"
    )

    print(
        "train samples:",
        training.n_samples,
    )

    print(
        "one-step RMSE:",
        one_step[
            "rmse"
        ],
    )

    print(
        "one-step baseline RMSE:",
        one_step[
            "train_mean_baseline_rmse"
        ],
    )

    print(
        "one-step Spearman:",
        one_step[
            "spearman"
        ],
    )

    print(
        "oracle H4 RMSE:",
        oracle_h4[
            "rmse"
        ],
    )

    print(
        "oracle H4 Spearman:",
        oracle_h4[
            "spearman"
        ],
    )

    print(
        "WM H4 RMSE:",
        wm_h4[
            "rmse"
        ],
    )

    print(
        "WM H4 baseline RMSE:",
        wm_h4[
            "constant_baseline_rmse"
        ],
    )

    print(
        "WM H4 Spearman:",
        wm_h4[
            "spearman"
        ],
    )

    print(
        "positive episode Spearman:",
        wm_h4[
            "positive_episode_spearman_count"
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
            "[A4.5c GATE NOT PASSED] "
            "Do not enter Step 4."
        )


if __name__ == "__main__":
    main()