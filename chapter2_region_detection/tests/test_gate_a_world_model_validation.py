import json
import tempfile
import unittest

from pathlib import Path

import numpy as np

from formal_experiments.evaluation.validate_bootstrap_world_model import (
    average_ranks,
    binary_auc,
    build_rollout_windows,
    load_replay_jsonl,
    mixture_nll_per_dim,
    quantile_table,
    select_target_mode,
    spearman,
    transition_from_json,
)

from formal_experiments.data_collection.decision_replay import (
    DecisionEpochTransition,
    EXECUTED_DURATION,
)

from shared.action_contract import (
    get_action,
)

from shared.formal_state import (
    FORMAL_STATE_DIM,
)


def state(
    value,
):
    return np.full(
        (FORMAL_STATE_DIM,),
        value,
        dtype=np.float32,
    )


def transition(
    *,
    seed=2000,
    agent="blue_agent_0",
    decision_index=0,
    family="Sleep",
    start_tick=0,
    start_value=0.0,
    next_value=0.1,
    completed=True,
):
    action_id = {
        "Sleep": 0,
        "Analyse": 1,
        "Remove": 2,
        "Restore": 3,
    }[
        family
    ]

    contract = get_action(
        action_id
    )

    duration = (
        EXECUTED_DURATION[
            family
        ]
    )

    dt = (
        duration
        if completed
        else max(
            1,
            duration - 1,
        )
    )

    return DecisionEpochTransition(
        episode_seed=seed,
        agent_name=agent,
        decision_index=decision_index,

        global_tick_start=start_tick,
        global_tick_end=(
            start_tick + dt
        ),
        decision_dt=dt,

        state=state(
            start_value
        ),
        next_state=state(
            next_value
        ),

        requested_action_id=(
            action_id
        ),
        requested_action_name=(
            contract.name
        ),
        requested_cyborg_family=(
            contract.cyborg_action
        ),
        requested_duration_ticks=(
            contract.duration_ticks
        ),

        executed_index=0,
        executed_label=family,
        executed_action_family=family,
        executed_duration_ticks=duration,

        target_host=(
            None
            if family == "Sleep"
            else "host_a"
        ),

        fallback=False,
        fallback_reason=None,

        action_completed=completed,
        completed_action_success=(
            True
            if completed
            else None
        ),

        incident_event_ids=tuple(),
        incident_host_ids=tuple(),

        incident_active_ticks=0,
        incident_host_lwf_count=0,
        incident_host_lwf_raw_penalty=0.0,

        response_reward=0.0,
        official_reward=0.0,

        done=(
            not completed
        ),
    )


def chained_sequence(
    n=6,
):
    result = []

    tick = 0
    value = 0.0

    families = [
        "Sleep",
        "Analyse",
        "Remove",
        "Restore",
        "Sleep",
        "Analyse",
    ]

    for index in range(
        n
    ):
        family = families[
            index
            % len(
                families
            )
        ]

        duration = (
            EXECUTED_DURATION[
                family
            ]
        )

        next_value = (
            value + 0.1
        )

        result.append(
            transition(
                decision_index=index,
                family=family,
                start_tick=tick,
                start_value=value,
                next_value=next_value,
            )
        )

        tick += duration
        value = next_value

    return result


class TestGateAWorldModelValidation(
    unittest.TestCase
):

    def test_transition_json_round_trip(
        self,
    ):
        item = transition()

        recovered = (
            transition_from_json(
                item.to_jsonable()
            )
        )

        np.testing.assert_allclose(
            recovered.state,
            item.state,
        )

        np.testing.assert_allclose(
            recovered.next_state,
            item.next_state,
        )

    def test_load_jsonl(
        self,
    ):
        items = (
            chained_sequence(
                2
            )
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = (
                Path(tmp)
                /
                "replay.jsonl"
            )

            path.write_text(
                "\n".join(
                    json.dumps(
                        item.to_jsonable()
                    )
                    for item
                    in items
                )
                + "\n",
                encoding="utf-8",
            )

            loaded = (
                load_replay_jsonl(
                    path
                )
            )

        self.assertEqual(
            len(loaded),
            2,
        )

    def test_h4_window_count(
        self,
    ):
        windows = (
            build_rollout_windows(
                chained_sequence(
                    6
                ),
                horizon=4,
            )
        )

        self.assertEqual(
            len(windows),
            3,
        )

    def test_window_does_not_cross_agent(
        self,
    ):
        left = (
            chained_sequence(
                4
            )
        )

        right = [
            transition(
                agent="blue_agent_1",
                decision_index=index,
                start_tick=index,
                start_value=float(
                    index
                ),
                next_value=float(
                    index + 1
                ),
            )
            for index
            in range(
                4
            )
        ]

        windows = (
            build_rollout_windows(
                left + right,
                horizon=4,
            )
        )

        self.assertEqual(
            len(windows),
            2,
        )

    def test_incomplete_breaks_window(
        self,
    ):
        sequence = (
            chained_sequence(
                5
            )
        )

        bad = transition(
            decision_index=5,
            family="Restore",
            start_tick=(
                sequence[
                    -1
                ]
                .global_tick_end
            ),
            start_value=0.5,
            next_value=0.6,
            completed=False,
        )

        windows = (
            build_rollout_windows(
                sequence + [bad],
                horizon=4,
            )
        )

        self.assertEqual(
            len(windows),
            2,
        )

    def test_average_ranks_with_tie(
        self,
    ):
        ranks = average_ranks(
            np.array(
                [
                    10.0,
                    20.0,
                    20.0,
                    30.0,
                ]
            )
        )

        np.testing.assert_allclose(
            ranks,
            [
                1.0,
                2.5,
                2.5,
                4.0,
            ],
        )

    def test_spearman_sign(
        self,
    ):
        x = np.arange(
            10,
            dtype=np.float64,
        )

        self.assertGreater(
            spearman(
                x,
                x,
            ),
            0.99,
        )

        self.assertLess(
            spearman(
                x,
                -x,
            ),
            -0.99,
        )

    def test_binary_auc_perfect(
        self,
    ):
        labels = np.array(
            [
                0,
                0,
                1,
                1,
            ]
        )

        scores = np.array(
            [
                0.1,
                0.2,
                0.8,
                0.9,
            ]
        )

        self.assertAlmostEqual(
            binary_auc(
                labels,
                scores,
            ),
            1.0,
        )

    def test_mixture_nll_prefers_good_prediction(
        self,
    ):
        target = np.zeros(
            (4, 2)
        )

        variance = (
            np.ones(
                (2, 4, 2)
            )
            * 0.1
        )

        good = np.zeros(
            (2, 4, 2)
        )

        bad = (
            np.ones(
                (2, 4, 2)
            )
            * 2.0
        )

        self.assertLess(
            mixture_nll_per_dim(
                target,
                good,
                variance,
            ),
            mixture_nll_per_dim(
                target,
                bad,
                variance,
            ),
        )

    def test_quantile_table_four_bins(
        self,
    ):
        uncertainty = np.arange(
            8,
            dtype=np.float64,
        )

        error = np.arange(
            8,
            dtype=np.float64,
        )

        table = quantile_table(
            uncertainty,
            error,
            bins=4,
        )

        self.assertEqual(
            len(table),
            4,
        )

        self.assertEqual(
            sum(
                row[
                    "count"
                ]
                for row
                in table
            ),
            8,
        )

    def test_delta_selection_requires_two_percent(
        self,
    ):
        absolute = {
            "rollout_h4": {
                "rmse": 1.0,
                "per_episode_rmse": {
                    str(index): 1.0
                    for index
                    in range(8)
                },
            }
        }

        delta = {
            "rollout_h4": {
                "rmse": 0.99,
                "per_episode_rmse": {
                    str(index): 0.9
                    for index
                    in range(8)
                },
            }
        }

        result = (
            select_target_mode(
                absolute,
                delta,
            )
        )

        self.assertEqual(
            result[
                "selected_target_mode"
            ],
            "absolute",
        )

    def test_delta_requires_six_episode_wins(
        self,
    ):
        absolute = {
            "rollout_h4": {
                "rmse": 1.0,
                "per_episode_rmse": {
                    str(index): 1.0
                    for index
                    in range(8)
                },
            }
        }

        delta_episode = {
            str(index):
                (
                    0.8
                    if index < 5
                    else 1.1
                )
            for index
            in range(8)
        }

        delta = {
            "rollout_h4": {
                "rmse": 0.95,
                "per_episode_rmse":
                    delta_episode,
            }
        }

        result = (
            select_target_mode(
                absolute,
                delta,
            )
        )

        self.assertEqual(
            result[
                "selected_target_mode"
            ],
            "absolute",
        )

    def test_delta_selected_when_both_rules_pass(
        self,
    ):
        absolute = {
            "rollout_h4": {
                "rmse": 1.0,
                "per_episode_rmse": {
                    str(index): 1.0
                    for index
                    in range(8)
                },
            }
        }

        delta_episode = {
            str(index):
                (
                    0.8
                    if index < 6
                    else 1.1
                )
            for index
            in range(8)
        }

        delta = {
            "rollout_h4": {
                "rmse": 0.95,
                "per_episode_rmse":
                    delta_episode,
            }
        }

        result = (
            select_target_mode(
                absolute,
                delta,
            )
        )

        self.assertEqual(
            result[
                "selected_target_mode"
            ],
            "delta",
        )


if __name__ == "__main__":
    unittest.main()