import unittest

import numpy as np

from formal_experiments.data_collection.decision_replay import (
    DecisionEpochReplayCollector,
)
from shared.action_contract import get_action
from shared.cyborg_action_adapter import (
    CybORGActionResolution,
)
from shared.formal_state import FORMAL_STATE_DIM


def state(
    value=0.0,
):
    return np.full(
        (FORMAL_STATE_DIM,),
        value,
        dtype=np.float32,
    )


def resolution(
    agent,
    action_id,
    *,
    fallback=False,
):
    action = get_action(
        action_id
    )

    if fallback:
        executed_family = "Sleep"
        executed_label = "Sleep"
        executed_index = 49
        target_host = None
        reason = (
            "no_valid_observable_target"
        )

    else:
        executed_family = (
            action.cyborg_action
        )

        reason = None

        if executed_family == "Sleep":
            executed_label = "Sleep"
            executed_index = 49
            target_host = None
        else:
            target_host = "host_a"
            executed_label = (
                f"{executed_family} "
                f"{target_host}"
            )

            executed_index = {
                "Analyse": 0,
                "Remove": 17,
                "Restore": 33,
            }[
                executed_family
            ]

    return CybORGActionResolution(
        agent_name=agent,

        requested_action_id=(
            action.action_id
        ),

        requested_action_name=(
            action.name
        ),

        requested_cyborg_family=(
            action.cyborg_action
        ),

        executed_index=(
            executed_index
        ),

        executed_label=(
            executed_label
        ),

        executed_action_family=(
            executed_family
        ),

        target_host=(
            target_host
        ),

        fallback=fallback,

        fallback_reason=(
            reason
        ),
    )


class TestDecisionEpochReplay(
    unittest.TestCase
):
    def test_incident_ids_accumulate_across_interval(
            self,
    ):
        c = DecisionEpochReplayCollector(
            42
        )

        c.begin_decision(
            agent_name="blue_agent_0",
            decision_index=0,
            global_tick_start=0,
            state=state(),
            resolution=resolution(
                "blue_agent_0",
                1,
            ),
        )

        c.record_tick(
            agent_name="blue_agent_0",
            global_tick_end=1,
            incident_event_ids=(
                "event_a",
            ),
            incident_host_ids=(
                "host_a",
            ),
        )

        c.record_tick(
            agent_name="blue_agent_0",
            global_tick_end=2,
            incident_event_ids=(
                "event_a",
                "event_b",
            ),
            incident_host_ids=(
                "host_a",
                "host_b",
            ),
        )

        t = c.end_decision(
            agent_name="blue_agent_0",
            global_tick_end=2,
            next_state=state(),
            done=False,
        )

        self.assertEqual(
            t.incident_event_ids,
            (
                "event_a",
                "event_b",
            ),
        )

        self.assertEqual(
            t.incident_host_ids,
            (
                "host_a",
                "host_b",
            ),
        )

    def test_sleep_transition_dt_one(
        self,
    ):
        c = (
            DecisionEpochReplayCollector(
                episode_seed=42
            )
        )

        c.begin_decision(
            agent_name="blue_agent_0",
            decision_index=0,
            global_tick_start=0,
            state=state(0),
            resolution=resolution(
                "blue_agent_0",
                0,
            ),
        )

        c.record_tick(
            agent_name="blue_agent_0",
            global_tick_end=1,
        )

        t = c.end_decision(
            agent_name="blue_agent_0",
            global_tick_end=1,
            next_state=state(1),
            done=False,
            completed_action_success=True,
        )

        self.assertEqual(
            t.decision_dt,
            1,
        )

        self.assertTrue(
            t.action_completed
        )

    def test_analyse_transition_dt_two(
        self,
    ):
        c = DecisionEpochReplayCollector(
            42
        )

        c.begin_decision(
            agent_name="blue_agent_0",
            decision_index=0,
            global_tick_start=10,
            state=state(),
            resolution=resolution(
                "blue_agent_0",
                1,
            ),
        )

        c.record_tick(
            agent_name="blue_agent_0",
            global_tick_end=11,
        )

        c.record_tick(
            agent_name="blue_agent_0",
            global_tick_end=12,
        )

        t = c.end_decision(
            agent_name="blue_agent_0",
            global_tick_end=12,
            next_state=state(1),
            done=False,
        )

        self.assertEqual(
            t.decision_dt,
            2,
        )

    def test_fallback_uses_executed_duration(
        self,
    ):
        c = DecisionEpochReplayCollector(
            42
        )

        c.begin_decision(
            agent_name="blue_agent_0",
            decision_index=0,
            global_tick_start=0,
            state=state(),
            resolution=resolution(
                "blue_agent_0",
                3,
                fallback=True,
            ),
        )

        c.record_tick(
            agent_name="blue_agent_0",
            global_tick_end=1,
        )

        t = c.end_decision(
            agent_name="blue_agent_0",
            global_tick_end=1,
            next_state=state(),
            done=False,
        )

        self.assertEqual(
            t.requested_duration_ticks,
            5,
        )

        self.assertEqual(
            t.executed_duration_ticks,
            1,
        )

        self.assertEqual(
            t.executed_action_family,
            "Sleep",
        )

    def test_nonterminal_early_close_rejected(
        self,
    ):
        c = DecisionEpochReplayCollector(
            42
        )

        c.begin_decision(
            agent_name="blue_agent_0",
            decision_index=0,
            global_tick_start=0,
            state=state(),
            resolution=resolution(
                "blue_agent_0",
                3,
            ),
        )

        c.record_tick(
            agent_name="blue_agent_0",
            global_tick_end=2,
        )

        with self.assertRaises(
            RuntimeError
        ):
            c.end_decision(
                agent_name="blue_agent_0",
                global_tick_end=2,
                next_state=state(),
                done=False,
            )

    def test_terminal_mid_action_allowed(
        self,
    ):
        c = DecisionEpochReplayCollector(
            42
        )

        c.begin_decision(
            agent_name="blue_agent_0",
            decision_index=0,
            global_tick_start=0,
            state=state(),
            resolution=resolution(
                "blue_agent_0",
                3,
            ),
        )

        c.record_tick(
            agent_name="blue_agent_0",
            global_tick_end=3,
        )

        t = c.end_decision(
            agent_name="blue_agent_0",
            global_tick_end=3,
            next_state=state(),
            done=True,
        )

        self.assertEqual(
            t.decision_dt,
            3,
        )

        self.assertFalse(
            t.action_completed
        )

    def test_final_tick_must_be_accounted(
        self,
    ):
        c = DecisionEpochReplayCollector(
            42
        )

        c.begin_decision(
            agent_name="blue_agent_0",
            decision_index=0,
            global_tick_start=0,
            state=state(),
            resolution=resolution(
                "blue_agent_0",
                0,
            ),
        )

        with self.assertRaises(
            RuntimeError
        ):
            c.end_decision(
                agent_name="blue_agent_0",
                global_tick_end=1,
                next_state=state(),
                done=False,
            )

    def test_multi_agent_asynchronous_epochs(
        self,
    ):
        c = DecisionEpochReplayCollector(
            42
        )

        c.begin_decision(
            agent_name="blue_agent_0",
            decision_index=0,
            global_tick_start=0,
            state=state(),
            resolution=resolution(
                "blue_agent_0",
                0,
            ),
        )

        c.begin_decision(
            agent_name="blue_agent_1",
            decision_index=0,
            global_tick_start=0,
            state=state(),
            resolution=resolution(
                "blue_agent_1",
                3,
            ),
        )

        c.record_tick(
            agent_name="blue_agent_0",
            global_tick_end=1,
        )

        c.record_tick(
            agent_name="blue_agent_1",
            global_tick_end=1,
        )

        a = c.end_decision(
            agent_name="blue_agent_0",
            global_tick_end=1,
            next_state=state(1),
            done=False,
        )

        for tick in (
            2,
            3,
            4,
            5,
        ):
            c.record_tick(
                agent_name="blue_agent_1",
                global_tick_end=tick,
            )

        b = c.end_decision(
            agent_name="blue_agent_1",
            global_tick_end=5,
            next_state=state(2),
            done=False,
        )

        self.assertEqual(
            a.decision_dt,
            1,
        )

        self.assertEqual(
            b.decision_dt,
            5,
        )

        self.assertEqual(
            len(c.buffer),
            2,
        )

    def test_interval_bookkeeping_accumulates(
        self,
    ):
        c = DecisionEpochReplayCollector(
            42
        )

        c.begin_decision(
            agent_name="blue_agent_0",
            decision_index=0,
            global_tick_start=0,
            state=state(),
            resolution=resolution(
                "blue_agent_0",
                1,
            ),
        )

        for tick in (1, 2):
            c.record_tick(
                agent_name="blue_agent_0",
                global_tick_end=tick,
                official_reward=-0.5,
                incident_active_ticks=1,
                incident_host_lwf_count=1,
                incident_host_lwf_raw_penalty=-1.0,
                response_reward=-2.0,
            )

        t = c.end_decision(
            agent_name="blue_agent_0",
            global_tick_end=2,
            next_state=state(),
            done=False,
        )

        self.assertEqual(
            t.incident_active_ticks,
            2,
        )

        self.assertEqual(
            t.incident_host_lwf_count,
            2,
        )

        self.assertEqual(
            t.incident_host_lwf_raw_penalty,
            -2.0,
        )

        self.assertEqual(
            t.response_reward,
            -4.0,
        )

        self.assertEqual(
            t.official_reward,
            -1.0,
        )

    def test_positive_raw_lwf_rejected(
        self,
    ):
        c = DecisionEpochReplayCollector(
            42
        )

        c.begin_decision(
            agent_name="blue_agent_0",
            decision_index=0,
            global_tick_start=0,
            state=state(),
            resolution=resolution(
                "blue_agent_0",
                0,
            ),
        )

        with self.assertRaises(
            ValueError
        ):
            c.record_tick(
                agent_name="blue_agent_0",
                global_tick_end=1,
                incident_host_lwf_raw_penalty=1.0,
            )

    def test_duplicate_open_rejected(
        self,
    ):
        c = DecisionEpochReplayCollector(
            42
        )

        r = resolution(
            "blue_agent_0",
            0,
        )

        c.begin_decision(
            agent_name="blue_agent_0",
            decision_index=0,
            global_tick_start=0,
            state=state(),
            resolution=r,
        )

        with self.assertRaises(
            RuntimeError
        ):
            c.begin_decision(
                agent_name="blue_agent_0",
                decision_index=0,
                global_tick_start=0,
                state=state(),
                resolution=r,
            )

    def test_decision_index_must_be_contiguous(
        self,
    ):
        c = DecisionEpochReplayCollector(
            42
        )

        with self.assertRaises(
            ValueError
        ):
            c.begin_decision(
                agent_name="blue_agent_0",
                decision_index=2,
                global_tick_start=0,
                state=state(),
                resolution=resolution(
                    "blue_agent_0",
                    0,
                ),
            )

    def test_resolution_agent_mismatch_rejected(
        self,
    ):
        c = DecisionEpochReplayCollector(
            42
        )

        with self.assertRaises(
            ValueError
        ):
            c.begin_decision(
                agent_name="blue_agent_0",
                decision_index=0,
                global_tick_start=0,
                state=state(),
                resolution=resolution(
                    "blue_agent_1",
                    0,
                ),
            )

    def test_state_is_copied(
        self,
    ):
        c = DecisionEpochReplayCollector(
            42
        )

        original = state(1)

        c.begin_decision(
            agent_name="blue_agent_0",
            decision_index=0,
            global_tick_start=0,
            state=original,
            resolution=resolution(
                "blue_agent_0",
                0,
            ),
        )

        original[:] = 99.0

        c.record_tick(
            agent_name="blue_agent_0",
            global_tick_end=1,
        )

        t = c.end_decision(
            agent_name="blue_agent_0",
            global_tick_end=1,
            next_state=state(),
            done=False,
        )

        self.assertTrue(
            np.all(
                t.state == 1.0
            )
        )

    def test_jsonable_state_is_list(
        self,
    ):
        c = DecisionEpochReplayCollector(
            42
        )

        c.begin_decision(
            agent_name="blue_agent_0",
            decision_index=0,
            global_tick_start=0,
            state=state(),
            resolution=resolution(
                "blue_agent_0",
                0,
            ),
        )

        c.record_tick(
            agent_name="blue_agent_0",
            global_tick_end=1,
        )

        t = c.end_decision(
            agent_name="blue_agent_0",
            global_tick_end=1,
            next_state=state(),
            done=False,
        )

        payload = t.to_jsonable()

        self.assertIsInstance(
            payload["state"],
            list,
        )

        self.assertEqual(
            len(
                payload["state"]
            ),
            FORMAL_STATE_DIM,
        )

    def test_invalid_state_dimension_rejected(
        self,
    ):
        c = DecisionEpochReplayCollector(
            42
        )

        with self.assertRaises(
            ValueError
        ):
            c.begin_decision(
                agent_name="blue_agent_0",
                decision_index=0,
                global_tick_start=0,
                state=np.zeros(
                    (10,),
                    dtype=np.float32,
                ),
                resolution=resolution(
                    "blue_agent_0",
                    0,
                ),
            )


if __name__ == "__main__":
    unittest.main()