import unittest

from formal_experiments.data_collection.incident_response import (
    IncidentResponseBookkeeper,
    LocalWorkFailure,
    ResponseRewardConfig,
)


HOST_A = "host_a"
HOST_B = "host_b"


def make_bookkeeper(
    *,
    config=None,
    initial_a=False,
    initial_b=False,
):
    book = IncidentResponseBookkeeper(
        episode_seed=42,
        agent_name="blue_agent_0",
        reward_config=config,
    )

    book.reset(
        initial_red_presence_by_host={
            HOST_A: initial_a,
            HOST_B: initial_b,
        },
        global_tick=0,
    )

    return book


class TestGateAIncidentResponse(
    unittest.TestCase
):

    def test_compromise_transition_starts_event(
        self,
    ):
        book = make_bookkeeper()

        tick = book.record_tick(
            global_tick_end=1,
            red_presence_after={
                HOST_A: True,
                HOST_B: False,
            },
        )

        self.assertEqual(
            tick.incident_active_ticks,
            0,
        )

        self.assertEqual(
            tick.response_reward,
            0.0,
        )

        self.assertEqual(
            len(tick.started_event_ids),
            1,
        )

        event = book.all_events[0]

        self.assertEqual(
            event.t_compromise,
            1,
        )

    def test_active_tick_penalty(
        self,
    ):
        book = make_bookkeeper()

        book.record_tick(
            global_tick_end=1,
            red_presence_after={
                HOST_A: True,
                HOST_B: False,
            },
        )

        tick = book.record_tick(
            global_tick_end=2,
            red_presence_after={
                HOST_A: True,
                HOST_B: False,
            },
        )

        self.assertEqual(
            tick.incident_active_ticks,
            1,
        )

        self.assertEqual(
            tick.response_reward,
            -1.0,
        )

    def test_recovery_interval_is_counted(
        self,
    ):
        book = make_bookkeeper()

        book.record_tick(
            global_tick_end=1,
            red_presence_after={
                HOST_A: True,
                HOST_B: False,
            },
        )

        book.record_tick(
            global_tick_end=2,
            red_presence_after={
                HOST_A: True,
                HOST_B: False,
            },
        )

        tick = book.record_tick(
            global_tick_end=3,
            red_presence_after={
                HOST_A: False,
                HOST_B: False,
            },
        )

        self.assertEqual(
            tick.incident_active_ticks,
            1,
        )

        event = book.completed_events[0]

        self.assertEqual(
            event.t_compromise,
            1,
        )

        self.assertEqual(
            event.t_normal,
            3,
        )

        self.assertEqual(
            event.attack_eradication_time,
            2,
        )

        self.assertEqual(
            book.total_incident_active_ticks,
            2,
        )

    def test_active_ticks_equal_eradication_time(
        self,
    ):
        book = make_bookkeeper()

        book.record_tick(
            global_tick_end=1,
            red_presence_after={
                HOST_A: True,
                HOST_B: False,
            },
        )

        for tick in (2, 3, 4):
            book.record_tick(
                global_tick_end=tick,
                red_presence_after={
                    HOST_A:
                        tick < 4,
                    HOST_B: False,
                },
            )

        event = book.completed_events[0]

        self.assertEqual(
            event.attack_eradication_time,
            3,
        )

        self.assertEqual(
            book.total_incident_active_ticks,
            3,
        )

    def test_incident_host_lwf_counted(
        self,
    ):
        book = make_bookkeeper()

        book.record_tick(
            global_tick_end=1,
            red_presence_after={
                HOST_A: True,
                HOST_B: False,
            },
        )

        tick = book.record_tick(
            global_tick_end=2,
            red_presence_after={
                HOST_A: True,
                HOST_B: False,
            },
            local_work_failures=[
                LocalWorkFailure(
                    hostname=HOST_A,
                    raw_lwf_penalty=-2.0,
                ),
            ],
        )

        self.assertEqual(
            tick.incident_host_lwf_count,
            1,
        )

        self.assertEqual(
            tick.incident_host_lwf_raw_penalty,
            -2.0,
        )

        self.assertEqual(
            tick.response_reward,
            -3.0,
        )

    def test_other_host_lwf_not_counted(
        self,
    ):
        book = make_bookkeeper()

        book.record_tick(
            global_tick_end=1,
            red_presence_after={
                HOST_A: True,
                HOST_B: False,
            },
        )

        tick = book.record_tick(
            global_tick_end=2,
            red_presence_after={
                HOST_A: True,
                HOST_B: False,
            },
            local_work_failures=[
                LocalWorkFailure(
                    hostname=HOST_B,
                    raw_lwf_penalty=-10.0,
                ),
            ],
        )

        self.assertEqual(
            tick.incident_host_lwf_count,
            0,
        )

        self.assertEqual(
            tick.incident_host_lwf_raw_penalty,
            0.0,
        )

        self.assertEqual(
            tick.response_reward,
            -1.0,
        )

    def test_concurrent_incidents_sum_active_ticks(
        self,
    ):
        book = make_bookkeeper()

        book.record_tick(
            global_tick_end=1,
            red_presence_after={
                HOST_A: True,
                HOST_B: True,
            },
        )

        tick = book.record_tick(
            global_tick_end=2,
            red_presence_after={
                HOST_A: True,
                HOST_B: True,
            },
        )

        self.assertEqual(
            tick.incident_active_ticks,
            2,
        )

        self.assertEqual(
            tick.response_reward,
            -2.0,
        )

        self.assertEqual(
            set(tick.active_host_ids),
            {
                HOST_A,
                HOST_B,
            },
        )

    def test_concurrent_lwf_counts_each_active_host(
        self,
    ):
        book = make_bookkeeper()

        book.record_tick(
            global_tick_end=1,
            red_presence_after={
                HOST_A: True,
                HOST_B: True,
            },
        )

        tick = book.record_tick(
            global_tick_end=2,
            red_presence_after={
                HOST_A: True,
                HOST_B: True,
            },
            local_work_failures=[
                LocalWorkFailure(
                    HOST_A,
                    -1.0,
                ),
                LocalWorkFailure(
                    HOST_B,
                    -2.0,
                ),
            ],
        )

        self.assertEqual(
            tick.incident_host_lwf_count,
            2,
        )

        self.assertEqual(
            tick.incident_host_lwf_raw_penalty,
            -3.0,
        )

        self.assertEqual(
            tick.response_reward,
            -5.0,
        )

    def test_repeated_incident_gets_new_id(
        self,
    ):
        book = make_bookkeeper()

        book.record_tick(
            global_tick_end=1,
            red_presence_after={
                HOST_A: True,
                HOST_B: False,
            },
        )

        book.record_tick(
            global_tick_end=2,
            red_presence_after={
                HOST_A: False,
                HOST_B: False,
            },
        )

        book.record_tick(
            global_tick_end=3,
            red_presence_after={
                HOST_A: True,
                HOST_B: False,
            },
        )

        events = book.all_events

        self.assertEqual(
            len(events),
            2,
        )

        self.assertNotEqual(
            events[0].incident_event_id,
            events[1].incident_event_id,
        )

        self.assertEqual(
            events[0].ordinal,
            1,
        )

        self.assertEqual(
            events[1].ordinal,
            2,
        )

    def test_initial_compromise_starts_at_reset_tick(
        self,
    ):
        book = make_bookkeeper(
            initial_a=True
        )

        tick = book.record_tick(
            global_tick_end=1,
            red_presence_after={
                HOST_A: True,
                HOST_B: False,
            },
        )

        event = book.all_events[0]

        self.assertEqual(
            event.t_compromise,
            0,
        )

        self.assertEqual(
            tick.incident_active_ticks,
            1,
        )

    def test_unknown_lwf_host_rejected(
        self,
    ):
        book = make_bookkeeper()

        with self.assertRaises(
            ValueError
        ):
            book.record_tick(
                global_tick_end=1,
                red_presence_after={
                    HOST_A: False,
                    HOST_B: False,
                },
                local_work_failures=[
                    LocalWorkFailure(
                        "unknown_host",
                        -1.0,
                    )
                ],
            )

    def test_host_scope_cannot_change(
        self,
    ):
        book = make_bookkeeper()

        with self.assertRaises(
            ValueError
        ):
            book.record_tick(
                global_tick_end=1,
                red_presence_after={
                    HOST_A: False,
                },
            )

    def test_tick_must_advance_exactly_one(
        self,
    ):
        book = make_bookkeeper()

        with self.assertRaises(
            ValueError
        ):
            book.record_tick(
                global_tick_end=2,
                red_presence_after={
                    HOST_A: False,
                    HOST_B: False,
                },
            )

    def test_custom_reward_weights(
        self,
    ):
        config = ResponseRewardConfig(
            lambda_time=0.5,
            lambda_failure=2.0,
        )

        reward = config.compute(
            incident_active_ticks=2,
            incident_host_lwf_raw_penalty=-3.0,
        )

        self.assertEqual(
            reward,
            -7.0,
        )

    def test_accounting_exposes_replay_kwargs(
        self,
    ):
        book = make_bookkeeper()

        tick = book.record_tick(
            global_tick_end=1,
            red_presence_after={
                HOST_A: True,
                HOST_B: False,
            },
        )

        payload = (
            tick.to_replay_kwargs()
        )

        self.assertIn(
            "incident_event_ids",
            payload,
        )

        self.assertIn(
            "incident_host_ids",
            payload,
        )

        self.assertIn(
            "response_reward",
            payload,
        )


if __name__ == "__main__":
    unittest.main()