import unittest

import numpy as np

from shared.formal_state import (
    FORMAL_STATE_DIM,
    FORMAL_STATE_FEATURE_NAMES,
    FormalStateEncoder,
    ObservableHostEvidenceTracker,
)


def reset_obs():
    """
    特意让 reset 中存在正常 Processes。

    A4.1a 已真实证明：
    reset Processes 是 baseline，
    不能算 threat evidence。
    """

    return {
        "success": "UNKNOWN",

        "host_a": {
            "System info": {
                "Hostname": "host_a",
            },

            "Interface": [
                {
                    "ip_address":
                        "10.0.0.1",
                }
            ],

            "Processes": [
                {
                    "PID": 100,
                    "username": "user",
                }
            ],
        },

        "host_b": {
            "System info": {
                "Hostname": "host_b",
            },

            "Interface": [
                {
                    "ip_address":
                        "10.0.0.2",
                }
            ],

            "Processes": [
                {
                    "PID": 200,
                    "username": "user",
                }
            ],
        },

        "host_c": {
            "System info": {
                "Hostname": "host_c",
            },

            "Interface": [
                {
                    "ip_address":
                        "10.0.0.3",
                }
            ],
        },
    }


class TestGateAFormalState(
    unittest.TestCase
):

    def make_tracker(
        self,
        agent="blue_agent_0",
    ):
        tracker = (
            ObservableHostEvidenceTracker(
                agent
            )
        )

        tracker.reset(
            reset_obs()
        )

        return tracker

    # ========================================================
    # 1. dimension frozen
    # ========================================================

    def test_state_dimension_is_27(
        self,
    ):
        self.assertEqual(
            FORMAL_STATE_DIM,
            27,
        )

        self.assertEqual(
            len(
                FORMAL_STATE_FEATURE_NAMES
            ),
            27,
        )

    # ========================================================
    # 2. reset builds inventory
    # ========================================================

    def test_reset_builds_inventory(
        self,
    ):
        tracker = (
            self.make_tracker()
        )

        self.assertEqual(
            tracker.inventory,
            (
                "host_a",
                "host_b",
                "host_c",
            ),
        )

        self.assertEqual(
            tracker.inventory_size,
            3,
        )

    # ========================================================
    # 3. IP mapping
    # ========================================================

    def test_reset_builds_ip_mapping(
        self,
    ):
        tracker = (
            self.make_tracker()
        )

        self.assertEqual(
            tracker.canonical_hostname(
                "10.0.0.2"
            ),
            "host_b",
        )

    # ========================================================
    # 4. baseline Processes != threat
    # ========================================================

    def test_reset_processes_are_not_evidence(
        self,
    ):
        tracker = (
            self.make_tracker()
        )

        self.assertEqual(
            tracker.observable_host_scores(),
            {},
        )

        self.assertEqual(
            tracker.suspicious_hosts(),
            tuple(),
        )

        self.assertEqual(
            tracker
            .evidence_for(
                "host_a"
            )
            .process_events,
            0,
        )

    # ========================================================
    # 5. post-reset Process evidence
    # ========================================================

    def test_post_reset_process_event_counted(
        self,
    ):
        tracker = (
            self.make_tracker()
        )

        tracker.update(
            observation={
                "success": "UNKNOWN",

                "host_a": {
                    "Processes": [
                        {
                            "PID": 999,
                        }
                    ],
                },
            },
            global_tick=1,
        )

        evidence = (
            tracker.evidence_for(
                "host_a"
            )
        )

        self.assertEqual(
            evidence.process_events,
            1,
        )

        self.assertEqual(
            evidence.connection_events,
            0,
        )

        self.assertEqual(
            evidence.observation_hits,
            1,
        )

    # ========================================================
    # 6. connection evidence
    # ========================================================

    def test_connection_event_counted(
        self,
    ):
        tracker = (
            self.make_tracker()
        )

        tracker.update(
            observation={
                "success": "UNKNOWN",

                "host_b": {
                    "Processes": [
                        {
                            "Connections": [
                                {
                                    "local_port": 22,
                                },
                                {
                                    "local_port": 80,
                                },
                            ]
                        }
                    ]
                },
            },
            global_tick=1,
        )

        evidence = (
            tracker.evidence_for(
                "host_b"
            )
        )

        self.assertEqual(
            evidence.process_events,
            1,
        )

        self.assertEqual(
            evidence.connection_events,
            2,
        )

    # ========================================================
    # 7. Analyse Files
    # ========================================================

    def test_file_evidence_counted(
        self,
    ):
        tracker = (
            self.make_tracker()
        )

        tracker.update(
            observation={
                "success": "TRUE",

                "host_c": {
                    "Files": [
                        {
                            "File Name":
                                "bad.bin",
                        }
                    ]
                },
            },
            global_tick=2,
        )

        evidence = (
            tracker.evidence_for(
                "host_c"
            )
        )

        self.assertEqual(
            evidence.file_events,
            1,
        )

        self.assertTrue(
            evidence.has_file()
        )

    # ========================================================
    # 8. unknown host rejected from evidence
    # ========================================================

    def test_unknown_host_not_added(
        self,
    ):
        tracker = (
            self.make_tracker()
        )

        tracker.update(
            observation={
                "success": "TRUE",

                "unknown_host": {
                    "Files": [
                        {
                            "File Name":
                                "x",
                        }
                    ]
                },
            },
            global_tick=1,
        )

        self.assertEqual(
            tracker.observable_host_scores(),
            {},
        )

        self.assertNotIn(
            "unknown_host",
            tracker.inventory,
        )

    # ========================================================
    # 9. IP observation canonicalized
    # ========================================================

    def test_ip_observation_maps_to_hostname(
        self,
    ):
        tracker = (
            self.make_tracker()
        )

        tracker.update(
            observation={
                "success": "UNKNOWN",

                "10.0.0.1": {
                    "Processes": [
                        {
                            "PID": 1,
                        }
                    ]
                },
            },
            global_tick=1,
        )

        scores = (
            tracker
            .observable_host_scores()
        )

        self.assertIn(
            "host_a",
            scores,
        )

        self.assertNotIn(
            "10.0.0.1",
            scores,
        )

    # ========================================================
    # 10. evidence persists
    # ========================================================

    def test_evidence_persists_across_quiet_step(
        self,
    ):
        tracker = (
            self.make_tracker()
        )

        tracker.update(
            observation={
                "success": "UNKNOWN",

                "host_a": {
                    "Processes": [
                        {
                            "PID": 3,
                        }
                    ]
                },
            },
            global_tick=1,
        )

        tracker.update(
            observation={
                "success": "UNKNOWN",
            },
            global_tick=2,
        )

        scores = (
            tracker
            .observable_host_scores()
        )

        self.assertIn(
            "host_a",
            scores,
        )

        self.assertEqual(
            tracker.current_stats
            .evidence_hosts,
            tuple(),
        )

    # ========================================================
    # 11. Restore success clears old evidence
    # ========================================================

    def test_successful_restore_clears_target(
        self,
    ):
        tracker = (
            self.make_tracker()
        )

        tracker.update(
            observation={
                "success": "UNKNOWN",

                "host_a": {
                    "Processes": [
                        {
                            "PID": 9,
                        }
                    ]
                },
            },
            global_tick=1,
        )

        self.assertIn(
            "host_a",
            tracker.observable_host_scores(),
        )

        tracker.update(
            observation={
                "success": "TRUE",
            },
            global_tick=6,
            completed_action_family=(
                "Restore"
            ),
            completed_target_host=(
                "host_a"
            ),
            completed_action_success=True,
        )

        self.assertNotIn(
            "host_a",
            tracker.observable_host_scores(),
        )

    # ========================================================
    # 12. Remove does not imply normal
    # ========================================================

    def test_remove_success_does_not_clear_evidence(
        self,
    ):
        tracker = (
            self.make_tracker()
        )

        tracker.update(
            observation={
                "success": "UNKNOWN",

                "host_a": {
                    "Processes": [
                        {
                            "PID": 9,
                        }
                    ]
                },
            },
            global_tick=1,
        )

        tracker.update(
            observation={
                "success": "TRUE",
            },
            global_tick=4,
            completed_action_family=(
                "Remove"
            ),
            completed_target_host=(
                "host_a"
            ),
            completed_action_success=True,
        )

        self.assertIn(
            "host_a",
            tracker.observable_host_scores(),
        )

    # ========================================================
    # 13. stronger evidence ranks higher
    # ========================================================

    def test_file_evidence_scores_above_process_only(
        self,
    ):
        tracker = (
            self.make_tracker()
        )

        tracker.update(
            observation={
                "success": "UNKNOWN",

                "host_a": {
                    "Processes": [
                        {
                            "PID": 1,
                        }
                    ]
                },

                "host_b": {
                    "Files": [
                        {
                            "File Name":
                                "suspicious",
                        }
                    ]
                },
            },
            global_tick=1,
        )

        scores = (
            tracker
            .observable_host_scores()
        )

        self.assertGreater(
            scores["host_b"],
            scores["host_a"],
        )

    # ========================================================
    # 14. ranked observable summary
    # ========================================================

    def test_ranked_host_evidence(
        self,
    ):
        tracker = (
            self.make_tracker()
        )

        tracker.update(
            observation={
                "success": "UNKNOWN",

                "host_a": {
                    "Processes": [
                        {
                            "PID": 1,
                        }
                    ]
                },
            },
            global_tick=1,
        )

        ranked = (
            tracker.ranked_host_evidence(
                global_tick=1,
            )
        )

        self.assertEqual(
            ranked[0][
                "hostname"
            ],
            "host_a",
        )

        self.assertEqual(
            ranked[0][
                "process_events"
            ],
            1,
        )

    # ========================================================
    # 15. state shape / dtype / finite
    # ========================================================

    def test_formal_state_shape_dtype_finite(
        self,
    ):
        tracker = (
            self.make_tracker()
        )

        encoder = (
            FormalStateEncoder()
        )

        observation = {
            "success": "UNKNOWN",
        }

        vector = encoder.encode(
            tracker=tracker,
            observation=observation,
            global_tick=0,
            episode_steps=100,
        )

        self.assertEqual(
            vector.shape,
            (27,),
        )

        self.assertEqual(
            vector.dtype,
            np.float32,
        )

        self.assertTrue(
            np.all(
                np.isfinite(
                    vector
                )
            )
        )

    # ========================================================
    # 16. agent one-hot
    # ========================================================

    def test_agent_identity_one_hot(
        self,
    ):
        tracker = (
            self.make_tracker(
                "blue_agent_3"
            )
        )

        encoder = (
            FormalStateEncoder()
        )

        vector = encoder.encode(
            tracker=tracker,
            observation={
                "success": "UNKNOWN",
            },
            global_tick=0,
            episode_steps=100,
        )

        np.testing.assert_array_equal(
            vector[:5],
            np.asarray(
                [
                    0,
                    0,
                    0,
                    1,
                    0,
                ],
                dtype=np.float32,
            ),
        )

    # ========================================================
    # 17. reset state no threat
    # ========================================================

    def test_reset_state_has_no_observable_target(
        self,
    ):
        tracker = (
            self.make_tracker()
        )

        encoder = (
            FormalStateEncoder()
        )

        vector = encoder.encode(
            tracker=tracker,
            observation={
                "success": "UNKNOWN",
            },
            global_tick=0,
            episode_steps=100,
        )

        name_to_index = {
            name: idx
            for idx, name
            in enumerate(
                FORMAL_STATE_FEATURE_NAMES
            )
        }

        self.assertEqual(
            vector[
                name_to_index[
                    "suspicious_host_fraction"
                ]
            ],
            0.0,
        )

        self.assertEqual(
            vector[
                name_to_index[
                    "any_observable_target"
                ]
            ],
            0.0,
        )

    # ========================================================
    # 18. evidence changes formal state
    # ========================================================

    def test_observable_evidence_changes_state(
        self,
    ):
        tracker = (
            self.make_tracker()
        )

        observation = {
            "success": "UNKNOWN",

            "host_b": {
                "Processes": [
                    {
                        "Connections": [
                            {
                                "remote_port":
                                    4444,
                            }
                        ]
                    }
                ]
            },
        }

        tracker.update(
            observation=observation,
            global_tick=1,
        )

        encoder = (
            FormalStateEncoder()
        )

        vector = encoder.encode(
            tracker=tracker,
            observation=observation,
            global_tick=1,
            episode_steps=100,
        )

        name_to_index = {
            name: idx
            for idx, name
            in enumerate(
                FORMAL_STATE_FEATURE_NAMES
            )
        }

        self.assertGreater(
            vector[
                name_to_index[
                    "connection_evidence_host_fraction"
                ]
            ],
            0.0,
        )

        self.assertEqual(
            vector[
                name_to_index[
                    "any_observable_target"
                ]
            ],
            1.0,
        )

        self.assertEqual(
            vector[
                name_to_index[
                    "top_host_has_connection"
                ]
            ],
            1.0,
        )

    # ========================================================
    # 19. four success states
    # ========================================================

    def test_success_one_hot_contract(
        self,
    ):
        tracker = (
            self.make_tracker()
        )

        encoder = (
            FormalStateEncoder()
        )

        names = [
            "success_false",
            "success_true",
            "success_unknown",
            "success_in_progress",
        ]

        name_to_index = {
            name: idx
            for idx, name
            in enumerate(
                FORMAL_STATE_FEATURE_NAMES
            )
        }

        cases = {
            "FALSE": 0,
            "TRUE": 1,
            "UNKNOWN": 2,
            "IN_PROGRESS": 3,
        }

        for raw, expected_index in (
            cases.items()
        ):
            with self.subTest(
                raw=raw
            ):
                vector = encoder.encode(
                    tracker=tracker,
                    observation={
                        "success": raw,
                    },
                    global_tick=0,
                    episode_steps=100,
                )

                got = [
                    vector[
                        name_to_index[
                            name
                        ]
                    ]
                    for name
                    in names
                ]

                expected = [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]

                expected[
                    expected_index
                ] = 1.0

                self.assertEqual(
                    got,
                    expected,
                )

    # ========================================================
    # 20. deterministic
    # ========================================================

    def test_encoder_is_deterministic(
        self,
    ):
        tracker = (
            self.make_tracker()
        )

        observation = {
            "success": "TRUE",

            "host_a": {
                "Files": [
                    {
                        "File Name":
                            "x",
                    }
                ]
            },
        }

        tracker.update(
            observation=observation,
            global_tick=2,
        )

        encoder = (
            FormalStateEncoder()
        )

        a = encoder.encode(
            tracker=tracker,
            observation=observation,
            global_tick=2,
            episode_steps=100,
        )

        b = encoder.encode(
            tracker=tracker,
            observation=observation,
            global_tick=2,
            episode_steps=100,
        )

        np.testing.assert_array_equal(
            a,
            b,
        )

    # ========================================================
    # 21. monotonic global tick
    # ========================================================

    def test_nonmonotonic_tick_rejected(
        self,
    ):
        tracker = (
            self.make_tracker()
        )

        tracker.update(
            observation={
                "success": "UNKNOWN",
            },
            global_tick=5,
        )

        with self.assertRaises(
            ValueError
        ):
            tracker.update(
                observation={
                    "success": "UNKNOWN",
                },
                global_tick=4,
            )


if __name__ == "__main__":
    unittest.main()