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
    reset 中故意保留正常 Processes。

    reset Processes 是 baseline，
    不得作为 threat evidence。
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
    # 1
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
    # 2
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
    # 3
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
    # 4
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
    # 5
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
    # 6
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
                                    "local_port":
                                        22,
                                },
                                {
                                    "local_port":
                                        80,
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
    # 7
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
    # 8
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
    # 9
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
    # 10
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
            tracker
            .current_stats
            .evidence_hosts,
            tuple(),
        )

    # ========================================================
    # 11
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
            tracker
            .observable_host_scores(),
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
            tracker
            .observable_host_scores(),
        )

    # ========================================================
    # 12
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
            tracker
            .observable_host_scores(),
        )

    # ========================================================
    # 13
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
            scores[
                "host_b"
            ],

            scores[
                "host_a"
            ],
        )

    # ========================================================
    # 14
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
            tracker
            .ranked_host_evidence(
                global_tick=1,
            )
        )

        self.assertEqual(
            ranked[
                0
            ][
                "hostname"
            ],
            "host_a",
        )

        self.assertEqual(
            ranked[
                0
            ][
                "process_events"
            ],
            1,
        )

    # ========================================================
    # 15
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

            any_valid_observable_target=False,
        )

        self.assertEqual(
            vector.shape,
            (
                27,
            ),
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
    # 16
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

            any_valid_observable_target=False,
        )

        np.testing.assert_array_equal(
            vector[
                :5
            ],

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
    # 17
    # ========================================================

    def test_reset_state_has_no_valid_observable_target(
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

            any_valid_observable_target=False,
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
                    "any_valid_observable_target"
                ]
            ],
            0.0,
        )

    # ========================================================
    # 18
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

            any_valid_observable_target=True,
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
                    "any_valid_observable_target"
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
    # 19
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

                    any_valid_observable_target=False,
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
    # 20
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

            any_valid_observable_target=True,
        )

        b = encoder.encode(
            tracker=tracker,

            observation=observation,

            global_tick=2,

            episode_steps=100,

            any_valid_observable_target=True,
        )

        np.testing.assert_array_equal(
            a,
            b,
        )

    # ========================================================
    # 21
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
                    "success":
                        "UNKNOWN",
                },

                global_tick=4,
            )

    # ========================================================
    # 22
    #
    # A4.1c regression:
    # evidence != valid action target
    # ========================================================

    def test_observable_evidence_does_not_imply_valid_target(
        self,
    ):
        tracker = (
            self.make_tracker()
        )

        observation = {
            "success": "UNKNOWN",

            "host_a": {
                "Processes": [
                    {
                        "PID": 1,
                    }
                ]
            },
        }

        tracker.update(
            observation=observation,

            global_tick=1,
        )

        self.assertTrue(
            tracker
            .observable_host_scores(
                1
            )
        )

        encoder = (
            FormalStateEncoder()
        )

        vector = encoder.encode(
            tracker=tracker,

            observation=observation,

            global_tick=1,

            episode_steps=100,

            any_valid_observable_target=False,
        )

        index = (
            FORMAL_STATE_FEATURE_NAMES
            .index(
                "any_valid_observable_target"
            )
        )

        self.assertEqual(
            vector[
                index
            ],
            0.0,
        )

    # ========================================================
    # 23
    # ========================================================

    def test_valid_observable_target_feature(
        self,
    ):
        tracker = (
            self.make_tracker()
        )

        observation = {
            "success": "UNKNOWN",

            "host_a": {
                "Processes": [
                    {
                        "PID": 1,
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

            any_valid_observable_target=True,
        )

        index = (
            FORMAL_STATE_FEATURE_NAMES
            .index(
                "any_valid_observable_target"
            )
        )

        self.assertEqual(
            vector[
                index
            ],
            1.0,
        )

    # ========================================================
    # 24
    # ========================================================

    def test_valid_target_requires_observable_evidence(
        self,
    ):
        tracker = (
            self.make_tracker()
        )

        encoder = (
            FormalStateEncoder()
        )

        with self.assertRaises(
            ValueError
        ):
            encoder.encode(
                tracker=tracker,

                observation={
                    "success":
                        "UNKNOWN",
                },

                global_tick=0,

                episode_steps=100,

                any_valid_observable_target=True,
            )


if __name__ == "__main__":
    unittest.main()