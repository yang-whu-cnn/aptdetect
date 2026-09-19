import unittest

from formal_experiments.evaluation.aggregate_tables import aggregate_five_repeats
from formal_experiments.evaluation.metrics_v3 import (
    aggregate_repeat,
    compute_episode_metrics,
    team_reward_once,
)


def episode(
    *,
    reward=1.0,
    failures=None,
    actions=None,
    incidents=None,
    seed=4000,
):
    return {
        "schema_version": 1,
        "protocol_version": "cc4_v3_20260917",
        "episode_seed": seed,
        "tick_count": 500,
        "episode_end_tick": 500,
        "tick_team_rewards": [
            {f"blue_agent_{index}": reward for index in range(5)} for _ in range(500)
        ],
        "operation_failure_events": failures or [],
        "recovery_actions": actions or [],
        "incidents": incidents or [],
    }


class TestMetricsV3Golden(unittest.TestCase):
    def test_five_duplicate_agent_rewards_count_once_per_tick(self):
        metrics = compute_episode_metrics(episode(reward=-2.0))
        self.assertEqual(metrics.cc4_official_reward, -1000.0)

    def test_disagreeing_agent_rewards_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "disagree"):
            team_reward_once({"blue_agent_0": -1, "blue_agent_1": -2})

    def test_operation_failure_penalty_keeps_signed_audit_sum(self):
        metrics = compute_episode_metrics(episode(failures=[
            {"raw_penalty": -3.5, "agent_name": "blue_agent_0", "host": "host-a"},
            {"raw_penalty": -1, "agent_name": "blue_agent_1", "host": "host-b"},
            {"raw_penalty": 2, "scope": "all_blue"},
        ]))
        self.assertEqual(metrics.operation_failure_signed_sum, -4.5)
        self.assertEqual(metrics.operation_failure_penalty, 4.5)
        self.assertEqual(metrics.operation_failure_count, 2)

    def test_recovery_precision_counts_only_started_completed_remove_restore(self):
        actions = [
            {"executed_action": "remove", "started": True, "completed": True,
             "fallback": False, "active_incident_before": True},
            {"executed_action": "restore", "started": True, "completed": True,
             "fallback": False, "active_incident_before": False},
            {"executed_action": "remove", "started": True, "completed": False,
             "fallback": False, "active_incident_before": True},
            {"executed_action": "sleep", "started": True, "completed": True,
             "fallback": True, "active_incident_before": True},
        ]
        metrics = compute_episode_metrics(episode(actions=actions))
        self.assertEqual(metrics.recovery_true_positives, 1)
        self.assertEqual(metrics.recovery_false_positives, 1)
        self.assertEqual(metrics.recovery_precision, 0.5)

    def test_precision_is_na_when_repeat_has_no_qualifying_recovery(self):
        self.assertIsNone(aggregate_repeat([episode()]).recovery_precision)

    def test_recovery_time_uses_tick_500_censor_and_reports_auxiliaries(self):
        metrics = compute_episode_metrics(
            episode(incidents=[
                {"incident_id": "done", "t_compromise": 10, "t_recovered": 30},
                {"incident_id": "open", "t_compromise": 450, "t_recovered": None},
            ])
        )
        self.assertEqual(metrics.recovery_time_censored_mean, 35.0)
        self.assertEqual(metrics.recovery_time_completed_only_mean, 20.0)
        self.assertEqual(metrics.recovery_unrecovered_rate, 0.5)

    def test_incident_starting_at_episode_end_has_zero_censored_duration(self):
        metrics = compute_episode_metrics(
            episode(incidents=[{"incident_id": "edge", "t_compromise": 500, "t_recovered": None}])
        )
        self.assertEqual(metrics.recovery_time_censored_mean, 0.0)
        self.assertEqual(metrics.recovery_unrecovered_rate, 1.0)

    def test_repeat_precision_and_time_are_micro_averages(self):
        one = episode(
            actions=[{"executed_action": "remove", "started": True, "completed": True,
                      "fallback": False, "active_incident_before": True}],
            incidents=[{"t_compromise": 0, "t_recovered": 100}],
        )
        two = episode(
            actions=[
                {"executed_action": "restore", "started": True, "completed": True,
                 "fallback": False, "active_incident_before": False},
                {"executed_action": "restore", "started": True, "completed": True,
                 "fallback": False, "active_incident_before": False},
            ],
            incidents=[
                {"t_compromise": 400, "t_recovered": None},
                {"t_compromise": 450, "t_recovered": None},
            ],
        )
        metrics = aggregate_repeat([one, two])
        self.assertAlmostEqual(metrics.recovery_precision, 1 / 3)
        self.assertAlmostEqual(metrics.recovery_time_censored_mean, 250 / 3)
        self.assertAlmostEqual(metrics.recovery_time_completed_only_mean, 100)
        self.assertAlmostEqual(metrics.recovery_unrecovered_rate, 2 / 3)

    def test_five_repeat_aggregation_uses_sample_standard_deviation(self):
        rows = []
        for repeat_index, value in enumerate((1, 2, 3, 4, 5), start=1):
            rows.append({
                "repeat_index": repeat_index,
                "training_seed": 51000 + repeat_index,
                "method": "golden",
                "run_mode": "formal",
                "formal_result_eligible": True,
                "cc4_official_reward": value,
                "operation_failure_penalty": value,
                "recovery_precision": value / 10,
                "recovery_time_censored_mean": value * 10,
                "recovery_time_completed_only_mean": value * 8,
                "recovery_unrecovered_rate": value / 10,
            })
        result = aggregate_five_repeats(rows)
        self.assertEqual(result["ddof"], 1)
        self.assertAlmostEqual(result["metrics"]["cc4_official_reward"]["mean"], 3)
        self.assertAlmostEqual(
            result["metrics"]["cc4_official_reward"]["std"], 1.5811388300841898
        )

    def test_five_repeat_aggregation_rejects_duplicate_and_seed_mismatch(self):
        rows = []
        for repeat_index in range(1, 6):
            rows.append({
                "repeat_index": repeat_index,
                "training_seed": 51000 + repeat_index,
                "run_mode": "formal",
                "formal_result_eligible": True,
                "cc4_official_reward": 0,
                "operation_failure_penalty": 0,
                "recovery_precision": 1,
                "recovery_time_censored_mean": 1,
                "recovery_time_completed_only_mean": 1,
                "recovery_unrecovered_rate": 0,
            })
        duplicate = [dict(row) for row in rows]
        duplicate[4]["repeat_index"] = 4
        with self.assertRaisesRegex(ValueError, "identities"):
            aggregate_five_repeats(duplicate)
        mismatch = [dict(row) for row in rows]
        mismatch[2]["training_seed"] = 51005
        with self.assertRaisesRegex(ValueError, "identities"):
            aggregate_five_repeats(mismatch)

    def test_five_repeat_aggregation_rejects_dev_rows(self):
        rows = []
        for repeat_index in range(1, 6):
            rows.append({
                "repeat_index": repeat_index, "training_seed": 51000 + repeat_index,
                "run_mode": "dev", "formal_result_eligible": False,
                "cc4_official_reward": 0, "operation_failure_penalty": 0,
                "recovery_precision": 1, "recovery_time_censored_mean": 1,
                "recovery_time_completed_only_mean": 1, "recovery_unrecovered_rate": 0,
            })
        with self.assertRaisesRegex(ValueError, "development or ineligible"):
            aggregate_five_repeats(rows)


if __name__ == "__main__":
    unittest.main()
