import unittest
from types import SimpleNamespace

import numpy as np

from formal_experiments.evaluation.audit_world_model_final import (
    ACTION_NAMES,
    FROZEN_REFERENCE,
    _bucket_metrics,
    b0_1_gate,
    completed_action_counts,
)


class TestGateB0WorldModelFinalAudit(unittest.TestCase):

    def _passing_actions(self):
        return {
            "no_op": {"count": 1600, "rmse_over_persistence": 0.70, "beats_persistence": True},
            "analyse": {"count": 214, "rmse_over_persistence": 0.90, "beats_persistence": True},
            "remove": {"count": 233, "rmse_over_persistence": 0.95, "beats_persistence": True},
            "restore": {"count": 248, "rmse_over_persistence": 1.10, "beats_persistence": False},
        }

    def _passing_h4(self):
        return {
            name: {"count": 200, "rmse_over_persistence": 1.0}
            for name in ACTION_NAMES
        }

    def test_action_names_are_exact_a4_contract(self):
        self.assertEqual(ACTION_NAMES, ("no_op", "analyse", "remove", "restore"))

    def test_gate_accepts_predeclared_passing_case(self):
        result = b0_1_gate(
            validation_per_action=self._passing_actions(),
            first_action_h4=self._passing_h4(),
            aggregate=dict(FROZEN_REFERENCE),
        )
        self.assertTrue(result["pass"])
        self.assertEqual(result["actions_beating_persistence"], 3)
        self.assertTrue(result["frozen_aggregate_reproduction_ok"])

    def test_gate_rejects_low_validation_action_coverage(self):
        actions = self._passing_actions()
        actions["restore"] = dict(actions["restore"], count=149)
        result = b0_1_gate(
            validation_per_action=actions,
            first_action_h4=self._passing_h4(),
            aggregate=dict(FROZEN_REFERENCE),
        )
        self.assertFalse(result["pass"])
        self.assertFalse(result["actions"]["restore"]["count_ok"])

    def test_gate_rejects_catastrophic_action_ratio(self):
        actions = self._passing_actions()
        actions["restore"] = dict(actions["restore"], rmse_over_persistence=1.251)
        result = b0_1_gate(
            validation_per_action=actions,
            first_action_h4=self._passing_h4(),
            aggregate=dict(FROZEN_REFERENCE),
        )
        self.assertFalse(result["pass"])
        self.assertFalse(result["actions"]["restore"]["ratio_ok"])

    def test_gate_requires_three_of_four_actions_to_beat_persistence(self):
        actions = self._passing_actions()
        actions["remove"] = dict(actions["remove"], beats_persistence=False)
        result = b0_1_gate(
            validation_per_action=actions,
            first_action_h4=self._passing_h4(),
            aggregate=dict(FROZEN_REFERENCE),
        )
        self.assertFalse(result["pass"])
        self.assertEqual(result["actions_beating_persistence"], 2)

    def test_gate_rejects_frozen_metric_drift(self):
        aggregate = dict(FROZEN_REFERENCE)
        aggregate["h4_rmse"] += 0.001
        result = b0_1_gate(
            validation_per_action=self._passing_actions(),
            first_action_h4=self._passing_h4(),
            aggregate=aggregate,
        )
        self.assertFalse(result["pass"])
        self.assertFalse(
            result["frozen_aggregate_reproduction"]["h4_rmse"]["within_tolerance"]
        )

    def test_bucket_metrics_compute_action_specific_persistence_and_calibration(self):
        target = np.asarray([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
        predicted = target + 0.1
        persistence = target + 0.2
        uncertainty = np.asarray([0.1, 0.2, 0.3, 0.4])
        result = _bucket_metrics(
            predicted=predicted,
            target=target,
            persistence=persistence,
            uncertainty=uncertainty,
        )
        self.assertEqual(result["count"], 4)
        self.assertAlmostEqual(result["rmse"], 0.1, places=6)
        self.assertAlmostEqual(result["persistence_rmse"], 0.2, places=6)
        self.assertAlmostEqual(result["rmse_over_persistence"], 0.5, places=6)
        self.assertTrue(result["beats_persistence"])
        self.assertEqual(len(result["uncertainty_quartiles"]), 4)

    def test_completed_action_counts_preserve_all_four_buckets(self):
        rows = [
            SimpleNamespace(action_completed=True, executed_action_family="Sleep"),
            SimpleNamespace(action_completed=True, executed_action_family="Analyse"),
            SimpleNamespace(action_completed=True, executed_action_family="Remove"),
            SimpleNamespace(action_completed=True, executed_action_family="Restore"),
            SimpleNamespace(action_completed=False, executed_action_family="Restore"),
        ]
        counts = completed_action_counts(rows)
        self.assertEqual(counts, {
            "no_op": 1,
            "analyse": 1,
            "remove": 1,
            "restore": 1,
        })


if __name__ == "__main__":
    unittest.main()
