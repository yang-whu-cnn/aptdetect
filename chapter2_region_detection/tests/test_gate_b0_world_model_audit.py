import unittest
from types import SimpleNamespace

from formal_experiments.evaluation.audit_world_model_final import (
    ACTION_NAMES,
    FROZEN_METRIC_ABS_TOLERANCE,
    FROZEN_REFERENCE,
    MAX_ACTION_RMSE_PERSISTENCE_RATIO,
    MAX_H4_FIRST_ACTION_RATIO_BEFORE_REVIEW,
    MIN_ACTIONS_BEATING_PERSISTENCE,
    MIN_VALIDATION_COUNT_PER_ACTION,
    b0_1_gate,
    completed_action_counts,
    h4_catastrophic_review,
)


def good_one_step():
    return {
        name: {
            "count": 200,
            "rmse_over_persistence": 0.90,
            "beats_persistence": True,
        }
        for name in ACTION_NAMES
    }


def good_h4():
    return {
        name: {
            "count": 100,
            "rmse_over_persistence": 1.10,
        }
        for name in ACTION_NAMES
    }


def frozen_aggregate():
    return dict(FROZEN_REFERENCE)


class TestGateB0WorldModelAudit(unittest.TestCase):

    def test_frozen_gate_constants(self):
        self.assertEqual(MIN_VALIDATION_COUNT_PER_ACTION, 150)
        self.assertEqual(MAX_ACTION_RMSE_PERSISTENCE_RATIO, 1.25)
        self.assertEqual(MIN_ACTIONS_BEATING_PERSISTENCE, 3)
        self.assertEqual(MAX_H4_FIRST_ACTION_RATIO_BEFORE_REVIEW, 2.0)
        self.assertEqual(FROZEN_METRIC_ABS_TOLERANCE, 5e-4)
        self.assertEqual(ACTION_NAMES, ("no_op", "analyse", "remove", "restore"))

    def test_gate_passes_only_when_all_hard_conditions_pass(self):
        result = b0_1_gate(
            validation_per_action=good_one_step(),
            first_action_h4=good_h4(),
            aggregate=frozen_aggregate(),
        )
        self.assertTrue(result["pass"])
        self.assertFalse(
            result["h4_catastrophic_review"]["manual_review_required"]
        )
        self.assertEqual(result["actions_beating_persistence"], 4)

    def test_gate_fails_insufficient_action_count(self):
        one = good_one_step()
        one["restore"] = dict(one["restore"], count=149)
        result = b0_1_gate(
            validation_per_action=one,
            first_action_h4=good_h4(),
            aggregate=frozen_aggregate(),
        )
        self.assertFalse(result["pass"])
        self.assertFalse(result["all_actions_have_min_count"])

    def test_gate_fails_action_ratio_above_limit(self):
        one = good_one_step()
        one["remove"] = dict(one["remove"], rmse_over_persistence=1.250001)
        result = b0_1_gate(
            validation_per_action=one,
            first_action_h4=good_h4(),
            aggregate=frozen_aggregate(),
        )
        self.assertFalse(result["pass"])
        self.assertFalse(result["all_actions_ratio_within_limit"])

    def test_gate_requires_three_actions_to_beat_persistence(self):
        one = good_one_step()
        one["analyse"] = dict(one["analyse"], beats_persistence=False, rmse_over_persistence=1.10)
        one["restore"] = dict(one["restore"], beats_persistence=False, rmse_over_persistence=1.10)
        result = b0_1_gate(
            validation_per_action=one,
            first_action_h4=good_h4(),
            aggregate=frozen_aggregate(),
        )
        self.assertFalse(result["pass"])
        self.assertEqual(result["actions_beating_persistence"], 2)

    def test_gate_fails_frozen_metric_reproduction_outside_tolerance(self):
        aggregate = frozen_aggregate()
        aggregate["h4_rmse"] += FROZEN_METRIC_ABS_TOLERANCE + 1e-6
        result = b0_1_gate(
            validation_per_action=good_one_step(),
            first_action_h4=good_h4(),
            aggregate=aggregate,
        )
        self.assertFalse(result["pass"])
        self.assertFalse(result["frozen_aggregate_reproduction_ok"])

    def test_h4_catastrophic_ratio_requires_review_and_blocks_pass(self):
        h4 = good_h4()
        h4["restore"] = dict(
            h4["restore"],
            rmse_over_persistence=MAX_H4_FIRST_ACTION_RATIO_BEFORE_REVIEW + 0.01,
        )
        review = h4_catastrophic_review(h4)
        self.assertTrue(review["manual_review_required"])
        self.assertTrue(review["actions"]["restore"]["catastrophic"])
        result = b0_1_gate(
            validation_per_action=good_one_step(),
            first_action_h4=h4,
            aggregate=frozen_aggregate(),
        )
        self.assertFalse(result["pass"])

    def test_h4_missing_bucket_requires_review(self):
        h4 = good_h4()
        h4["analyse"] = {"count": 0, "missing": True}
        review = h4_catastrophic_review(h4)
        self.assertTrue(review["manual_review_required"])
        self.assertTrue(review["actions"]["analyse"]["missing"])

    def test_completed_action_counts_ignores_incomplete(self):
        rows = [
            SimpleNamespace(action_completed=True, executed_action_family="Sleep"),
            SimpleNamespace(action_completed=True, executed_action_family="Analyse"),
            SimpleNamespace(action_completed=True, executed_action_family="Remove"),
            SimpleNamespace(action_completed=True, executed_action_family="Restore"),
            SimpleNamespace(action_completed=False, executed_action_family="Restore"),
        ]
        counts = completed_action_counts(rows)
        self.assertEqual(
            counts,
            {"no_op": 1, "analyse": 1, "remove": 1, "restore": 1},
        )


if __name__ == "__main__":
    unittest.main()
