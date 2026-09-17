import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from formal_experiments.evaluation.audit_b0_2_policy_ood import (
    KNN_K,
    MAX_OOD_FRACTION,
    MAX_PROBE_VALIDATION_RMSE_RATIO,
    VALIDATION_PERCENTILE,
    _h4_windows,
    _knn_distances,
    audit_probe_rows,
    load_probe_jsonl,
)


class _IdentityNormalizer:
    def normalize_np(self, x):
        return np.asarray(x, dtype=np.float32)


class _FakeModel:
    def _require_fitted(self):
        return _IdentityNormalizer()


def row(*, action=0, decision=0, episode=0, model="m", seed=1000):
    return {
        "model_alias": model,
        "policy_version": 0,
        "episode_ordinal": episode,
        "episode_seed": seed,
        "agent_name": "blue_agent_0",
        "decision_index": decision,
        "global_tick_start": decision,
        "global_tick_end": decision + 1,
        "state": [float(decision)] + [0.0] * 26,
        "next_state": [float(decision + 1)] + [0.0] * 26,
        "requested_action_id": action,
        "canonical_action_id": action,
        "executed_action_family": ("Sleep", "Analyse", "Remove", "Restore")[action],
        "fallback": False,
        "action_completed": True,
        "selected_candidate_index": 0,
        "selected_plan": [action, 0, 0, 0],
        "response_reward": 0.0,
        "decision_dt": (1, 2, 3, 5)[action],
        "done": False,
        "cache_key": "a" * 64,
        "cache_hit": False,
    }


def good_one_step(*, coverage=True, rmse=0.10, spearman=0.5, bottom=0.1, top=0.2):
    counts = {"0": 10, "1": 10, "2": 10, "3": 10}
    if not coverage:
        counts["3"] = 0
    return {
        "completed_count": 40,
        "incomplete_count": 0,
        "rmse": rmse,
        "mae": rmse,
        "uncertainty_error_spearman": spearman,
        "bottom_uncertainty_quartile_mean_error": bottom,
        "top_uncertainty_quartile_mean_error": top,
        "per_action": {},
        "canonical_action_counts": counts,
    }


class TestGateB02PolicyOOD(unittest.TestCase):
    def setUp(self):
        self.rows = [row(action=i % 4, decision=i) for i in range(40)]
        self.reference = {
            "train_z": np.zeros((20, 27), dtype=np.float32),
            "threshold": 1.0,
        }

    def test_frozen_threshold_contract(self):
        self.assertEqual(KNN_K, 5)
        self.assertEqual(VALIDATION_PERCENTILE, 99.0)
        self.assertEqual(MAX_OOD_FRACTION, 0.10)
        self.assertEqual(MAX_PROBE_VALIDATION_RMSE_RATIO, 1.25)

    def test_knn_distance_uses_kth_neighbor(self):
        reference = np.asarray([[0.0], [1.0], [2.0], [3.0], [4.0]], dtype=np.float64)
        query = np.asarray([[0.0], [2.5]], dtype=np.float64)
        distance = _knn_distances(query, reference, k=2, chunk=1)
        np.testing.assert_allclose(distance, np.asarray([1.0, 0.5]), atol=1e-8)

    def test_h4_windows_require_contiguous_same_episode_model_agent(self):
        rows = [row(decision=i, action=i % 4, episode=0, model="m") for i in range(5)]
        rows.append(row(decision=5, action=1, episode=1, model="m"))
        windows = _h4_windows(rows)
        self.assertEqual(len(windows), 2)
        np.testing.assert_array_equal(windows[0][1], np.asarray([0, 1, 2, 3]))

    def test_gate_passes_when_all_frozen_checks_pass(self):
        with patch(
            "formal_experiments.evaluation.audit_b0_2_policy_ood._probe_one_step",
            return_value=good_one_step(),
        ), patch(
            "formal_experiments.evaluation.audit_b0_2_policy_ood._probe_h4",
            return_value={"window_count": 1, "rmse": 0.1, "uncertainty_error_spearman": 0.2},
        ), patch(
            "formal_experiments.evaluation.audit_b0_2_policy_ood._knn_distances",
            return_value=np.full(len(self.rows), 0.5),
        ):
            report = audit_probe_rows(
                rows=self.rows,
                model=_FakeModel(),
                reference=self.reference,
                validation_one_step_rmse=0.1,
            )
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["pass"])
        self.assertTrue(report["coverage_complete"])

    def test_only_missing_action_coverage_is_coverage_incomplete(self):
        with patch(
            "formal_experiments.evaluation.audit_b0_2_policy_ood._probe_one_step",
            return_value=good_one_step(coverage=False),
        ), patch(
            "formal_experiments.evaluation.audit_b0_2_policy_ood._probe_h4",
            return_value={"window_count": 0, "rmse": None, "uncertainty_error_spearman": None},
        ), patch(
            "formal_experiments.evaluation.audit_b0_2_policy_ood._knn_distances",
            return_value=np.full(len(self.rows), 0.5),
        ):
            report = audit_probe_rows(
                rows=self.rows,
                model=_FakeModel(),
                reference=self.reference,
                validation_one_step_rmse=0.1,
            )
        self.assertEqual(report["status"], "COVERAGE_INCOMPLETE")
        self.assertFalse(report["pass"])

    def test_rmse_shift_is_hard_fail_not_coverage_extension(self):
        with patch(
            "formal_experiments.evaluation.audit_b0_2_policy_ood._probe_one_step",
            return_value=good_one_step(coverage=False, rmse=0.2),
        ), patch(
            "formal_experiments.evaluation.audit_b0_2_policy_ood._probe_h4",
            return_value={"window_count": 0, "rmse": None, "uncertainty_error_spearman": None},
        ), patch(
            "formal_experiments.evaluation.audit_b0_2_policy_ood._knn_distances",
            return_value=np.full(len(self.rows), 0.5),
        ):
            report = audit_probe_rows(
                rows=self.rows,
                model=_FakeModel(),
                reference=self.reference,
                validation_one_step_rmse=0.1,
            )
        self.assertEqual(report["status"], "FAIL")
        self.assertFalse(report["gate_checks"]["rmse_ratio_ok"])

    def test_nonpositive_uncertainty_spearman_is_fail(self):
        with patch(
            "formal_experiments.evaluation.audit_b0_2_policy_ood._probe_one_step",
            return_value=good_one_step(spearman=0.0),
        ), patch(
            "formal_experiments.evaluation.audit_b0_2_policy_ood._probe_h4",
            return_value={"window_count": 0, "rmse": None, "uncertainty_error_spearman": None},
        ), patch(
            "formal_experiments.evaluation.audit_b0_2_policy_ood._knn_distances",
            return_value=np.full(len(self.rows), 0.5),
        ):
            report = audit_probe_rows(
                rows=self.rows,
                model=_FakeModel(),
                reference=self.reference,
                validation_one_step_rmse=0.1,
            )
        self.assertEqual(report["status"], "FAIL")

    def test_bad_uncertainty_quartile_order_is_fail(self):
        with patch(
            "formal_experiments.evaluation.audit_b0_2_policy_ood._probe_one_step",
            return_value=good_one_step(bottom=0.3, top=0.2),
        ), patch(
            "formal_experiments.evaluation.audit_b0_2_policy_ood._probe_h4",
            return_value={"window_count": 0, "rmse": None, "uncertainty_error_spearman": None},
        ), patch(
            "formal_experiments.evaluation.audit_b0_2_policy_ood._knn_distances",
            return_value=np.full(len(self.rows), 0.5),
        ):
            report = audit_probe_rows(
                rows=self.rows,
                model=_FakeModel(),
                reference=self.reference,
                validation_one_step_rmse=0.1,
            )
        self.assertEqual(report["status"], "FAIL")

    def test_ood_fraction_over_ten_percent_is_fail(self):
        distances = np.asarray([2.0] * 5 + [0.5] * 35, dtype=np.float64)
        with patch(
            "formal_experiments.evaluation.audit_b0_2_policy_ood._probe_one_step",
            return_value=good_one_step(),
        ), patch(
            "formal_experiments.evaluation.audit_b0_2_policy_ood._probe_h4",
            return_value={"window_count": 0, "rmse": None, "uncertainty_error_spearman": None},
        ), patch(
            "formal_experiments.evaluation.audit_b0_2_policy_ood._knn_distances",
            return_value=distances,
        ):
            report = audit_probe_rows(
                rows=self.rows,
                model=_FakeModel(),
                reference=self.reference,
                validation_one_step_rmse=0.1,
            )
        self.assertGreater(report["gate_checks"]["ood_fraction"], 0.10)
        self.assertEqual(report["status"], "FAIL")

    def test_probe_loader_rejects_nontrain_seed(self):
        bad = row(seed=2000)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "probe.jsonl"
            path.write_text(__import__("json").dumps(bad) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_probe_jsonl(path)


if __name__ == "__main__":
    unittest.main()
