import inspect
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from formal_experiments.ours.model_registry import load_model_registry
from formal_experiments.ours.posterior_features import (
    PLAN_ONE_HOT_DIM,
    POSTERIOR_CANDIDATE_DIM,
    build_posterior_candidate_features,
    encode_plan_one_hot,
    evidence_from_rollout,
)
from shared.formal_state import FORMAL_STATE_DIM


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "configs" / "llm_model_registry_v1.yaml"
CONFIG_PATH = ROOT / "configs" / "lwm_rl_gate_b_v2_1.yaml"
POSTERIOR_SOURCE = ROOT / "formal_experiments" / "ours" / "posterior_features.py"


def plans6():
    return np.asarray(
        [
            [0, 1, 2, 3],
            [3, 2, 1, 0],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [3, 3, 3, 3],
        ],
        dtype=np.int64,
    )


def priors6():
    return np.asarray([0.25, 0.20, 0.20, 0.15, 0.10, 0.10], dtype=np.float32)


def member_returns6():
    return torch.tensor(
        [
            [-10.0, -9.0, -11.0, -10.5, -9.5],
            [-7.0, -8.0, -7.5, -8.5, -6.5],
            [-12.0, -12.0, -11.0, -13.0, -12.0],
            [-5.0, -6.0, -5.5, -4.5, -5.0],
            [-9.0, -8.0, -10.0, -9.5, -8.5],
            [-6.0, -7.0, -6.5, -5.5, -6.0],
        ],
        dtype=torch.float32,
    )


def rollout(members=None):
    members = member_returns6() if members is None else torch.as_tensor(members, dtype=torch.float32)
    return SimpleNamespace(
        expected_return=members.mean(dim=1),
        member_returns=members,
    )


class TestGateB3PosteriorRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = load_model_registry(REGISTRY_PATH)
        with CONFIG_PATH.open("r", encoding="utf-8") as handle:
            cls.cfg = yaml.safe_load(handle)

    def test_dimensions_and_frozen_component_contract(self):
        self.assertEqual(FORMAL_STATE_DIM, 27)
        self.assertEqual(PLAN_ONE_HOT_DIM, 16)
        self.assertEqual(POSTERIOR_CANDIDATE_DIM, 46)
        self.assertEqual(46, 27 + 16 + 1 + 1 + 1)
        posterior = self.cfg["posterior_observation"]
        self.assertEqual(posterior["candidate_feature_dim"], 46)
        self.assertEqual(
            posterior["components"],
            [
                "current_state",
                "candidate_plan",
                "llm_prior_preference",
                "predicted_value",
                "predictive_uncertainty",
            ],
        )
        self.assertFalse(posterior["action_cost_evidence"])
        self.assertFalse(posterior["checkpoint_coverage_evidence"])
        self.assertFalse(posterior["delay_evidence"])
        self.assertFalse(posterior["early_warning_evidence"])

    def test_exact_feature_column_layout(self):
        state = np.arange(27, dtype=np.float32) / 10.0
        plans = plans6()
        prior = priors6()
        members = member_returns6()
        features = build_posterior_candidate_features(state, plans, prior, rollout(members))
        self.assertEqual(tuple(features.shape), (6, 46))

        expected_state = torch.as_tensor(state).view(1, -1).expand(6, -1)
        torch.testing.assert_close(features[:, :27], expected_state)
        torch.testing.assert_close(features[:, 27:43], encode_plan_one_hot(plans))
        torch.testing.assert_close(features[:, 43], torch.as_tensor(prior))
        torch.testing.assert_close(features[:, 44], members.mean(dim=1))
        torch.testing.assert_close(features[:, 45], members.std(dim=1, unbiased=False))

    def test_provider_neutral_builder_signature_and_registry(self):
        self.assertFalse(self.registry.primary_model_selected)
        self.assertEqual(len(self.registry.models), 3)
        signature = inspect.signature(build_posterior_candidate_features)
        self.assertEqual(
            list(signature.parameters),
            ["state", "plans", "prior_preferences", "rollout_result"],
        )

        state = np.zeros(27, dtype=np.float32)
        baseline = build_posterior_candidate_features(state, plans6(), priors6(), rollout())
        for spec in self.registry.models.values():
            self.assertTrue(spec.exact_model_id)
            actual = build_posterior_candidate_features(state, plans6(), priors6(), rollout())
            torch.testing.assert_close(actual, baseline)

    def test_candidate_permutation_equivariance(self):
        state = np.linspace(0.0, 1.0, 27, dtype=np.float32)
        plans = plans6()
        prior = priors6()
        members = member_returns6()
        base = build_posterior_candidate_features(state, plans, prior, rollout(members))

        permutation = np.asarray([5, 2, 0, 4, 1, 3], dtype=np.int64)
        permuted = build_posterior_candidate_features(
            state,
            plans[permutation],
            prior[permutation],
            rollout(members[permutation]),
        )
        torch.testing.assert_close(
            permuted,
            base[torch.as_tensor(permutation, dtype=torch.long)],
        )

    def test_value_is_member_mean_and_uncertainty_is_population_std(self):
        members = member_returns6()
        evidence = evidence_from_rollout(rollout(members))
        torch.testing.assert_close(evidence.expected_return, members.mean(dim=1))
        torch.testing.assert_close(
            evidence.predictive_uncertainty,
            members.std(dim=1, unbiased=False),
        )

    def test_evidence_rejects_inconsistent_or_nonfinite_inputs(self):
        members = member_returns6()
        with self.assertRaises(ValueError):
            evidence_from_rollout(
                SimpleNamespace(
                    expected_return=torch.zeros(6),
                    member_returns=members,
                )
            )
        bad = members.clone()
        bad[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            evidence_from_rollout(
                SimpleNamespace(
                    expected_return=torch.zeros(6),
                    member_returns=bad,
                )
            )

    def test_prior_probability_contract_rejects_bad_inputs(self):
        state = np.zeros(27, dtype=np.float32)
        good = priors6()
        bad_cases = [
            np.ones(6, dtype=np.float32),
            np.asarray([0.4, 0.3, 0.2, 0.1, 0.1, -0.1], dtype=np.float32),
            np.asarray([0.2, 0.2, 0.2, 0.2, 0.1, np.nan], dtype=np.float32),
            np.asarray([0.5, 0.5], dtype=np.float32),
        ]
        build_posterior_candidate_features(state, plans6(), good, rollout())
        for bad in bad_cases:
            with self.assertRaises(ValueError):
                build_posterior_candidate_features(state, plans6(), bad, rollout())

    def test_plan_encoding_rejects_bad_shape_or_action_ids(self):
        encode_plan_one_hot(plans6())
        with self.assertRaises(ValueError):
            encode_plan_one_hot(np.zeros((5, 4), dtype=np.int64))
        invalid = plans6()
        invalid[0, 0] = 4
        with self.assertRaises(ValueError):
            encode_plan_one_hot(invalid)
        invalid = plans6()
        invalid[0, 0] = -1
        with self.assertRaises(ValueError):
            encode_plan_one_hot(invalid)

    def test_state_contract_rejects_bad_shape_or_nonfinite(self):
        with self.assertRaises(ValueError):
            build_posterior_candidate_features(
                np.zeros(26, dtype=np.float32), plans6(), priors6(), rollout()
            )
        bad = np.zeros(27, dtype=np.float32)
        bad[3] = np.inf
        with self.assertRaises(ValueError):
            build_posterior_candidate_features(bad, plans6(), priors6(), rollout())

    def test_posterior_source_has_no_provider_specific_or_legacy_evidence(self):
        source = POSTERIOR_SOURCE.read_text(encoding="utf-8").lower()
        for forbidden in (
            "gpt-5",
            "gemini",
            "ofox",
            "api_key",
            "action_cost",
            "checkpoint_coverage",
            "early_warning",
            "lambda_delay",
            "control_traffic",
            "light_evidence",
            "heavy_evidence",
            "local_mitigate",
            "strong_mitigate",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
