import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

from formal_experiments.data_collection.incident_response import ResponseRewardConfig
from formal_experiments.evaluation import audit_model_space_action_consistency as action_audit
from formal_experiments.evaluation.validate_response_reward_predictor import GAMMA_TICK
from formal_experiments.training.bootstrap_world_model import BootstrapWorldModelConfig
from formal_experiments.training.response_reward_predictor import ResponseRewardPredictorConfig
from shared.action_contract import ACTION_CONTRACTS, N_ACTIONS
from shared.formal_state import FORMAL_STATE_DIM
from shared.model_space_action import (
    ANY_TARGET_INDEX,
    TARGET_THRESHOLD,
    canonicalize_requested_action,
    canonicalize_requested_tensor,
)


ROOT = Path(__file__).resolve().parents[1]
FORMAL_CONFIG_PATH = ROOT / "configs" / "compare_ug_cem_formal_v2_1.yaml"


def load_config():
    with FORMAL_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class TestGateAFinalIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = load_config()

    def test_core_dimensions_and_action_contract_are_one_source(self):
        self.assertEqual(FORMAL_STATE_DIM, 27)
        self.assertEqual(N_ACTIONS, 4)
        self.assertEqual(self.config["state"]["dim"], FORMAL_STATE_DIM)
        self.assertEqual(self.config["action_space"]["n_actions"], N_ACTIONS)

        durations = {
            int(item.action_id): int(item.duration_ticks)
            for item in ACTION_CONTRACTS
        }
        self.assertEqual(durations, {0: 1, 1: 2, 2: 3, 3: 5})
        self.assertEqual(action_audit.ACTION_DURATION, durations)

    def test_shared_canonicalizer_is_the_audit_entrypoint(self):
        self.assertIs(
            action_audit.canonicalize_requested_action,
            canonicalize_requested_action,
        )
        self.assertIs(
            action_audit.canonicalize_requested_tensor,
            canonicalize_requested_tensor,
        )
        self.assertEqual(
            self.config["action_space"]["model_space_canonicalizer"],
            "shared.model_space_action.canonicalize_requested_action",
        )
        self.assertEqual(ANY_TARGET_INDEX, 17)
        self.assertEqual(float(TARGET_THRESHOLD), 0.5)

    def test_requested_to_canonical_action_contract(self):
        unavailable = np.zeros(FORMAL_STATE_DIM, dtype=np.float32)
        available = unavailable.copy()
        available[ANY_TARGET_INDEX] = 1.0

        self.assertEqual(canonicalize_requested_action(unavailable, 0), 0)
        self.assertEqual(canonicalize_requested_action(available, 0), 0)

        for action_id in (1, 2, 3):
            self.assertEqual(canonicalize_requested_action(unavailable, action_id), 0)
            self.assertEqual(canonicalize_requested_action(available, action_id), action_id)

        states = torch.as_tensor(np.stack([unavailable, available, unavailable, available]))
        requested = torch.as_tensor([0, 1, 2, 3])
        result = canonicalize_requested_tensor(states, requested).cpu().numpy()
        np.testing.assert_array_equal(result, np.asarray([0, 1, 0, 3]))

    def test_world_model_contract_matches_formal_config(self):
        source = BootstrapWorldModelConfig()
        formal = self.config["world_model"]

        self.assertEqual(source.state_dim, formal["state_dim"])
        self.assertEqual(source.n_actions, formal["n_actions"])
        self.assertEqual(source.ensemble_size, formal["ensemble_size"])
        self.assertEqual(source.hidden_dim, formal["hidden_dim"])
        self.assertEqual(source.target_mode, formal["target_mode"])
        self.assertEqual(source.target_mode, "absolute")

    def test_reward_model_and_objective_match_formal_config(self):
        predictor = ResponseRewardPredictorConfig()
        reward_cfg = self.config["response_reward_predictor"]
        objective = ResponseRewardConfig()
        objective_cfg = self.config["response_objective"]

        self.assertEqual(predictor.state_dim, reward_cfg["input_state_dim"])
        self.assertEqual(predictor.state_dim, reward_cfg["next_state_dim"])
        self.assertEqual(predictor.n_actions, reward_cfg["action_dim"])
        self.assertEqual(predictor.hidden_dim, reward_cfg["hidden_dim"])
        self.assertEqual(float(GAMMA_TICK), float(reward_cfg["gamma_tick"]))
        self.assertEqual(float(objective.lambda_time), float(objective_cfg["lambda_time"]))
        self.assertEqual(float(objective.lambda_failure), float(objective_cfg["lambda_failure"]))

    def test_planning_semantics_match_runtime_audit(self):
        planning = self.config["planning_contract"]
        self.assertEqual(action_audit.HORIZON, planning["horizon"])
        self.assertEqual(float(GAMMA_TICK), float(planning["gamma_tick"]))
        self.assertTrue(planning["execute_first_action_only"])
        self.assertTrue(planning["replan_each_decision_epoch"])
        self.assertTrue(planning["duration_aware_discount"])

    def test_formal_artifacts_are_v2_and_train_validation_only(self):
        artifacts = self.config["artifacts"]
        for key, path in artifacts.items():
            self.assertIn("_v2", path, key)
            self.assertNotIn("outputs/formal_replay/", path, key)
            self.assertNotIn("outputs/world_model/", path, key)
            self.assertNotIn("calibration", path.lower(), key)
            self.assertNotIn("test.jsonl", path.lower(), key)

        fairness = self.config["fairness"]
        self.assertTrue(fairness["calibration_test_update_forbidden"])

    def test_planner_visible_shared_layer_does_not_import_hidden_bookkeeping(self):
        planner_files = (
            ROOT / "shared" / "formal_state.py",
            ROOT / "shared" / "model_space_action.py",
            ROOT / "shared" / "cyborg_target_resolver.py",
            ROOT / "shared" / "cyborg_action_adapter.py",
        )

        forbidden = (
            "formal_experiments.data_collection.incident_response",
            "ground_truth_red_presence",
            "IncidentResponseBookkeeper",
        )

        for path in planner_files:
            source = path.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, source, f"{path.name}: {token}")


if __name__ == "__main__":
    unittest.main()
