import unittest
from pathlib import Path

import yaml

from shared.action_contract import ACTION_CONTRACTS, N_ACTIONS
from shared.formal_state import FORMAL_STATE_DIM
from formal_experiments.data_collection.incident_response import ResponseRewardConfig


ROOT = Path(__file__).resolve().parents[1]
FORMAL_CONFIG_PATH = ROOT / "configs" / "compare_ug_cem_formal_v2_1.yaml"
LEGACY_CONFIG_PATH = ROOT / "configs" / "compare_ug_cem_local_online.yaml"


def load_formal_config():
    with FORMAL_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class TestGateAFormalComparisonConfig(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = load_formal_config()

    def test_config_marked_formal_v2_1(self):
        meta = self.config["config"]
        self.assertEqual(meta["version"], "2.1")
        self.assertEqual(meta["status"], "formal_gate_a")

    def test_state_contract_matches_code(self):
        state = self.config["state"]
        self.assertEqual(state["dim"], FORMAL_STATE_DIM)
        self.assertEqual(state["dim"], 27)
        self.assertEqual(state["availability_feature"], "any_valid_observable_target")
        self.assertEqual(state["availability_index"], 17)
        self.assertTrue(state["planner_visible_only"])
        self.assertFalse(state["hidden_red_truth"])

    def test_action_contract_matches_code(self):
        action_cfg = self.config["action_space"]
        self.assertEqual(action_cfg["n_actions"], N_ACTIONS)
        self.assertEqual(action_cfg["n_actions"], 4)

        expected = [
            {
                "id": int(item.action_id),
                "name": item.name,
                "cyborg_action": item.cyborg_action,
                "duration_ticks": int(item.duration_ticks),
            }
            for item in ACTION_CONTRACTS
        ]
        self.assertEqual(action_cfg["actions"], expected)
        self.assertEqual(action_cfg["fallback_on_no_valid_observable_target"], "Sleep")
        self.assertTrue(action_cfg["record_requested_and_executed"])
        self.assertEqual(
            action_cfg["model_space_canonicalizer"],
            "shared.model_space_action.canonicalize_requested_action",
        )

    def test_formal_artifacts_are_v2(self):
        artifacts = self.config["artifacts"]
        self.assertEqual(artifacts["train_replay"], "outputs/formal_replay_v2/train.jsonl")
        self.assertEqual(artifacts["validation_replay"], "outputs/formal_replay_v2/validation.jsonl")
        self.assertEqual(artifacts["selected_world_model"], "outputs/world_model_v2/a4_5b/world_model_absolute.pt")
        self.assertEqual(artifacts["response_reward_predictor"], "outputs/world_model_v2/a4_5c/response_reward_predictor.pt")
        self.assertEqual(artifacts["action_consistency_report"], "outputs/world_model_v2/a4_6a/action_consistency_report.json")

    def test_world_model_and_reward_contract(self):
        wm = self.config["world_model"]
        reward = self.config["response_reward_predictor"]
        objective = self.config["response_objective"]
        planning = self.config["planning_contract"]

        self.assertEqual(wm["state_dim"], 27)
        self.assertEqual(wm["n_actions"], 4)
        self.assertEqual(wm["ensemble_size"], 5)
        self.assertEqual(wm["hidden_dim"], 128)
        self.assertEqual(wm["target_mode"], "absolute")
        self.assertTrue(wm["fixed_member_through_horizon"])
        self.assertTrue(wm["deterministic_mean_rollout"])
        self.assertFalse(wm["aleatoric_resampling"])

        self.assertEqual(reward["action_dim"], 4)
        self.assertEqual(reward["hidden_dim"], 128)
        self.assertEqual(float(reward["gamma_tick"]), 0.99)
        self.assertFalse(reward["hidden_reward_inputs"])

        self.assertEqual(planning["horizon"], 4)
        self.assertEqual(float(planning["gamma_tick"]), 0.99)
        self.assertTrue(planning["duration_aware_discount"])

        source_objective = ResponseRewardConfig()
        self.assertEqual(
            float(objective["lambda_time"]),
            float(source_objective.lambda_time),
        )
        self.assertEqual(
            float(objective["lambda_failure"]),
            float(source_objective.lambda_failure),
        )
        self.assertTrue(objective["attack_eradication_time"])
        self.assertFalse(objective["incident_host_lwf_only"])
        self.assertEqual(objective["reward_protocol"], "final_paper_20260917_v1")
        self.assertEqual(objective["delay_term"], "sum_active_incident_compromise_age")
        self.assertEqual(objective["failure_scope"], "all_tracked_host_local_work_failures")
        self.assertTrue(objective["official_team_reward_separate"])

    def test_seed_protocol_is_exact(self):
        seeds = self.config["seed_protocol"]
        self.assertEqual(seeds["train"], list(range(1000, 1032)))
        self.assertEqual(seeds["validation"], list(range(2000, 2008)))
        self.assertEqual(seeds["calibration"], list(range(3000, 3008)))
        self.assertEqual(seeds["test"], list(range(4000, 4020)))

    def test_final_paper_evaluation_protocol(self):
        evaluation = self.config["final_paper_evaluation"]
        self.assertEqual(evaluation["episodes_per_repeat"], 100)
        self.assertEqual(evaluation["timesteps_per_episode"], 500)
        self.assertEqual(evaluation["independent_training_repeats"], 5)
        self.assertEqual(
            evaluation["metrics"],
            [
                "cc4_official_reward",
                "operation_failure_penalty",
                "recovery_precision",
                "recovery_time",
            ],
        )

    def test_fairness_contract(self):
        fairness = self.config["fairness"]
        self.assertEqual(fairness["methods"], ["lwm_rl", "ug_cem_apt", "cem_apt"])
        for key in (
            "shared_state_encoder",
            "shared_action_contract",
            "shared_target_resolver",
            "shared_cyborg_action_adapter",
            "shared_world_model_checkpoint",
            "shared_response_reward_predictor",
            "shared_seed_protocol",
            "shared_decision_epoch_semantics",
            "calibration_test_update_forbidden",
        ):
            self.assertTrue(fairness[key], key)

    def test_no_legacy_objective_keys_in_formal_config(self):
        text = FORMAL_CONFIG_PATH.read_text(encoding="utf-8")
        for forbidden in (
            "control_traffic",
            "lambda_cost",
            "lambda_delay",
            "costs:",
            "delays:",
            "action_dim: 5",
            "n_actions: 5",
            "state_dim: 8",
        ):
            self.assertNotIn(forbidden, text)

    def test_legacy_config_is_marked_and_excluded(self):
        legacy_text = LEGACY_CONFIG_PATH.read_text(encoding="utf-8")
        self.assertIn("LEGACY / DEVELOPMENT ONLY", legacy_text)
        self.assertIn("compare_ug_cem_formal_v2_1.yaml", legacy_text)

        isolation = self.config["legacy_isolation"]
        self.assertEqual(
            isolation["legacy_development_config"],
            "configs/compare_ug_cem_local_online.yaml",
        )
        self.assertFalse(isolation["legacy_config_allowed_for_formal_runs"])
        self.assertFalse(isolation["old_replay_allowed_for_formal_runs"])
        self.assertFalse(isolation["old_world_model_allowed_for_formal_runs"])


if __name__ == "__main__":
    unittest.main()
