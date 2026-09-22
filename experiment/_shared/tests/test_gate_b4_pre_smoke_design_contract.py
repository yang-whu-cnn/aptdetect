import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "lwm_rl_gate_b_v2_1.yaml"
REGISTRY = ROOT / "configs" / "llm_model_registry_v1.yaml"


class TestGateB4PreSmokeDesignContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        cls.registry = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))

    def test_formal_llm_selection_is_registry_driven_and_primary_unselected(self):
        root = self.config["config"]
        prior = self.config["llm_prior"]
        self.assertEqual(root["version"], "2.3")
        self.assertEqual(prior["mode"], "registry_driven_multi_model")
        self.assertEqual(prior["model_registry"], "configs/llm_model_registry_v1.yaml")
        self.assertFalse(prior["primary_model_selected"])
        self.assertFalse(self.registry["registry"]["primary_model_selected"])

    def test_registry_contains_exactly_three_formal_llm_tiers(self):
        models = self.registry["models"]
        self.assertEqual(tuple(models.keys()), ("tier_h", "tier_m", "tier_l"))
        aliases = {entry["experiment_alias"] for entry in models.values()}
        self.assertEqual(
            aliases,
            {
                "llm_h_gpt56_sol",
                "llm_m_gpt54_mini",
                "llm_l_gemini35_flash_lite",
            },
        )

    def test_prompt_context_is_d27_plus_public_agent_only_and_cache_binds_agent(self):
        prior = self.config["llm_prior"]
        context = prior["prompt_context"]
        cache = prior["cache_identity"]
        self.assertEqual(context["formal_state"], "planner_visible_d27")
        self.assertTrue(context["public_agent_identifier"])
        self.assertFalse(context["hidden_truth"])
        self.assertTrue(cache["exact_d27_float32"])
        self.assertTrue(cache["public_agent_identifier"])
        self.assertTrue(cache["split"])
        self.assertTrue(cache["model_registry_identity"])

    def test_ppo_reward_and_duration_contract_matches_taskbook(self):
        ppo = self.config["ppo_pretrain_contract"]
        self.assertEqual(ppo["candidate_feature_dim"], 46)
        self.assertEqual(ppo["candidate_count"], 6)
        self.assertEqual(ppo["formal_action_space"], 4)
        self.assertEqual(ppo["gamma_tick"], 0.99)
        self.assertEqual(ppo["gae_lambda"], 0.95)
        self.assertTrue(ppo["duration_aware_discount"])
        self.assertEqual(
            ppo["duration_gae_rule"],
            "gamma_tick_power_dt_times_lambda_per_decision",
        )
        self.assertEqual(ppo["reward_source"], "real_cc4_accumulated_response_reward_only")
        self.assertFalse(ppo["wm_predicted_value_is_reward_target"])

    def test_provisional_and_ood_cover_all_three_variants_with_train_only_limits(self):
        ppo = self.config["ppo_pretrain_contract"]
        expected = [
            "llm_h_gpt56_sol",
            "llm_m_gpt54_mini",
            "llm_l_gemini35_flash_lite",
        ]
        self.assertEqual(ppo["provisional_policy_variants"], expected)
        self.assertEqual(ppo["provisional_max_decision_transitions_per_variant"], 20000)
        self.assertFalse(ppo["provisional_weights_formal_eligible"])
        self.assertEqual(ppo["train_seeds"], list(range(1000, 1032)))
        self.assertEqual(ppo["validation_seeds"], list(range(2000, 2008)))
        self.assertTrue(ppo["calibration_test_forbidden"])

        ood = self.config["b0_2_ood_contract"]
        self.assertTrue(ood["evaluate_all_formal_llm_variants"])
        self.assertTrue(ood["train_only"])
        self.assertTrue(ood["no_test_or_calibration"])
        self.assertEqual(ood["knn_k"], 5)
        self.assertEqual(ood["validation_to_train_percentile"], 99)
        self.assertEqual(ood["max_ood_fraction"], 0.10)
        self.assertEqual(ood["max_probe_to_validation_rmse_ratio"], 1.25)

    def test_shared_model_and_objective_exclude_legacy_reward_terms(self):
        shared = self.config["shared_contract"]
        objective = shared["response_objective"]
        self.assertEqual(shared["state_dim"], 27)
        self.assertEqual(shared["n_actions"], 4)
        self.assertEqual(shared["horizon"], 4)
        self.assertIn("world_model_absolute.pt", shared["world_model"])
        self.assertIn("response_reward_predictor.pt", shared["response_reward_predictor"])
        self.assertFalse(objective["early_warning_lead_reward"])
        self.assertFalse(objective["fixed_action_cost"])
        self.assertEqual(objective["reward_protocol"], "final_paper_20260917_v1")
        self.assertEqual(objective["delay_term"], "sum_active_incident_compromise_age")
        self.assertEqual(objective["failure_scope"], "all_tracked_host_local_work_failures")


if __name__ == "__main__":
    unittest.main()
