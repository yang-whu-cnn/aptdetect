import hashlib
import json
import unittest

from formal_experiments.evaluation.run_method_episode import (
    derive_runtime_seed,
    plan_rsmbrl_with_visible_mask,
    incident_active_at_action_start,
    root_action_mask_from_availability,
    validate_formal_learned_checkpoint,
    validate_episode_request,
)


class TestRunMethodEpisodeContract(unittest.TestCase):
    def learned_payload(self, method):
        common = {
            "method": method, "policy_seed": 51001,
            "formal_training_complete": True, "training_split": "train",
            "training_episode_seeds": list(range(1000, 1032)),
            "validation_episode_seeds": list(range(2000, 2008)),
            "test_seeds_used": False, "ticks_per_episode": 500,
            "code_commit": "a" * 40, "git_dirty": False,
            "git_diff_sha256": "0" * 64,
        }
        if method == "terla_a4":
            common.update({
                "schema": "terla_a4_formal_checkpoint_v1",
                "shared_policy_across_agents": True,
                "per_agent_history_isolated": True,
                "hidden_truth_policy_input": False,
                "reward": "original_terla_cyber_reward",
                "selected_training_episodes": 16,
                "selected_checkpoint_sha256": "b" * 64,
            })
        else:
            common.update({
                "schema": "carl_cc4_formal_checkpoint_v1",
                "world_model_frozen": True, "world_model_sha256": "c" * 64,
                "synthetic_rollouts_per_real_rollout": 8,
                "requested_imagination_horizon": 256,
                "effective_imagination_horizon": 4,
                "imagination_gate_status": "ADAPTED_TRUNCATED",
                "imagination_disclosure": "validated H=4 adaptation",
                "hidden_truth_policy_input": False,
                "decision_time_model_use": False,
            })
        return common

    def test_dev_accepts_short_exact_tick_request(self):
        validate_episode_request(run_mode="dev", ticks=20)

    def test_formal_hard_locks_500_ticks(self):
        validate_episode_request(run_mode="formal", ticks=500)
        with self.assertRaisesRegex(ValueError, "500"):
            validate_episode_request(run_mode="formal", ticks=20)

    def test_invalid_mode_and_tick_type_fail_closed(self):
        with self.assertRaises(ValueError):
            validate_episode_request(run_mode="pilot", ticks=20)
        with self.assertRaises(ValueError):
            validate_episode_request(run_mode="dev", ticks=True)

    def test_rsmbrl_receives_planner_visible_root_mask(self):
        class FakePlanner:
            def plan(self, state, *, root_action_mask=None):
                self.state = state
                self.mask = root_action_mask
                return "planned"

        planner = FakePlanner()
        result, mask = plan_rsmbrl_with_visible_mask(
            planner, "D27", {"Analyse": True, "Remove": False, "Restore": False}
        )
        self.assertEqual(result, "planned")
        self.assertEqual(mask, [True, True, False, False])
        self.assertEqual(planner.mask, mask)

    def test_root_mask_requires_all_three_boolean_target_families(self):
        with self.assertRaises(ValueError):
            root_action_mask_from_availability({"Analyse": True, "Remove": False})

    def test_recovery_label_uses_pre_step_truth_snapshot(self):
        previous = {"host-a": True}
        self.assertTrue(incident_active_at_action_start("host-a", previous))
        previous_after_step = {"host-a": False}
        self.assertFalse(incident_active_at_action_start("host-a", previous_after_step))

    def test_policy_and_episode_seeds_are_independently_bound(self):
        base = derive_runtime_seed(policy_seed=51001, episode_seed=4000, agent_name="blue_agent_0")
        self.assertNotEqual(base, derive_runtime_seed(policy_seed=51002, episode_seed=4000, agent_name="blue_agent_0"))
        self.assertNotEqual(base, derive_runtime_seed(policy_seed=51001, episode_seed=4001, agent_name="blue_agent_0"))

    def test_formal_episode_requires_frozen_policy_seed(self):
        from formal_experiments.evaluation.run_method_episode import run_method_episode
        with self.assertRaisesRegex(ValueError, "policy_seed"):
            run_method_episode(method="dca_cc4", seed=4000, ticks=500, run_mode="formal")

    def test_formal_episode_rejects_non_test_seed_before_environment_start(self):
        from formal_experiments.evaluation.run_method_episode import run_method_episode
        with self.assertRaisesRegex(ValueError, r"4000\.\.4099"):
            run_method_episode(method="dca_cc4", seed=3999, ticks=500,
                               run_mode="formal", policy_seed=51001)

    def test_uamcts_is_registered_without_relaxing_formal_seed_gate(self):
        from formal_experiments.evaluation.run_method_episode import METHODS, run_method_episode
        self.assertIn("uamcts_cc4", METHODS)
        with self.assertRaisesRegex(ValueError, "policy_seed"):
            run_method_episode(method="uamcts_cc4", seed=3999, ticks=500, run_mode="formal")

    def test_terla_formal_checkpoint_requires_bound_training_manifest(self):
        payload = self.learned_payload("terla_a4")
        checkpoint_sha = "d" * 64
        manifest = {key: value for key, value in payload.items() if key != "state_dict"}
        manifest["checkpoint_sha256"] = checkpoint_sha
        validate_formal_learned_checkpoint(
            method="terla_a4", payload=payload, policy_seed=51001,
            checkpoint_sha256=checkpoint_sha, training_manifest=manifest,
        )
        manifest["training_episode_seeds"] = [4000]
        with self.assertRaisesRegex(ValueError, "does not bind"):
            validate_formal_learned_checkpoint(
                method="terla_a4", payload=payload, policy_seed=51001,
                checkpoint_sha256=checkpoint_sha, training_manifest=manifest,
            )

    def test_carl_formal_checkpoint_requires_bound_selection_and_world_model(self):
        payload = self.learned_payload("carl_cc4")
        selection = {
            "schema": "carl_cc4_validation_selection_v1", "method": "carl_cc4",
            "training_episode_seeds": list(range(1000, 1032)),
            "validation_episode_seeds": list(range(2000, 2008)),
            "test_seeds_used": False, "world_model_sha256": "c" * 64,
            "code_commit": "a" * 40, "git_dirty": False,
            "git_diff_sha256": "0" * 64,
        }
        encoded = json.dumps(selection, indent=2, sort_keys=True) + "\n"
        payload["validation_selection_sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
        validate_formal_learned_checkpoint(
            method="carl_cc4", payload=payload, policy_seed=51001,
            checkpoint_sha256="d" * 64, validation_selection=selection,
            current_world_model_sha256="c" * 64,
        )
        with self.assertRaisesRegex(ValueError, "world-model"):
            validate_formal_learned_checkpoint(
                method="carl_cc4", payload=payload, policy_seed=51001,
                checkpoint_sha256="d" * 64, validation_selection=selection,
                current_world_model_sha256="e" * 64,
            )


if __name__ == "__main__":
    unittest.main()
