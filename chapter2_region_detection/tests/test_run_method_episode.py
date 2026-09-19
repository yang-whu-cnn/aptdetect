import unittest

from formal_experiments.evaluation.run_method_episode import (
    derive_runtime_seed,
    plan_rsmbrl_with_visible_mask,
    incident_active_at_action_start,
    root_action_mask_from_availability,
    validate_episode_request,
)


class TestRunMethodEpisodeContract(unittest.TestCase):
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

    def test_uamcts_is_registered_without_relaxing_formal_seed_gate(self):
        from formal_experiments.evaluation.run_method_episode import METHODS, run_method_episode
        self.assertIn("uamcts_cc4", METHODS)
        with self.assertRaisesRegex(ValueError, "policy_seed"):
            run_method_episode(method="uamcts_cc4", seed=3999, ticks=500, run_mode="formal")


if __name__ == "__main__":
    unittest.main()
