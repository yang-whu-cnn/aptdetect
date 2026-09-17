import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from formal_experiments.ours.ppo_training import RolloutStep
from formal_experiments.ours.provisional_protocol import (
    FORMAL_ROLLOUT_TARGET,
    MAX_PROVISIONAL_TRANSITIONS,
    PROVISIONAL_STAGE_TARGETS,
    append_probe_records,
    count_probe_records,
    deserialize_rollout_step,
    safe_episode_steps_for_remaining,
    safe_update_barrier,
    serialize_rollout_step,
    validate_probe_record,
    validate_stage_target,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "formal_experiments" / "evaluation" / "run_b4_provisional_ppo.py"


def sample_step():
    return RolloutStep(
        episode_seed=1000,
        agent_name="blue_agent_0",
        decision_index=3,
        candidate_features=torch.arange(6 * 46, dtype=torch.float32).reshape(6, 46),
        candidate_index=2,
        behavior_log_prob=-1.2,
        critic_value=0.3,
        critic_next_value=0.4,
        real_response_reward=-1.0,
        decision_dt=2,
        done=False,
    )


def sample_probe():
    return {
        "model_alias": "llm_l_gemini35_flash_lite",
        "policy_version": 1,
        "episode_ordinal": 0,
        "episode_seed": 1000,
        "agent_name": "blue_agent_0",
        "decision_index": 0,
        "global_tick_start": 0,
        "global_tick_end": 1,
        "state": [0.0] * 27,
        "next_state": [0.1] * 27,
        "requested_action_id": 1,
        "canonical_action_id": 0,
        "executed_action_family": "Sleep",
        "fallback": True,
        "action_completed": True,
        "selected_candidate_index": 2,
        "selected_plan": [1, 0, 0, 0],
        "response_reward": -1.0,
        "decision_dt": 1,
        "done": False,
        "cache_key": "a" * 64,
        "cache_hit": False,
    }


class TestGateB4ProvisionalProtocol(unittest.TestCase):
    def test_frozen_stages_and_hard_cap(self):
        self.assertEqual(PROVISIONAL_STAGE_TARGETS, (2000, 5000, 10000, 20000))
        self.assertEqual(MAX_PROVISIONAL_TRANSITIONS, 20000)
        for target in PROVISIONAL_STAGE_TARGETS:
            self.assertEqual(validate_stage_target(target), target)
        for bad in (0, 1999, 2500, 20001):
            with self.assertRaises(ValueError):
                validate_stage_target(bad)

    def test_episode_steps_conservatively_respect_remaining_budget(self):
        self.assertEqual(safe_episode_steps_for_remaining(2000), 100)
        self.assertEqual(safe_episode_steps_for_remaining(500), 100)
        self.assertEqual(safe_episode_steps_for_remaining(200), 40)
        self.assertEqual(safe_episode_steps_for_remaining(25), 5)
        self.assertIsNone(safe_episode_steps_for_remaining(24))
        for remaining in (25, 26, 99, 500, 2000):
            steps = safe_episode_steps_for_remaining(remaining)
            self.assertLessEqual(5 * steps, remaining)

    def test_regular_update_requires_target_and_closed_async_barrier(self):
        self.assertEqual(FORMAL_ROLLOUT_TARGET, 128)
        self.assertFalse(
            safe_update_barrier(127, ppo_open_agents=(), replay_open_agents=())
        )
        self.assertTrue(
            safe_update_barrier(128, ppo_open_agents=(), replay_open_agents=())
        )
        self.assertFalse(
            safe_update_barrier(256, ppo_open_agents=("blue_agent_0",), replay_open_agents=())
        )
        self.assertFalse(
            safe_update_barrier(256, ppo_open_agents=(), replay_open_agents=("blue_agent_0",))
        )

    def test_stage_final_flush_allows_small_batch_only_at_closed_barrier(self):
        self.assertTrue(
            safe_update_barrier(17, ppo_open_agents=(), replay_open_agents=(), final_flush=True)
        )
        self.assertFalse(
            safe_update_barrier(0, ppo_open_agents=(), replay_open_agents=(), final_flush=True)
        )
        self.assertFalse(
            safe_update_barrier(
                17,
                ppo_open_agents=("blue_agent_1",),
                replay_open_agents=(),
                final_flush=True,
            )
        )

    def test_rollout_step_checkpoint_roundtrip_is_exact(self):
        original = sample_step()
        restored = deserialize_rollout_step(serialize_rollout_step(original))
        self.assertEqual(restored.episode_seed, original.episode_seed)
        self.assertEqual(restored.agent_name, original.agent_name)
        self.assertEqual(restored.decision_index, original.decision_index)
        self.assertEqual(restored.candidate_index, original.candidate_index)
        self.assertEqual(restored.decision_dt, original.decision_dt)
        self.assertEqual(restored.done, original.done)
        torch.testing.assert_close(restored.candidate_features, original.candidate_features)

    def test_probe_record_contract_accepts_real_reward_and_plan0(self):
        validate_probe_record(sample_probe())

    def test_probe_record_rejects_plan0_action_mismatch_and_nonfinite_state(self):
        bad = sample_probe()
        bad["requested_action_id"] = 2
        with self.assertRaises(ValueError):
            validate_probe_record(bad)
        bad = sample_probe()
        bad["state"][0] = float("nan")
        with self.assertRaises(ValueError):
            validate_probe_record(bad)

    def test_probe_jsonl_append_count_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "probe.jsonl"
            append_probe_records(path, [sample_probe(), sample_probe()])
            self.assertEqual(count_probe_records(path), 2)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 2)

    def test_runner_is_single_model_train_only_and_development_only(self):
        source = RUNNER.read_text(encoding="utf-8").lower()
        self.assertIn('split="train"', source)
        self.assertIn('formal_result_eligible": false', source)
        self.assertIn("formal_train_seeds", source)
        self.assertNotIn("validation_seeds", source)
        self.assertNotIn("calibration_seeds", source)
        self.assertNotIn("test_seeds", source)
        self.assertNotIn("range(4000", source)

    def test_runner_does_not_offer_all_models_implicit_batch_flag(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn('parser.add_argument("--model-alias", required=True)', source)
        self.assertNotIn("--all-models", source)


if __name__ == "__main__":
    unittest.main()
