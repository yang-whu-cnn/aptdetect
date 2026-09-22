import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from formal_experiments.evaluation.run_b4_provisional_stage import (
    DEFAULT_PPO_SEED,
    DEFAULT_PROVISIONAL_SCENARIO_STEPS,
    cumulative_probe_summary,
    frozen_protocol_manifest,
    run_strict_stage,
    write_or_validate_manifest,
)
from formal_experiments.ours.provisional_protocol import PROVISIONAL_STAGE_TARGETS


def probe_record(*, agent="blue_agent_0", seed=1000, requested=0, canonical=0, reward=0.0):
    return {
        "model_alias": "llm_l_gemini35_flash_lite",
        "policy_version": 0,
        "episode_ordinal": 0,
        "episode_seed": seed,
        "agent_name": agent,
        "decision_index": 0,
        "global_tick_start": 0,
        "global_tick_end": 1,
        "state": np.zeros(27, dtype=np.float32).tolist(),
        "next_state": np.zeros(27, dtype=np.float32).tolist(),
        "requested_action_id": requested,
        "canonical_action_id": canonical,
        "executed_action_family": "Sleep" if canonical == 0 else "Analyse",
        "fallback": bool(requested != canonical),
        "action_completed": True,
        "selected_candidate_index": 0,
        "selected_plan": [requested, 0, 0, 0],
        "response_reward": reward,
        "official_reward": reward * 2.0,
        "decision_dt": 1,
        "done": False,
        "cache_key": "a" * 64,
        "cache_hit": False,
    }


class TestGateB4StrictProvisionalStage(unittest.TestCase):
    def test_manifest_freezes_stage_invariants(self):
        manifest = frozen_protocol_manifest(model_alias="llm_l_gemini35_flash_lite")
        self.assertEqual(manifest["scenario_steps"], DEFAULT_PROVISIONAL_SCENARIO_STEPS)
        self.assertEqual(manifest["ppo_seed"], DEFAULT_PPO_SEED)
        self.assertEqual(tuple(manifest["stage_targets"]), PROVISIONAL_STAGE_TARGETS)
        self.assertEqual(manifest["rollout_target"], 128)
        self.assertEqual(manifest["ppo_training_config"]["update_epochs"], 5)
        self.assertEqual(manifest["ppo_training_config"]["minibatch_size"], 64)
        self.assertEqual(manifest["format_version"], 3)
        self.assertEqual(manifest["response_reward_protocol"], "final_paper_20260917_v1")
        self.assertEqual(
            manifest["world_model_sha256"],
            "3b86593aa8adda3e0e173bfb700c543bd88641d23cf3e73ddc3992284c8f900f",
        )
        self.assertEqual(
            manifest["reward_model_sha256"],
            "f333330510b5de8e78fb9e2dc4287dd87fee29695fe38cbec5e60700ee615624",
        )
        self.assertTrue(manifest["formal_reuse_forbidden"])

    def test_manifest_resume_requires_exact_match(self):
        with tempfile.TemporaryDirectory() as temp:
            path, expected, sha = write_or_validate_manifest(
                out_root=temp,
                model_alias="llm_l_gemini35_flash_lite",
                resume=False,
            )
            path2, actual, sha2 = write_or_validate_manifest(
                out_root=temp,
                model_alias="llm_l_gemini35_flash_lite",
                resume=True,
            )
            self.assertEqual(path, path2)
            self.assertEqual(expected, actual)
            self.assertEqual(sha, sha2)

    def test_manifest_mismatch_blocks_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            path, payload, _sha = write_or_validate_manifest(
                out_root=temp,
                model_alias="llm_l_gemini35_flash_lite",
                resume=False,
            )
            payload["scenario_steps"] += 1
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                write_or_validate_manifest(
                    out_root=temp,
                    model_alias="llm_l_gemini35_flash_lite",
                    resume=True,
                )

    def test_cumulative_probe_summary_recomputes_full_jsonl(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "probe.jsonl"
            records = [
                probe_record(requested=1, canonical=0, reward=-1.0),
                probe_record(agent="blue_agent_1", requested=1, canonical=1, reward=-2.0),
            ]
            path.write_text("\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8")
            result = cumulative_probe_summary(path)
            self.assertEqual(result["record_count"], 2)
            self.assertEqual(result["requested_action_counts"]["1"], 2)
            self.assertEqual(result["canonical_action_counts"]["0"], 1)
            self.assertEqual(result["canonical_action_counts"]["1"], 1)
            self.assertEqual(result["fallback_count"], 1)
            self.assertEqual(result["plan0_match_count"], 2)
            self.assertAlmostEqual(result["response_reward_total"], -3.0)
            self.assertAlmostEqual(result["official_reward_total"], -6.0)

    def test_stage_above_2000_requires_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                run_strict_stage(
                    model_alias="llm_l_gemini35_flash_lite",
                    stage_target=5000,
                    out_root=temp,
                    resume=False,
                )

    def test_strict_stage_calls_low_level_with_only_frozen_config_and_enriches_report(self):
        with tempfile.TemporaryDirectory() as temp:
            probe = Path(temp) / "probe.jsonl"
            probe.write_text(json.dumps(probe_record()) + "\n", encoding="utf-8")
            fake = {
                "model_alias": "llm_l_gemini35_flash_lite",
                "exact_model_id": "google/gemini-3.5-flash-lite",
                "stage_target": 2000,
                "transition_count": 1,
                "probe_record_count": 1,
                "probe_path": str(probe),
                "episode_count": 1,
                "update_count": 1,
                "policy_version": 1,
                "cache_hits_cumulative": 0,
                "cache_misses_cumulative": 1,
                "live_api_calls_cumulative": 1,
                "estimated_cost_usd_cumulative": 0.01,
                "safe_update_barriers": True,
                "primary_model_selected_is_false": True,
                "pass": True,
            }
            with patch(
                "formal_experiments.evaluation.run_b4_provisional_stage.run_provisional_stage",
                return_value=fake,
            ) as mocked:
                report = run_strict_stage(
                    model_alias="llm_l_gemini35_flash_lite",
                    stage_target=2000,
                    out_root=temp,
                    resume=False,
                )
            kwargs = mocked.call_args.kwargs
            self.assertEqual(kwargs["scenario_steps"], DEFAULT_PROVISIONAL_SCENARIO_STEPS)
            self.assertEqual(kwargs["ppo_seed"], DEFAULT_PPO_SEED)
            self.assertFalse(kwargs["resume"])
            self.assertTrue(report["strict_entrypoint"])
            self.assertEqual(report["cumulative_probe_summary"]["record_count"], 1)
            self.assertEqual(len(report["strict_protocol_manifest_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
