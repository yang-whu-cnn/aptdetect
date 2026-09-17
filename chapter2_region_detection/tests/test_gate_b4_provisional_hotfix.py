import json
import tempfile
import unittest
from pathlib import Path

from formal_experiments.evaluation.run_b4_provisional_stage import (
    write_or_validate_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
COLLECTOR = (
    ROOT
    / "formal_experiments"
    / "evaluation"
    / "run_b4_provisional_ppo.py"
)
MODEL_ALIAS = "llm_l_gemini35_flash_lite"


class TestGateB4ProvisionalHotfix(unittest.TestCase):
    def test_collector_registers_every_incident_bookkeeper(self):
        source = COLLECTOR.read_text(encoding="utf-8")
        self.assertIn("bookkeepers[agent] = bookkeeper", source)
        self.assertLess(
            source.index("bookkeeper.reset("),
            source.index("bookkeepers[agent] = bookkeeper"),
        )
        self.assertLess(
            source.index("bookkeepers[agent] = bookkeeper"),
            source.index("accounting = bookkeepers[agent].record_tick("),
        )

    def test_manifest_only_aborted_attempt_can_restart_fresh(self):
        with tempfile.TemporaryDirectory() as temp:
            first_path, first_payload, first_sha = write_or_validate_manifest(
                out_root=temp,
                model_alias=MODEL_ALIAS,
                resume=False,
            )
            second_path, second_payload, second_sha = write_or_validate_manifest(
                out_root=temp,
                model_alias=MODEL_ALIAS,
                resume=False,
            )
            self.assertEqual(first_path, second_path)
            self.assertEqual(first_payload, second_payload)
            self.assertEqual(first_sha, second_sha)

    def test_fresh_restart_is_blocked_once_checkpoint_exists(self):
        with tempfile.TemporaryDirectory() as temp:
            path, _payload, _sha = write_or_validate_manifest(
                out_root=temp,
                model_alias=MODEL_ALIAS,
                resume=False,
            )
            (path.parent / "resume.pt").write_bytes(b"development-checkpoint")
            with self.assertRaises(RuntimeError):
                write_or_validate_manifest(
                    out_root=temp,
                    model_alias=MODEL_ALIAS,
                    resume=False,
                )

    def test_fresh_restart_is_blocked_once_probe_exists(self):
        with tempfile.TemporaryDirectory() as temp:
            path, _payload, _sha = write_or_validate_manifest(
                out_root=temp,
                model_alias=MODEL_ALIAS,
                resume=False,
            )
            (path.parent / "probe_transitions.jsonl").write_text("{}\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                write_or_validate_manifest(
                    out_root=temp,
                    model_alias=MODEL_ALIAS,
                    resume=False,
                )

    def test_manifest_only_restart_still_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            path, payload, _sha = write_or_validate_manifest(
                out_root=temp,
                model_alias=MODEL_ALIAS,
                resume=False,
            )
            payload["scenario_steps"] = int(payload["scenario_steps"]) + 1
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                write_or_validate_manifest(
                    out_root=temp,
                    model_alias=MODEL_ALIAS,
                    resume=False,
                )


if __name__ == "__main__":
    unittest.main()
