import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from formal_experiments.ours.reward_ablation import RewardMode
from formal_experiments.ours.reward_artifact_contract import (
    FAIL_ONLY_CHECKPOINT_SHA256,
    FAIL_ONLY_MANIFEST_FORMAT,
    FAIL_ONLY_MANIFEST_SHA256,
    validate_frozen_reward_artifact,
)


def approved_manifest(checkpoint: Path) -> dict:
    return {
        "format": FAIL_ONLY_MANIFEST_FORMAT,
        "mode": "Fail-Only",
        "reward_model_eligible_for_policy_training": True,
        "formal_result_eligible": False,
        "test_seeds_used": False,
        "architecture": {"component_config": {
            "loss": "smooth_l1", "smooth_l1_beta": 1.0,
        }},
        "formal_contract_architecture_match": True,
        "formal_contract_training_match": True,
        "diagnostic_hurdle_used_for_formal_artifact": False,
        "training_loss": {
            "name": "smooth_l1", "beta": 1.0,
            "target_space": "standardized_reward_label", "reduction": "mean",
            "sample_weighting": "none", "resampling": "none",
        },
        "quality_gate": {"pass": True},
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": FAIL_ONLY_CHECKPOINT_SHA256,
        "frozen": True,
    }


class TestRewardArtifactContract(unittest.TestCase):
    def validate(self, payload):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        root = Path(temporary.name); checkpoint = root / "response_reward_predictor.pt"
        checkpoint.write_bytes(b"frozen-checkpoint")
        payload = {**payload, "checkpoint": str(checkpoint)}
        manifest = root / "frozen_manifest.json"
        manifest.write_text(json.dumps(payload), encoding="utf-8")

        def digest(path):
            return (FAIL_ONLY_MANIFEST_SHA256
                    if Path(path).name == "frozen_manifest.json"
                    else FAIL_ONLY_CHECKPOINT_SHA256)

        with patch(
            "formal_experiments.ours.reward_artifact_contract.sha256_file",
            side_effect=digest,
        ):
            return validate_frozen_reward_artifact(
                manifest, mode=RewardMode.FAIL_ONLY, project_root=root
            )

    def test_approved_fail_only_huber_contract_passes(self):
        payload, _, audit = self.validate(approved_manifest(Path("unused")))
        self.assertEqual(payload["training_loss"]["name"], "smooth_l1")
        self.assertTrue(audit["pass"])

    def test_rejects_legacy_mse_weighted_or_resampled_artifacts(self):
        mutations = {
            "legacy schema": lambda p: p.update(format="table3_reward_ablation_v1"),
            "mse": lambda p: p["training_loss"].update(name="mse"),
            "wrong beta": lambda p: p["training_loss"].update(beta=0.5),
            "weighted": lambda p: p["training_loss"].update(
                sample_weighting="positive_class", positive_weight=4.0
            ),
            "resampled": lambda p: p["training_loss"].update(
                resampling="positive_oversample", oversample_factor=8
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                payload = approved_manifest(Path("unused")); mutate(payload)
                with self.assertRaisesRegex(RuntimeError, "frozen reward contract FAIL"):
                    self.validate(payload)

    def test_rejects_failed_quality_gate_or_checkpoint_identity(self):
        for field in ("quality", "checkpoint"):
            with self.subTest(field=field):
                payload = approved_manifest(Path("unused"))
                if field == "quality": payload["quality_gate"]["pass"] = False
                else: payload["checkpoint_sha256"] = "0" * 64
                with self.assertRaisesRegex(RuntimeError, "frozen reward contract FAIL"):
                    self.validate(payload)


if __name__ == "__main__":
    unittest.main()
