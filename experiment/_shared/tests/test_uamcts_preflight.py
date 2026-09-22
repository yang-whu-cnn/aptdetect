import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from baselines.uamcts_cc4 import preflight


class _Progress:
    sha256 = "p" * 64


class _Normalizers:
    sha256 = "n" * 64
    def __init__(self, _path):
        self.payload = {"progress_sha256": "p" * 64, "world_model_sha256": "w" * 64,
            "prior_artifact_sha256": "e" * 64, "validation_aligned_total": 2,
            "record_count": 2, "prior_coverage_hits": 2, "prior_coverage_misses": 0}


class _Prototype:
    file_sha256 = "t" * 64
    payload = {"prototype_sha256": "q" * 64}


class TestUAMCTSPreflight(unittest.TestCase):
    def test_defaults_include_runtime_prior_and_calibration_evidence(self):
        self.assertIn("frozen_prototypes.json", preflight.DEFAULT_PROTOTYPES)
        self.assertIn("validation_prior_entropy.json", preflight.DEFAULT_PRIOR_ENTROPY)

    def test_stale_prototype_binding_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("world.pt", "reward.pt", "progress.pt", "normalizers.pt", "prototype.json"):
                (root / name).write_bytes(b"x")
            entropy = root / "entropy.json"
            entropy.write_text(json.dumps({"schema": "uamcts_validation_prior_entropy_v1",
                "split": "validation", "allowed_seeds": list(range(2000, 2008)),
                "provider_calls": 0, "prototype_artifact_sha256": "old",
                "prototype_sha256": "q" * 64, "aligned_unique_records": 2,
                "hits": 2, "misses": 0, "entropy": [0.1, 0.2]}), encoding="utf-8")
            hashes = {root / "world.pt": "w" * 64, entropy: "e" * 64}
            with patch.object(preflight, "FrozenProgressEnsemble", return_value=_Progress()), \
                    patch.object(preflight, "FrozenUncertaintyNormalizers", _Normalizers), \
                    patch.object(preflight, "UAMCTSPrototypeRetriever", return_value=_Prototype()), \
                    patch.object(preflight, "sha256_file", side_effect=lambda path: hashes.get(Path(path), "x" * 64)):
                report = preflight.run_preflight(world_model=root / "world.pt",
                    reward_model=root / "reward.pt", progress=root / "progress.pt",
                    normalizers=root / "normalizers.pt", prototype_artifact=root / "prototype.json",
                    prior_entropy_artifact=entropy)
            self.assertFalse(report["eligible"])
            self.assertIn("prior entropy prototype_artifact_sha256 provenance mismatch", report["errors"])

    def test_refrozen_matching_chain_passes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("world.pt", "reward.pt", "progress.pt", "normalizers.pt", "prototype.json"):
                (root / name).write_bytes(b"x")
            entropy = root / "entropy.json"
            entropy.write_text(json.dumps({"schema": "uamcts_validation_prior_entropy_v1",
                "split": "validation", "allowed_seeds": list(range(2000, 2008)),
                "provider_calls": 0, "prototype_artifact_sha256": "t" * 64,
                "prototype_sha256": "q" * 64, "aligned_unique_records": 2,
                "hits": 2, "misses": 0, "entropy": [0.1, 0.2]}), encoding="utf-8")
            hashes = {root / "world.pt": "w" * 64, entropy: "e" * 64}
            with patch.object(preflight, "FrozenProgressEnsemble", return_value=_Progress()), \
                    patch.object(preflight, "FrozenUncertaintyNormalizers", _Normalizers), \
                    patch.object(preflight, "UAMCTSPrototypeRetriever", return_value=_Prototype()), \
                    patch.object(preflight, "sha256_file", side_effect=lambda path: hashes.get(Path(path), "x" * 64)):
                report = preflight.run_preflight(world_model=root / "world.pt",
                    reward_model=root / "reward.pt", progress=root / "progress.pt",
                    normalizers=root / "normalizers.pt", prototype_artifact=root / "prototype.json",
                    prior_entropy_artifact=entropy)
            self.assertTrue(report["eligible"], report["errors"])


if __name__ == "__main__": unittest.main()
