import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

import formal_experiments.ours.reward_provenance as reward_provenance
from formal_experiments.ours.reward_provenance import (
    canonical_sha256, checkpoint_normalizer_digests,
    find_unique_upstream_manifest, sha256_file, write_immutable_sidecar,
    validate_candidate_fail_only_provenance, validate_fail_only_provenance,
)


class TestRewardProvenance(unittest.TestCase):
    def test_seed_list_hash_is_canonical_and_order_sensitive(self):
        self.assertEqual(canonical_sha256([1000, 1001]), canonical_sha256([1000, 1001]))
        self.assertNotEqual(canonical_sha256([1000, 1001]), canonical_sha256([1001, 1000]))

    def test_upstream_manifest_discovery_fails_closed_on_missing_or_ambiguous(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary); docs = root / "docs"; docs.mkdir()
            output = root / "outputs/replay"; output.mkdir(parents=True)
            with self.assertRaisesRegex(RuntimeError, "no unique existing manifest"):
                find_unique_upstream_manifest(
                    project_root=root, output_directory=output, label="replay",
                )
            value = {"output_directory": "outputs/replay"}
            (docs / "A_MANIFEST.json").write_text(json.dumps(value), encoding="utf-8")
            path, payload = find_unique_upstream_manifest(
                project_root=root, output_directory=output, label="replay",
            )
            self.assertEqual(path.name, "A_MANIFEST.json"); self.assertEqual(payload, value)
            (docs / "B_MANIFEST.json").write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "found 2"):
                find_unique_upstream_manifest(
                    project_root=root, output_directory=output, label="replay",
                )

    def test_checkpoint_normalizer_digest_is_stable_and_detects_change(self):
        payload = {
            "format_version": 1, "config": {"loss": "smooth_l1"},
            "state_normalizer": {
                "mean": np.asarray([1, 2], dtype=np.float32),
                "std": np.asarray([3, 4], dtype=np.float32), "eps": 1e-6,
            },
            "reward_normalizer": {"mean": 0.5, "std": 2.0, "eps": 1e-6},
        }
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"; torch.save(payload, path)
            first = checkpoint_normalizer_digests(path)
            self.assertEqual(first, checkpoint_normalizer_digests(path))
            payload["reward_normalizer"]["std"] = 3.0; torch.save(payload, path)
            second = checkpoint_normalizer_digests(path)
        self.assertNotEqual(first["reward_normalizer_sha256"], second["reward_normalizer_sha256"])
        self.assertNotEqual(first["combined_sha256"], second["combined_sha256"])

    def test_immutable_sidecar_refuses_different_overwrite(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "sidecar.json"
            digest = write_immutable_sidecar(path, {"a": 1})
            self.assertEqual(digest, sha256_file(path))
            self.assertEqual(write_immutable_sidecar(path, {"a": 1}), digest)
            with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
                write_immutable_sidecar(path, {"a": 2})

    def test_provenance_validator_rejects_unapproved_sidecar_sha_first(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "sidecar.json"
            path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "provenance SHA mismatch"):
                validate_candidate_fail_only_provenance(
                    path, project_root=Path(temporary), expected_sha256="0" * 64,
                )

    def test_formal_validator_stays_blocked_without_clean_commit_approval(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "sidecar.json"; path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "no clean-commit approved"):
                validate_fail_only_provenance(path, project_root=Path(temporary))

    def test_final_binding_requires_exact_candidate_sources_and_inputs(self):
        sections = {
            name: {"value": name}
            for name in reward_provenance.SNAPSHOT_MATCH_SECTIONS
        }
        sections["training_source_snapshot"] = {"aggregate_sha256": "source-aggregate"}
        candidate = {
            **sections,
            "format": reward_provenance.FORMAT,
            "status": "candidate",
            "mode": "Fail-Only",
            "current_clean": False,
            "formal_result_eligible": False,
            "test_seeds_used": False,
        }
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / reward_provenance.CANDIDATE_PROVENANCE_PATH
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(candidate), encoding="utf-8")
            digest = sha256_file(path)
            with (
                patch.object(
                    reward_provenance,
                    "CANDIDATE_FAIL_ONLY_PROVENANCE_SHA256",
                    digest,
                ),
                patch.object(
                    reward_provenance,
                    "CANDIDATE_TRAINING_SOURCE_AGGREGATE_SHA256",
                    "source-aggregate",
                ),
            ):
                binding = reward_provenance._bind_training_snapshot_candidate(
                    root, dict(sections),
                )
                self.assertEqual(binding["candidate_sha256"], digest)
                changed = dict(sections)
                changed["replay"] = {"value": "changed"}
                with self.assertRaisesRegex(RuntimeError, "matches training-snapshot"):
                    reward_provenance._bind_training_snapshot_candidate(root, changed)


if __name__ == "__main__":
    unittest.main()
