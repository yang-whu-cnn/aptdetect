import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from baselines.priorrl_cc4.freeze_prototype_artifact import (
    _atomic_commit_json_set, freeze_pre_supplement_validation_threshold,
)
from baselines.priorrl_cc4.prototype_retrieval import (
    PrototypeInput, fit_train_prototypes, freeze_validation_radius, validate_payload,
)
from shared.formal_state import BLUE_AGENTS


class TestPriorRLFreezeThreshold(unittest.TestCase):
    @staticmethod
    def digest(label: str) -> str:
        return hashlib.sha256(label.encode("utf-8")).hexdigest()

    def fixtures(self):
        records = []
        validation = []
        for index, agent in enumerate(BLUE_AGENTS):
            state = np.zeros(27, dtype=np.float32); state[index] = 1.0
            validation.append((agent, state.copy()))
            records.append(PrototypeInput(
                state=state, agent_name=agent,
                action_prior=np.full(4, 0.25, dtype=np.float32),
                cache_key=f"{index + 1:x}" * 64,
                cache_entry_sha256=f"{index + 6:x}" * 64,
                plans=np.zeros((6, 4), dtype=np.int64),
                raw_prior_scores=np.full(6, 0.5, dtype=np.float32),
                prior_preferences=np.full(6, 1 / 6, dtype=np.float32),
                sources=("llm",) * 6,
            ))
        payload = fit_train_prototypes(records, source_sha256=self.digest("source"))
        threshold = {
            "radius": 2.560749706662155, "weights": payload["weights"],
            "scaler_by_agent": {
                agent: {"mean": payload["agents"][agent]["scaler_mean"],
                        "scale": payload["agents"][agent]["scaler_scale"]}
                for agent in BLUE_AGENTS
            },
            "threshold_source": "validation_pre_supplement",
            "validation_source_sha256": self.digest("validation"),
            "sha256": self.digest("threshold audit"),
        }
        supplement = {
            "supplement_train_only": True, "provider_calls": 6,
            "manifest_file_sha256": self.digest("manifest file"),
            "logical_manifest_sha256": self.digest("logical manifest"),
            "entry_sha256_set_sha256": self.digest("entry set"),
            "online_allowed": False, "runtime_mode": "offline_read_only",
        }
        base_frozen = freeze_validation_radius(
            payload, validation, validation_source_sha256=self.digest("validation"),
        )
        base_binding = {
            "artifact_file_sha256": self.digest("artifact file"),
            "prototype_sha256": base_frozen["prototype_sha256"],
            "migration": "v1_to_v2_recomputed",
            "source_v1_recomputed_file_sha256": self.digest("v1 file"),
            "source_v1_recomputed_prototype_sha256": self.digest("v1 prototype"),
        }
        supplement_entries = [
            {"cache_key": self.digest("cache key"),
             "entry_sha256": self.digest("cache entry"),
             "agent_name": "blue_agent_4"}
        ]
        return (payload, validation, threshold, supplement, base_frozen,
                base_binding, supplement_entries)

    def test_preserves_pre_supplement_validation_geometry_and_audit(self):
        (payload, validation, threshold, supplement, base_frozen,
         base_binding, supplement_entries) = self.fixtures()
        frozen = freeze_pre_supplement_validation_threshold(
            payload, validation, threshold=threshold, supplement=supplement,
            base_frozen=base_frozen, base_binding=base_binding,
            supplement_entries=supplement_entries,
        )
        validate_payload(frozen, require_frozen=True)
        self.assertEqual(frozen["radius"], 2.560749706662155)
        self.assertEqual(frozen["threshold_source"], "validation_pre_supplement")
        self.assertTrue(frozen["supplement_train_only"])
        self.assertEqual(frozen["provider_calls"], 6)
        self.assertFalse(frozen["online_allowed"])
        self.assertEqual(frozen["supplement_logical_manifest_sha256"],
                         self.digest("logical manifest"))
        self.assertEqual(frozen["supplement_entry_sha256_set_sha256"],
                         self.digest("entry set"))

    def test_rejects_threshold_that_no_longer_covers_validation(self):
        (payload, validation, threshold, supplement, base_frozen,
         base_binding, supplement_entries) = self.fixtures()
        far = validation[0][1].copy(); far[5] = 1.0
        validation[0] = (validation[0][0], far)
        threshold["radius"] = 0.0
        with self.assertRaisesRegex(RuntimeError, "violate"):
            freeze_pre_supplement_validation_threshold(
                payload, validation, threshold=threshold, supplement=supplement,
                base_frozen=base_frozen, base_binding=base_binding,
                supplement_entries=supplement_entries,
            )

    def test_three_file_commit_failure_rolls_back_every_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            finals = [root / "artifact.json", root / "coverage.json",
                      root / "provenance.json"]
            originals = []
            for index, final in enumerate(finals):
                final.write_text(json.dumps({"old": index}) + "\n", encoding="utf-8")
                originals.append(final.read_bytes())
            real_replace = os.replace
            commit_count = 0

            def fail_second_commit(source, destination):
                nonlocal commit_count
                if str(source).endswith(".tmp"):
                    commit_count += 1
                    if commit_count == 2:
                        raise OSError("injected atomic commit failure")
                return real_replace(source, destination)

            items = [(final, {"new": index}, False)
                     for index, final in enumerate(finals)]
            with patch("baselines.priorrl_cc4.freeze_prototype_artifact.os.replace",
                       side_effect=fail_second_commit):
                with self.assertRaisesRegex(OSError, "injected"):
                    _atomic_commit_json_set(items)
            self.assertEqual([final.read_bytes() for final in finals], originals)
            self.assertEqual(list(root.glob(".*.tmp")), [])
            self.assertEqual(list(root.glob(".*.restore")), [])


if __name__ == "__main__":
    unittest.main()
