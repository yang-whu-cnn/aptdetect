import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from baselines.priorrl_cc4.build_supplemental_cache import load_records
from baselines.priorrl_cc4.freeze_prototype_artifact import (
    _canonical_sha, load_verified_supplement,
)
from formal_experiments.evaluation.build_final_prior_cache import (
    EXACT_MODEL_ID, GENERATION_CONFIG, MODEL_ALIAS, PROMPT_VERSION, PROVIDER,
)
from formal_experiments.ours.prior_cache import exact_state_sha256, make_identity


class TestPriorRLSupplementalCache(unittest.TestCase):
    def _fixtures(self, root: Path, *, online_seed=1000):
        replay = root / "train.jsonl"
        rows = []
        selections = []
        for index in range(5):
            state = np.zeros(27, np.float32); state[index] = 1; state[17] = index % 2
            row = {"agent_name": f"blue_agent_{index}", "episode_seed": 1000 + index,
                   "requested_action_id": index % 4, "state": state.tolist()}
            rows.append(row)
            selections.append({"source_index": index, "episode_seed": 1000 + index,
                "agent_name": f"blue_agent_{index}", "state": state.tolist(),
                "state_sha256": exact_state_sha256(state)})
        replay.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        selection = root / "selection.json"
        selection.write_text(json.dumps({"selection_count": 5, "selections": selections,
            "train_replay": {"file_sha256": hashlib.sha256(replay.read_bytes()).hexdigest()}}),
            encoding="utf-8")
        online_state = np.zeros(27, np.float32); online_state[4] = 1; online_state[17] = 1
        probe = root / "probe.json"
        probe_payload = {
            "schema": "priorrl_online_prototype_coverage_probe_v1", "status": "BLOCKED",
            "episode_seed": online_seed, "policy_seed": 61001, "agent_name": "blue_agent_4",
            "state": online_state.tolist(), "state_sha256": exact_state_sha256(online_state),
        }
        probe_payload["report_sha256"] = hashlib.sha256(
            json.dumps(probe_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        probe.write_text(json.dumps(probe_payload), encoding="utf-8")
        return selection, probe, replay

    def test_exact_five_replay_plus_one_train_probe(self):
        with tempfile.TemporaryDirectory() as temp:
            selection, probe, replay = self._fixtures(Path(temp))
            expected_replay_sha = hashlib.sha256(replay.read_bytes()).hexdigest()
            records, provenance, inputs = load_records(selection, probe, replay)
        self.assertEqual(len(records), 6); self.assertEqual(len(provenance), 6)
        self.assertEqual(provenance[-1]["origin"], "fail_closed_on_policy_train_probe")
        self.assertEqual(len({(x.agent_name, x.state_sha256) for x in records}), 6)
        self.assertEqual(inputs["train_replay_sha256"], expected_replay_sha)

    def test_validation_or_test_probe_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            selection, probe, replay = self._fixtures(Path(temp), online_seed=2000)
            with self.assertRaisesRegex(ValueError, "train seed"):
                load_records(selection, probe, replay)

    def test_source_row_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            selection, probe, replay = self._fixtures(Path(temp))
            payload = json.loads(selection.read_text(encoding="utf-8"))
            payload["selections"][0]["agent_name"] = "blue_agent_4"
            selection.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no longer matches"):
                load_records(selection, probe, replay)

    def _verified_manifest(self, root: Path):
        selection, probe, replay = self._fixtures(root)
        records, provenance, _ = load_records(selection, probe, replay)
        registry_sha = hashlib.sha256(b"registry fixture").hexdigest()
        keys = [make_identity(
            split="train", model_alias=MODEL_ALIAS, provider=PROVIDER,
            exact_model_id=EXACT_MODEL_ID, registry_version=1,
            registry_sha256=registry_sha, prompt_version=PROMPT_VERSION,
            agent_name=row.agent_name, generation_config=GENERATION_CONFIG,
            state=row.state,
        ).cache_key for row in records]
        payload = {
            "schema": "formal_prior_cache_train_supplement_v1",
            "status": "PASS", "mode": "live", "split": "train",
            "target_size": 6, "provider_calls": 6, "generated": 6,
            "verified_entries": 6, "failed_entries": 0,
            "registry_sha256": registry_sha, "cache_keys": keys,
            "selection_sha256": _canonical_sha(provenance),
            "entry_sha256_set_sha256": hashlib.sha256(b"entry set fixture").hexdigest(),
            "inputs": {
                "selection_path": str(selection), "online_probe_path": str(probe),
                "train_replay_path": str(replay),
            },
        }
        payload["logical_manifest_sha256"] = _canonical_sha(payload)
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        return manifest, replay, payload

    def test_verified_manifest_binds_six_train_only_offline_entries(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest, replay, _ = self._verified_manifest(Path(temp))
            records, audit = load_verified_supplement(manifest, train=replay)
        self.assertEqual(len(records), 6)
        self.assertEqual(audit["provider_calls"], 6)
        self.assertTrue(audit["supplement_train_only"])
        self.assertFalse(audit["online_allowed"])
        self.assertEqual(audit["runtime_mode"], "offline_read_only")

    def test_rejects_incomplete_tampered_or_non_train_provider_audit(self):
        mutations = {
            "not live": lambda p: p.update(mode="dry_run"),
            "not train": lambda p: p.update(split="validation"),
            "provider calls": lambda p: p.update(provider_calls=5),
            "generated": lambda p: p.update(generated=5),
            "verified": lambda p: p.update(verified_entries=5),
            "failed": lambda p: p.update(failed_entries=1),
            "target size": lambda p: p.update(target_size=5),
            "entry hash": lambda p: p.update(entry_sha256_set_sha256="bad"),
            "logical hash": lambda p: p.update(logical_manifest_sha256="bad"),
            "valid-format wrong logical hash": lambda p: p.update(
                logical_manifest_sha256=hashlib.sha256(b"wrong payload").hexdigest()
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temp:
                manifest, replay, payload = self._verified_manifest(Path(temp))
                mutate(payload); manifest.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_verified_supplement(manifest, train=replay)


if __name__ == "__main__":
    unittest.main()
