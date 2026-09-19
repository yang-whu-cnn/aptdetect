import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from baselines.priorrl_cc4.build_supplemental_cache import load_records
from formal_experiments.ours.prior_cache import exact_state_sha256


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
        probe.write_text(json.dumps({
            "schema": "priorrl_online_prototype_coverage_probe_v1", "status": "BLOCKED",
            "episode_seed": online_seed, "policy_seed": 61001, "agent_name": "blue_agent_4",
            "state": online_state.tolist(), "state_sha256": exact_state_sha256(online_state),
            "report_sha256": "a" * 64,
        }), encoding="utf-8")
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


if __name__ == "__main__":
    unittest.main()
