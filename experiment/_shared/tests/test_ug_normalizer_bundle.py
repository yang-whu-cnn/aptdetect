import tempfile
import unittest
from pathlib import Path

import torch

from baselines.ug_cem_apt.normalizer_bundle import (
    EXPECTED_CALIBRATION_SEEDS,
    load_ug_normalizer_bundle,
    restore_agent_normalizer,
    snapshot_matches_bundle,
)
from baselines.ug_cem_apt.uncertainty import UGUncertainty, UGUncertaintyConfig
from shared.formal_state import BLUE_AGENTS, FORMAL_STATE_DIM


def bundle():
    return {
        "format_version": 1,
        "source_split": "calibration",
        "calibration_seeds": EXPECTED_CALIBRATION_SEEDS,
        "planner_calls_per_agent": 100,
        "freeze_after_warmup": False,
        "online_updates_after_warmup": True,
        "agents": {
            agent: {
                "obs_mean": torch.full((FORMAL_STATE_DIM,), 0.1 + i),
                "obs_std": torch.full((FORMAL_STATE_DIM,), 0.2 + i),
                "horizon_std": torch.full((4,), 0.3 + i),
            }
            for i, agent in enumerate(BLUE_AGENTS)
        },
    }


class TestUGNormalizerBundle(unittest.TestCase):

    def save_and_load(self, payload):
        tmp = tempfile.TemporaryDirectory()
        path = Path(tmp.name) / "bundle.pt"
        torch.save(payload, path)
        return tmp, path

    def test_load_valid_bundle(self):
        tmp, path = self.save_and_load(bundle())
        try:
            loaded = load_ug_normalizer_bundle(path)
        finally:
            tmp.cleanup()
        self.assertEqual(set(loaded["agents"]), set(BLUE_AGENTS))
        self.assertEqual(tuple(loaded["agents"]["blue_agent_0"]["obs_mean"].shape), (27,))
        self.assertEqual(tuple(loaded["agents"]["blue_agent_0"]["horizon_std"].shape), (4,))

    def test_rejects_wrong_split_seed_count_and_freeze_policy(self):
        cases = []
        x=bundle(); x["source_split"]="train"; cases.append(x)
        x=bundle(); x["calibration_seeds"]=(3000,); cases.append(x)
        x=bundle(); x["planner_calls_per_agent"]=99; cases.append(x)
        x=bundle(); x["freeze_after_warmup"]=True; cases.append(x)
        x=bundle(); x["online_updates_after_warmup"]=False; cases.append(x)
        for payload in cases:
            tmp, path = self.save_and_load(payload)
            try:
                with self.assertRaises(ValueError):
                    load_ug_normalizer_bundle(path)
            finally:
                tmp.cleanup()

    def test_rejects_missing_agent(self):
        x=bundle()
        del x["agents"]["blue_agent_4"]
        tmp, path = self.save_and_load(x)
        try:
            with self.assertRaises(ValueError):
                load_ug_normalizer_bundle(path)
        finally:
            tmp.cleanup()

    def test_rejects_bad_shapes_nonfinite_and_nonpositive_scale(self):
        bad=[]
        x=bundle(); x["agents"]["blue_agent_0"]["obs_mean"]=torch.zeros(26); bad.append(x)
        x=bundle(); x["agents"]["blue_agent_0"]["obs_std"]=torch.zeros(27); bad.append(x)
        x=bundle(); x["agents"]["blue_agent_0"]["horizon_std"]=torch.tensor([1.0,1.0,float("nan"),1.0]); bad.append(x)
        for payload in bad:
            tmp, path = self.save_and_load(payload)
            try:
                with self.assertRaises(ValueError):
                    load_ug_normalizer_bundle(path)
            finally:
                tmp.cleanup()

    def test_restore_sets_internal_shape_contract(self):
        x=bundle()
        u=UGUncertainty(UGUncertaintyConfig(device="cpu"))
        restore_agent_normalizer(u,bundle=x,agent_name="blue_agent_2")
        self.assertEqual(u._state_dim,27)
        self.assertEqual(u._horizon,4)
        self.assertTrue(snapshot_matches_bundle(u,bundle=x,agent_name="blue_agent_2"))

    def test_restore_is_agent_specific(self):
        x=bundle()
        u0=UGUncertainty(UGUncertaintyConfig(device="cpu"))
        u4=UGUncertainty(UGUncertaintyConfig(device="cpu"))
        restore_agent_normalizer(u0,bundle=x,agent_name="blue_agent_0")
        restore_agent_normalizer(u4,bundle=x,agent_name="blue_agent_4")
        self.assertFalse(torch.equal(u0.obs_mean,u4.obs_mean))

    def test_restored_normalizer_can_compute_without_reinitializing(self):
        x=bundle()
        u=UGUncertainty(UGUncertaintyConfig(device="cpu"))
        restore_agent_normalizer(u,bundle=x,agent_name="blue_agent_1")
        before=u.obs_mean.clone()
        values=torch.zeros(4,3,5,27)
        out=u.compute(values,update_stats=False)
        self.assertEqual(tuple(out.shape),(3,))
        torch.testing.assert_close(u.obs_mean,before)

    def test_snapshot_matches_false_before_restore(self):
        x=bundle()
        u=UGUncertainty(UGUncertaintyConfig(device="cpu"))
        self.assertFalse(snapshot_matches_bundle(u,bundle=x,agent_name="blue_agent_0"))


if __name__ == "__main__":
    unittest.main()
