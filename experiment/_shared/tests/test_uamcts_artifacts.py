import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from baselines.uamcts_cc4.artifacts import (
    FrozenProgressEnsemble, FrozenUncertaintyNormalizers, ProgressNet,
    build_future_progress_dataset, freeze_uncertainty_normalizers, sha256_file,
)


def row(seed, index, reward=0.0, done=False):
    return {"episode_seed":seed,"agent_name":"blue_agent_0","decision_index":index,
            "decision_dt":1,"state":[float(index)/10]+[0.0]*26,
            "response_reward":reward,"done":done,"requested_action_id":0}


class TestUAMCTSArtifacts(unittest.TestCase):
    def write_rows(self, path, rows):
        path.write_text("".join(json.dumps(x)+"\n" for x in rows),encoding="utf-8")

    def test_train_only_future_progress_and_test_seed_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"train.jsonl"; self.write_rows(path,[row(1000,0),row(1000,1,-2,True)])
            states,target,meta=build_future_progress_dataset(path)
            self.assertEqual(states.shape,(2,27)); self.assertEqual(meta["label"],"normalized_future_response_return")
            self.assertTrue(np.all((target>=0)&(target<=1)))
            self.write_rows(path,[row(4000,0)])
            with self.assertRaisesRegex(ValueError,"split"):
                build_future_progress_dataset(path)

    def test_progress_artifact_requires_five_train_members(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"progress.pt"
            payload={"schema":"uamcts_progress_ensemble_v1","split":"train",
                "allowed_seeds":list(range(1000,1032)),"ensemble_size":5,
                "members":[ProgressNet().state_dict() for _ in range(5)]}
            torch.save(payload,path); scorer=FrozenProgressEnsemble(path)
            mean,var=scorer.score(np.zeros(27,np.float32)); self.assertTrue(0<=mean<=1); self.assertGreaterEqual(var,0)
            payload["split"]="test"; torch.save(payload,path)
            with self.assertRaisesRegex(ValueError,"train-only"):
                FrozenProgressEnsemble(path)

    def test_three_source_validation_freeze_and_read_only_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); replay=root/"validation.jsonl"
            self.write_rows(replay,[row(2000,0),row(2001,0)])
            values={"world_model":[.1,.2],"progress":[.01,.03],"prior_entropy":[.5,.6]}
            out=root/"normalizers.pt"
            result=freeze_uncertainty_normalizers(validation_replay=replay,values=values,
                progress_sha256="a"*64,world_model_sha256="b"*64,
                prior_artifact_sha256="c"*64,output=out)
            frozen=FrozenUncertaintyNormalizers(out); before=sha256_file(out)
            frozen.normalize("progress",.02); self.assertEqual(before,sha256_file(out))
            self.assertFalse(frozen.payload["test_updates_allowed"])
            with self.assertRaisesRegex(ValueError,"three"):
                freeze_uncertainty_normalizers(validation_replay=replay,
                    values={"world_model":[1,2]},progress_sha256="a"*64,
                    world_model_sha256="b"*64,prior_artifact_sha256="c"*64,output=out)
            result=freeze_uncertainty_normalizers(validation_replay=replay,
                values={"world_model":[.1,.2],"progress":[.01,.02],
                        "prior_entropy":[.5,float("nan")]},
                progress_sha256="a"*64,world_model_sha256="b"*64,
                prior_artifact_sha256="c"*64,output=out,expected_record_count=2)
            self.assertEqual((result["record_count"],result["prior_misses"]),(1,1))
            self.write_rows(replay,[row(4000,0)])
            with self.assertRaisesRegex(ValueError,"split"):
                freeze_uncertainty_normalizers(validation_replay=replay,values={k:[1] for k in values},
                    progress_sha256="a"*64,world_model_sha256="b"*64,
                    prior_artifact_sha256="c"*64,output=out)

if __name__=="__main__": unittest.main()
