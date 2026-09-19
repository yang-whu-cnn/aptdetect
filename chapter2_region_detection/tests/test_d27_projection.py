import re, unittest
import torch
from shared.d27_projection import (D27ProjectionContext, PROJECTION_SHA256,
                                   PROJECTION_VERSION, project_d27_tensor)

class TestD27Projection(unittest.TestCase):
    def test_1700_node_invariants_determinism_and_raw_audit(self):
        g=torch.Generator().manual_seed(73001); raw=torch.randn(1700,27,generator=g)*3
        root=torch.zeros(27); root[2]=1; root[5]=.07; root[6]=.9
        context=D27ProjectionContext.from_root(root,root_tick=7,episode_steps=100)
        elapsed=torch.arange(1700)%17
        first=project_d27_tensor(raw,context=context,elapsed_ticks=elapsed)
        second=project_d27_tensor(raw.clone(),context=context,elapsed_ticks=elapsed.clone())
        torch.testing.assert_close(first,second,rtol=0,atol=0)
        self.assertTrue(torch.all((first>=0)&(first<=1)))
        self.assertTrue(torch.all(first[:,2]==1)); self.assertTrue(torch.all(first[:,:2]==0)); self.assertTrue(torch.all(first[:,3:5]==0))
        self.assertTrue(torch.all(first[:,6]==.9)); self.assertTrue(torch.all((first[:,17:21]==0)|(first[:,17:21]==1)))
        self.assertTrue(torch.all(first[:,23:27].sum(-1)==1))
        self.assertGreater(int(((raw<0)|(raw>1)).sum()),0)

    def test_version_hash_and_context_fail_closed(self):
        self.assertEqual(PROJECTION_VERSION,"formal_d27_semantic_projection_v1")
        self.assertRegex(PROJECTION_SHA256,r"^[0-9a-f]{64}$")
        with self.assertRaisesRegex(ValueError,"one-hot"):
            D27ProjectionContext.from_root(torch.zeros(27),root_tick=0,episode_steps=100)
        root=torch.zeros(27); root[0]=1; root[5]=.5; root[6]=.2
        with self.assertRaisesRegex(ValueError,"tick fraction"):
            D27ProjectionContext.from_root(root,root_tick=0,episode_steps=100)
        context=D27ProjectionContext.from_root(root,root_tick=50,episode_steps=100)
        with self.assertRaisesRegex(ValueError,"broadcastable"):
            project_d27_tensor(torch.zeros(2,3,27),context=context,elapsed_ticks=torch.zeros(4))

if __name__=="__main__": unittest.main()
