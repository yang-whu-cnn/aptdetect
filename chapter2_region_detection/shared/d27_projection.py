"""Versioned semantic projection for all model-generated formal D27 states."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib, json
import torch

PROJECTION_VERSION="formal_d27_semantic_projection_v1"
PROJECTION_RULES={"agent_0_4":"root_one_hot","tick_5":"(root_tick+elapsed)/episode_steps",
 "inventory_6":"root_immutable","binary_17_20":"threshold_ge_0.5",
 "success_23_26":"argmax_one_hot","continuous":"clip_0_1:[7..16,21,22]"}
PROJECTION_SHA256=hashlib.sha256(json.dumps({"version":PROJECTION_VERSION,"rules":PROJECTION_RULES},sort_keys=True,separators=(",",":")).encode()).hexdigest()

@dataclass(frozen=True)
class D27ProjectionContext:
    agent_index:int; root_tick:int; episode_steps:int; inventory_feature:float
    def __post_init__(self):
        if self.agent_index not in range(5) or self.root_tick<0 or self.episode_steps<=0 or not 0<=float(self.inventory_feature)<=1:
            raise ValueError("invalid D27 projection root context")
    @classmethod
    def from_root(cls,state,*,root_tick:int,episode_steps:int):
        x=torch.as_tensor(state); agent=int(torch.argmax(x[:5]).item())
        expected=torch.zeros(5); expected[agent]=1
        if not torch.equal(x[:5].cpu().to(torch.float32),expected): raise ValueError("root agent one-hot invalid")
        expected_tick=min(max(float(root_tick)/float(episode_steps),0.0),1.0)
        if abs(float(x[5])-expected_tick)>1e-5: raise ValueError("root tick fraction/context mismatch")
        return cls(agent,root_tick,episode_steps,float(x[6]))

def project_d27_tensor(raw,*,context:D27ProjectionContext,elapsed_ticks):
    x=torch.as_tensor(raw)
    if x.shape[-1]!=27 or not torch.isfinite(x).all(): raise ValueError("raw model state must be finite D27")
    y=torch.clamp(x,0,1).clone(); elapsed=torch.as_tensor(elapsed_ticks,device=y.device)
    if not torch.isfinite(elapsed.to(torch.float32)).all() or torch.any(elapsed<0): raise ValueError("elapsed ticks must be finite nonnegative")
    try: elapsed=torch.broadcast_to(elapsed,y.shape[:-1])
    except RuntimeError as exc: raise ValueError("elapsed ticks are not broadcastable to model-state batch") from exc
    y[...,0:5]=0; y[...,context.agent_index]=1
    y[...,5]=torch.clamp((elapsed.to(y.dtype)+context.root_tick)/context.episode_steps,0,1)
    y[...,6]=context.inventory_feature; y[...,17:21]=(y[...,17:21]>=.5).to(y.dtype)
    idx=torch.argmax(y[...,23:27],dim=-1,keepdim=True); y[...,23:27]=0; y[...,23:27].scatter_(-1,idx,1)
    return y
