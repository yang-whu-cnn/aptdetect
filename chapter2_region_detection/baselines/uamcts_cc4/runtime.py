"""Frozen production UAMCTS adapters."""
from __future__ import annotations
import math
import numpy as np
from formal_experiments.training.bootstrap_world_model import BootstrapProbabilisticWorldModel
from formal_experiments.training.response_reward_predictor import ResponseRewardPredictor
from shared.formal_state import BLUE_AGENTS
from .artifacts import FrozenProgressEnsemble, FrozenUncertaintyNormalizers, UAMCTSPrototypeRetriever
from .planner import UAMCTSPlanner, UAMCTSConfig
from .preflight import DEFAULT_PRIOR_ENTROPY, run_preflight

class WMAdapter:
    def __init__(self,model): self.model=model
    def predict_ensemble(self,state,action):
        return self.model.predict_ensemble(np.asarray(state,np.float32)[None],np.asarray([action]))["member_means"][:,0]
class RewardAdapter:
    def __init__(self,model): self.model=model
    def predict(self,state,action,next_state):
        return float(self.model.predict(np.asarray(state,np.float32)[None],np.asarray([action]),np.asarray(next_state,np.float32)[None])[0])
class AgentPrior:
    def __init__(self,retriever,agent): self.retriever=retriever; self.agent=agent; self.hits=0; self.misses=0
    def probabilities(self,state,actions):
        try: full=self.retriever.lookup(state,agent_name=self.agent).numpy().astype(np.float64); self.hits+=1
        except RuntimeError: self.misses+=1; raise
        selected=full[np.asarray(actions,dtype=int)]; return selected/selected.sum()

def build_runtime(*,world_path,reward_path,progress_path,prototype_path,normalizers_path,
                  prior_entropy_path=DEFAULT_PRIOR_ENTROPY,
                  device="cpu",simulations=64,planner_seed=73001):
    gate=run_preflight(world_model=world_path,reward_model=reward_path,progress=progress_path,
        normalizers=normalizers_path,prototype_artifact=prototype_path,
        prior_entropy_artifact=prior_entropy_path)
    if not gate["eligible"]:
        raise RuntimeError("UAMCTS artifact preflight failed: " + "; ".join(gate["errors"]))
    wm=WMAdapter(BootstrapProbabilisticWorldModel.load_checkpoint(world_path,device=device))
    reward=RewardAdapter(ResponseRewardPredictor.load_checkpoint(reward_path,device=device))
    progress=FrozenProgressEnsemble(progress_path,device=device); retriever=UAMCTSPrototypeRetriever(prototype_path)
    normalizers=FrozenUncertaintyNormalizers(normalizers_path); planners={}; priors={}
    for i,agent in enumerate(BLUE_AGENTS):
        priors[agent]=AgentPrior(retriever,agent)
        planners[agent]=UAMCTSPlanner(world_model=wm,reward_model=reward,progress_model=progress,
            prior_model=priors[agent],config=UAMCTSConfig(simulations=simulations),seed=int(planner_seed)+i,
            uncertainty_normalizers=normalizers)
    return planners,priors
