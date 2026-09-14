# src/env_region.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from src.cc4_client import MockCC4Client, ReplayCC4Client, OnlineCC4Client, CC4ClientBase
from src.state_summary import SlidingWindowSummarizer, FEATURE_KEYS, StateSummary
from src.llm_prior import build_llm_prior, softmax
from src.plan_eval import evaluate_plan_with_model
from src.action_space import ACTION_SPACE
from src.world_model import DynamicsEnsemble, EnsembleConfig
from src.ppo_posterior import PPOPosteriorAgent, PPOConfig


def build_client(mode: str, region_id: int, cfg: Dict) -> CC4ClientBase:
    mode = str(mode).lower()
    if mode == "mock":
        return MockCC4Client(region_id=region_id, cfg=cfg)
    if mode == "replay":
        raw_dir = cfg["paths"]["raw_dir"]
        path = f"{raw_dir}/cc4_export_region{region_id}.jsonl"
        return ReplayCC4Client(path, cfg=cfg, recompute_reward=True)
    if mode == "online":
        return OnlineCC4Client(region_id=region_id, cfg=cfg)
    raise ValueError(f"Unknown mode={mode}")


@dataclass
class RegionObjects:
    client: CC4ClientBase
    summarizer: SlidingWindowSummarizer
    llm_prior: object
    world_model: DynamicsEnsemble
    ppo: PPOPosteriorAgent


def build_region_objects(cfg: Dict, region_id: int, device: str = "cpu") -> RegionObjects:
    client = build_client(cfg["mode"], region_id, cfg)
    summarizer = SlidingWindowSummarizer(
        window_size=int(cfg["summary"]["window_size"]),
        max_text_len=int(cfg["summary"]["max_text_len"]),
    )
    llm_prior = build_llm_prior(cfg)

    # world model
    state_dim = len(FEATURE_KEYS)
    n_actions = len(ACTION_SPACE)
    wm_cfg = cfg["world_model"]
    ens_cfg = EnsembleConfig(
        ensemble_size=int(wm_cfg["ensemble_size"]),
        hidden_dim=int(wm_cfg["hidden_dim"]),
        lr=float(wm_cfg["lr"]),
        batch_size=int(wm_cfg["batch_size"]),
        epochs=int(wm_cfg["epochs"]),
    )
    world_model = DynamicsEnsemble(state_dim=state_dim, n_actions=n_actions, cfg=ens_cfg, device=device)

    # PPO
    ppo_cfg = cfg["ppo"]
    ppo_config = PPOConfig(
        lr=float(ppo_cfg["lr"]),
        clip_eps=float(ppo_cfg["clip_eps"]),
        entropy_coef=float(ppo_cfg["entropy_coef"]),
        value_coef=float(ppo_cfg["value_coef"]),
        gae_lambda=float(ppo_cfg["gae_lambda"]),
        gamma=float(ppo_cfg["gamma"]),
        rollout_len=int(ppo_cfg["rollout_len"]),
        minibatch_size=int(ppo_cfg["minibatch_size"]),
        update_epochs=int(ppo_cfg["update_epochs"]),
    )
    n_candidates = int(cfg["llm_prior"]["k_candidates"])
    ppo = PPOPosteriorAgent(n_candidates=n_candidates, cfg=ppo_config, device=device)

    return RegionObjects(client=client, summarizer=summarizer, llm_prior=llm_prior, world_model=world_model, ppo=ppo)


def build_evidence(cfg: Dict, world_model: DynamicsEnsemble, summary: StateSummary, prior_out) -> Tuple[np.ndarray, np.ndarray, List[List[int]], np.ndarray]:
    plans = prior_out.plans
    scores = prior_out.prior_scores
    checkpoints = prior_out.checkpoints
    temp = float(cfg["llm_prior"]["temperature"])

    prior_prob = softmax(scores, temperature=temp)
    prior_logits = np.log(np.array(prior_prob, dtype=np.float32) + 1e-12)

    plan_ranks = np.argsort(np.argsort(-np.array(scores)))  # 分数越高排名越靠前（0为最高）

    E_list = []
    for plan in plans:
        ev = evaluate_plan_with_model(
            model=world_model,
            s0=summary.vec,
            plan=plan,
            checkpoints=checkpoints,
            cfg=cfg,
        )
        E_list.append(ev.as_vec())
    E = np.stack(E_list, axis=0).astype(np.float32)  # (N,4)
    return prior_logits, E, plans, plan_ranks
