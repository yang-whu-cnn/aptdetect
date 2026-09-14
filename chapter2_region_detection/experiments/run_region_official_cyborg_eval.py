from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path
from typing import Any, Dict

import numpy as np

# ---- robust project/package bootstrap ----
THIS = Path(__file__).resolve()
PROJ = THIS.parents[1]
SRC = PROJ / 'src'
EXPS = PROJ / 'experiments'
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))
if 'src' not in sys.modules:
    m = types.ModuleType('src')
    m.__path__ = [str(SRC)]
    sys.modules['src'] = m
if 'experiments' not in sys.modules:
    m = types.ModuleType('experiments')
    m.__path__ = [str(EXPS)]
    sys.modules['experiments'] = m

CYBORG_ROOT = os.environ.get('CYBORG_ROOT', r'D:\python1\cage-challenge-4')
if CYBORG_ROOT and CYBORG_ROOT not in sys.path:
    sys.path.insert(0, CYBORG_ROOT)

from src.utils import load_yaml, set_seed  # type: ignore
from src.env_region import build_region_objects, build_evidence  # type: ignore
from experiments.run_end2end_collab_cc4 import (  # type: ignore
    _safe_to_dict,
    _extract_events_and_risk,
    _host_score_map,
    _pick_best_action,
)

from CybORG import CybORG  # type: ignore
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator  # type: ignore
from CybORG.Agents import SleepAgent, FiniteStateRedAgent, EnterpriseGreenAgent  # type: ignore
from CybORG.Agents.Wrappers.BlueFixedActionWrapper import BlueFixedActionWrapper  # type: ignore

ACTION_TYPE_MAP = {
    0: 'Monitor',
    1: 'Analyse',
    2: 'Analyse',
    3: 'Remove',
    4: 'Restore',
}


def write_jsonl(path: Path, obj: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')


def main():
    ap = argparse.ArgumentParser(description='Second chapter single-region official reward eval on CybORG/CC4')
    ap.add_argument('--cfg', default='configs/cc4.yaml')
    ap.add_argument('--region_id', type=int, required=True)
    ap.add_argument('--episodes', type=int, default=20)
    ap.add_argument('--steps', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--deterministic', action='store_true')
    ap.add_argument('--jsonl_path', default='outputs/official_eval/region_eval.jsonl')
    args = ap.parse_args()

    cfg_path = Path(args.cfg)
    if not cfg_path.is_absolute():
        cfg_path = (PROJ / args.cfg).resolve()
    cfg = load_yaml(str(cfg_path))
    set_seed(int(args.seed))

    # IMPORTANT: pass CONFIG DICT, not cfg path string
    objs = build_region_objects(cfg, region_id=int(args.region_id), device=args.device)

    ckpt_dir = (PROJ / cfg['paths']['ckpt_dir']).resolve()
    wm_path = ckpt_dir / f'world_model_region{args.region_id}.pt'
    ppo_path = ckpt_dir / f'ppo_posterior_region{args.region_id}.pt'

    if wm_path.exists():
        import torch
        sd_list = torch.load(wm_path, map_location=objs.world_model.device)
        for m, sd in zip(objs.world_model.models, sd_list):
            m.load_state_dict(sd)
            m.eval()
        print('[OK] loaded world model:', wm_path)
    else:
        print('[WARN] world model not found:', wm_path)

    if ppo_path.exists():
        objs.ppo.load(str(ppo_path))
        print('[OK] loaded PPO:', ppo_path)
    else:
        print('[WARN] PPO not found:', ppo_path)

    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=int(args.steps),
    )
    cyborg = CybORG(scenario_generator=sg, seed=int(args.seed))
    env = BlueFixedActionWrapper(cyborg, pad_spaces=True)
    all_blue_agents = sorted([a for a in env.action_spaces().keys() if 'blue_agent' in a])
    target_agent = f'blue_agent_{int(args.region_id)}'
    if target_agent not in all_blue_agents:
        raise RuntimeError(f'target_agent={target_agent} not in {all_blue_agents}')

    out = Path(args.jsonl_path)
    if out.exists():
        out.unlink()

    episode_returns = []
    plan_rank_hist = []

    for ep in range(int(args.episodes)):
        obs_dict, info_dict = env.reset(seed=int(args.seed) + ep)
        objs.summarizer.reset()
        ep_ret = 0.0
        target_obs = _safe_to_dict(obs_dict.get(target_agent, {}))
        ev, risk = _extract_events_and_risk(target_obs)
        obs = {'t': 0, 'events': ev, 'risk_proxy': risk, 'region_id': int(args.region_id)}

        for t in range(int(args.steps)):
            objs.summarizer.update(obs.get('events', []))
            summary = objs.summarizer.build(risk_proxy=float(obs.get('risk_proxy', 0.0) or 0.0))
            prior_out = objs.llm_prior.generate(summary)
            prior_logits, E, plans, plan_ranks = build_evidence(cfg, objs.world_model, summary, prior_out)
            idx, _, _ = objs.ppo.act(prior_logits=prior_logits, E=E, deterministic=bool(args.deterministic))
            choose_plan = int(idx)
            action_id = int(plans[choose_plan][0])
            chosen_rank = int(plan_ranks[choose_plan])
            plan_rank_hist.append(chosen_rank)
            desired_type = ACTION_TYPE_MAP.get(action_id, 'Monitor')

            actions_idx = {}
            chosen_labels = {}
            for agent in all_blue_agents:
                idx0, lab0, _, _ = _pick_best_action(env, agent, 'Monitor', {})
                actions_idx[agent] = idx0
                chosen_labels[agent] = lab0

            host_scores = _host_score_map(target_obs)
            idx_a, lab_a, host_a, score_a = _pick_best_action(env, target_agent, desired_type, host_scores)
            actions_idx[target_agent] = idx_a
            chosen_labels[target_agent] = lab_a

            obs_dict, rewards_dict, terminated, truncated, info_dict = env.step(actions_idx)
            # rewards/terminated/truncated may be dicts in the CybORG wrapper;
            # do NOT use them directly in boolean context.
            if isinstance(rewards_dict, dict):
                if target_agent in rewards_dict:
                    rr = float(rewards_dict.get(target_agent, 0.0))
                elif '__all__' in rewards_dict:
                    rr = float(rewards_dict.get('__all__', 0.0))
                else:
                    # fallback: mean reward across blue agents if target key missing
                    blue_vals = [float(v) for k, v in rewards_dict.items() if isinstance(k, str) and 'blue_agent' in k]
                    rr = float(np.mean(blue_vals)) if blue_vals else 0.0
            else:
                rr = float(rewards_dict) if rewards_dict is not None else 0.0
            ep_ret += rr

            next_raw = _safe_to_dict(obs_dict.get(target_agent, {}))
            ev, risk = _extract_events_and_risk(next_raw)
            obs = {'t': t+1, 'events': ev, 'risk_proxy': risk, 'region_id': int(args.region_id)}
            target_obs = next_raw

            if ep == 0 and t == 0:
                try:
                    print('[DEBUG] reward_keys=', list(rewards_dict.keys()) if isinstance(rewards_dict, dict) else type(rewards_dict).__name__)
                    print('[DEBUG] terminated_type=', type(terminated).__name__, 'truncated_type=', type(truncated).__name__)
                    print('[DEBUG] target_agent=', target_agent, 'step_reward=', rr)
                except Exception:
                    pass

            rec = {
                'episode': ep,
                't': t,
                'region_id': int(args.region_id),
                'choose_plan': choose_plan,
                'chosen_plan_rank': chosen_rank,
                'action_id': action_id,
                'desired_type': desired_type,
                'executed_label': chosen_labels[target_agent],
                'official_reward': rr,
                'reward': rr,
                'done': bool(terminated or truncated),
            }
            write_jsonl(out, rec)

            # In CybORG wrappers, terminated/truncated can be dicts keyed by agent.
            if isinstance(terminated, dict):
                done_term = bool(terminated.get(target_agent, False) or terminated.get('__all__', False) or any(bool(v) for v in terminated.values()))
            else:
                done_term = bool(terminated)
            if isinstance(truncated, dict):
                done_trunc = bool(truncated.get(target_agent, False) or truncated.get('__all__', False) or any(bool(v) for v in truncated.values()))
            else:
                done_trunc = bool(truncated)
            if done_term or done_trunc:
                break

        episode_returns.append(ep_ret)
        print(f'[EP={ep}] return={ep_ret:.3f}')

    mean_ret = float(np.mean(episode_returns)) if episode_returns else 0.0
    std_ret = float(np.std(episode_returns, ddof=1)) if len(episode_returns) > 1 else 0.0

    N = int(cfg['llm_prior']['k_candidates'])
    rank_counts = np.bincount(plan_rank_hist, minlength=N) if plan_rank_hist else np.zeros(N)
    rank_total = len(plan_rank_hist) or 1
    rank_probs = [f"{(rank_counts[i] / rank_total) * 100:.1f}%" for i in range(N)]

    print('\n[OFFICIAL-REWARD EVAL]')
    print('  region_id:', args.region_id)
    print('  episodes:', len(episode_returns))
    print('  steps_per_episode:', args.steps)
    print('  mean_reward:', mean_ret)
    print('  std_reward:', std_ret)
    print('  plan_rank_dist:', rank_probs)
    print('  jsonl:', str(out))


if __name__ == '__main__':
    main()
