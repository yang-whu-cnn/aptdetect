from __future__ import annotations

import argparse
import json
import os
import re
import sys
import types
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


HOST_RE = re.compile(r'([A-Za-z]+[A-Za-z0-9_\-]*\d+|Op_[A-Za-z0-9_\-]+|User\d+|Enterprise\d+|Server\d+)', re.I)


def _extract_host_from_label(label: str) -> Optional[str]:
    if not label:
        return None
    m = HOST_RE.findall(label)
    if not m:
        return None
    return m[-1]


def _coerce_score(v: Any) -> float:
    """Robustly convert host score values to float.
    _host_score_map sometimes returns nested dicts like {'score': x, ...}.
    """
    if v is None:
        return 0.0
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except Exception:
            return 0.0
    if isinstance(v, dict):
        # Prefer common score-like keys.
        for k in ('score', 'risk', 'risk_score', 'value', 'suspicion', 'host_score'):
            if k in v:
                return _coerce_score(v.get(k))
        # Otherwise search all values and take the max numeric-ish signal.
        vals = [_coerce_score(x) for x in v.values()]
        vals = [x for x in vals if np.isfinite(x)]
        return max(vals) if vals else 0.0
    if isinstance(v, (list, tuple)):
        vals = [_coerce_score(x) for x in v]
        vals = [x for x in vals if np.isfinite(x)]
        return max(vals) if vals else 0.0
    try:
        return float(v)
    except Exception:
        return 0.0


class TacticalMemory:
    def __init__(self):
        self.suspicion = defaultdict(float)
        self.last_seen = {}
        self.last_analysed = {}
        self.last_removed = {}
        self.last_restored = {}
        self.last_action = None
        self.last_host = None

    def decay(self, t: int, decay: float = 0.90):
        for h in list(self.suspicion.keys()):
            age = t - self.last_seen.get(h, t)
            if age > 0:
                self.suspicion[h] *= (decay ** min(age, 3))
            if self.suspicion[h] < 0.05:
                self.suspicion.pop(h, None)
                self.last_seen.pop(h, None)

    def update_from_scores(self, host_scores: Dict[str, Any], t: int):
        self.decay(t)
        for h, raw_s in host_scores.items():
            s = _coerce_score(raw_s)
            self.last_seen[str(h)] = t
            if s > 0:
                self.suspicion[str(h)] = 0.72 * self.suspicion.get(str(h), 0.0) + 0.95 * s

    def mark(self, action_type: str, host: Optional[str], t: int):
        self.last_action = action_type
        self.last_host = host
        if not host:
            return
        if action_type == 'Analyse':
            self.last_analysed[host] = t
        elif action_type == 'Remove':
            self.last_removed[host] = t
            self.suspicion[host] = min(self.suspicion.get(host, 0.0), 0.75)
        elif action_type == 'Restore':
            self.last_restored[host] = t
            self.suspicion[host] = min(self.suspicion.get(host, 0.0), 0.35)

    def recently_analysed(self, host: str, t: int, window: int = 3) -> bool:
        return (t - self.last_analysed.get(host, -10**9)) <= window

    def recently_removed(self, host: str, t: int, window: int = 2) -> bool:
        return (t - self.last_removed.get(host, -10**9)) <= window

    def recently_restored(self, host: str, t: int, window: int = 3) -> bool:
        return (t - self.last_restored.get(host, -10**9)) <= window


def _rank_hosts(host_scores: Dict[str, Any]) -> List[Tuple[str, float]]:
    pairs = [(str(h), _coerce_score(s)) for h, s in host_scores.items()]
    pairs = [x for x in pairs if np.isfinite(x[1])]
    return sorted(pairs, key=lambda x: x[1], reverse=True)


def _choose_target_action(
    t: int,
    desired_type_rl: str,
    host_scores: Dict[str, Any],
    mem: TacticalMemory,
    warmup_steps: int,
    analyse_thresh: float,
    remove_thresh: float,
    restore_gap: int,
) -> Tuple[str, Optional[str], str]:
    ranked = _rank_hosts(host_scores)
    top_host = ranked[0][0] if ranked else None
    top_score = ranked[0][1] if ranked else 0.0

    if t < warmup_steps:
        if top_host is not None and top_score >= analyse_thresh:
            return 'Analyse', top_host, 'warmup_analyse'
        return 'Monitor', None, 'warmup_monitor'

    if mem.last_action == 'Remove' and mem.last_host is not None:
        h = mem.last_host
        if (t - mem.last_removed.get(h, -10**9)) >= restore_gap and not mem.recently_restored(h, t, window=2):
            return 'Restore', h, 'post_remove_restore'

    if top_host is not None:
        susp = float(mem.suspicion.get(top_host, 0.0))
        analysed_recently = mem.recently_analysed(top_host, t, window=3)
        removed_recently = mem.recently_removed(top_host, t, window=2)

        if not removed_recently:
            if desired_type_rl == 'Remove' and (top_score >= analyse_thresh or susp >= remove_thresh):
                if analysed_recently or susp >= (remove_thresh + 0.35):
                    return 'Remove', top_host, 'rl_remove_with_memory'
            if top_score >= remove_thresh and analysed_recently:
                return 'Remove', top_host, 'threshold_remove'
            if susp >= (remove_thresh + 0.55):
                return 'Remove', top_host, 'persistent_remove'

        if top_score >= analyse_thresh:
            return 'Analyse', top_host, 'threshold_analyse'
        if desired_type_rl == 'Analyse' and top_score > 0:
            return 'Analyse', top_host, 'rl_analyse'
        if desired_type_rl == 'Restore' and mem.recently_removed(top_host, t, window=2):
            return 'Restore', top_host, 'rl_restore_after_remove'

    if desired_type_rl in ('Analyse', 'Remove', 'Restore') and top_host is not None and top_score > 0:
        return desired_type_rl, top_host, 'rl_fallback'

    return 'Monitor', None, 'default_monitor'


def _pick_action_with_optional_host(env, agent: str, desired_type: str, host_scores: Dict[str, Any], preferred_host: Optional[str]):
    # Important: project-native _pick_best_action / _match_label_score expects the
    # original rich metadata structure from _host_score_map (often dict-valued).
    # We therefore keep raw host_scores for that call, and only coerce to float for
    # our own ranking/fallback logic.
    raw_host_scores = {str(k): v for k, v in host_scores.items()}
    coerced_host_scores = {str(k): _coerce_score(v) for k, v in host_scores.items()}
    idx, lab, host, score = _pick_best_action(env, agent, desired_type, raw_host_scores)

    if preferred_host and isinstance(lab, str) and preferred_host.lower() in lab.lower():
        return idx, lab, host, score

    try:
        a_space = env.action_space(agent)
        labels = getattr(a_space, 'labels', None)
        if labels is None:
            labels = getattr(a_space, '_labels', None)
        if labels is not None and preferred_host:
            desired_low = desired_type.lower()
            pref_low = preferred_host.lower()
            for i, x in enumerate(labels):
                sx = str(x)
                if desired_low in sx.lower() and pref_low in sx.lower():
                    return int(i), sx, preferred_host, float(coerced_host_scores.get(preferred_host, 0.0))
    except Exception:
        pass

    return idx, lab, host, score


def main():
    ap = argparse.ArgumentParser(description='Hybrid boosted official reward eval on CybORG/CC4')
    ap.add_argument('--cfg', default='configs/cc4.yaml')
    ap.add_argument('--region_id', type=int, required=True)
    ap.add_argument('--episodes', type=int, default=20)
    ap.add_argument('--steps', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--deterministic', action='store_true')
    ap.add_argument('--jsonl_path', default='outputs/official_eval/region_eval_hybrid_boost.jsonl')
    ap.add_argument('--warmup_steps', type=int, default=2)
    ap.add_argument('--analyse_thresh', type=float, default=0.55)
    ap.add_argument('--remove_thresh', type=float, default=1.10)
    ap.add_argument('--restore_gap', type=int, default=1)
    args = ap.parse_args()

    cfg_path = Path(args.cfg)
    if not cfg_path.is_absolute():
        cfg_path = (PROJ / args.cfg).resolve()
    cfg = load_yaml(str(cfg_path))
    set_seed(int(args.seed))

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

    for ep in range(int(args.episodes)):
        obs_dict, info_dict = env.reset(seed=int(args.seed) + ep)
        objs.summarizer.reset()
        mem = TacticalMemory()
        ep_ret = 0.0
        target_obs = _safe_to_dict(obs_dict.get(target_agent, {}))
        ev, risk = _extract_events_and_risk(target_obs)
        obs = {'t': 0, 'events': ev, 'risk_proxy': risk, 'region_id': int(args.region_id)}

        for t in range(int(args.steps)):
            objs.summarizer.update(obs.get('events', []))
            summary = objs.summarizer.build(risk_proxy=float(obs.get('risk_proxy', 0.0) or 0.0))
            prior_out = objs.llm_prior.generate(summary)
            prior_logits, E, plans, _ = build_evidence(cfg, objs.world_model, summary, prior_out)
            idx, _, _ = objs.ppo.act(prior_logits=prior_logits, E=E, deterministic=bool(args.deterministic))
            choose_plan = int(idx)
            action_id = int(plans[choose_plan][0])
            desired_type_rl = ACTION_TYPE_MAP.get(action_id, 'Monitor')

            actions_idx = {}
            chosen_labels = {}
            decision_reason = 'default'
            chosen_host = None
            raw_host_scores = _host_score_map(target_obs)
            host_scores = {str(k): _coerce_score(v) for k, v in raw_host_scores.items()}
            mem.update_from_scores(raw_host_scores, t)

            for agent in all_blue_agents:
                idx0, lab0, _, _ = _pick_best_action(env, agent, 'Monitor', {})
                actions_idx[agent] = idx0
                chosen_labels[agent] = lab0

            desired_type, preferred_host, decision_reason = _choose_target_action(
                t=t,
                desired_type_rl=desired_type_rl,
                host_scores=raw_host_scores,
                mem=mem,
                warmup_steps=int(args.warmup_steps),
                analyse_thresh=float(args.analyse_thresh),
                remove_thresh=float(args.remove_thresh),
                restore_gap=int(args.restore_gap),
            )

            idx_a, lab_a, host_a, score_a = _pick_action_with_optional_host(
                env=env,
                agent=target_agent,
                desired_type=desired_type,
                host_scores=raw_host_scores,
                preferred_host=preferred_host,
            )
            exec_host = preferred_host or host_a or _extract_host_from_label(str(lab_a))
            actions_idx[target_agent] = idx_a
            chosen_labels[target_agent] = lab_a
            mem.mark(desired_type, exec_host, t)
            chosen_host = exec_host

            obs_dict, rewards_dict, terminated, truncated, info_dict = env.step(actions_idx)
            if isinstance(rewards_dict, dict):
                if target_agent in rewards_dict:
                    rr = float(rewards_dict.get(target_agent, 0.0))
                elif '__all__' in rewards_dict:
                    rr = float(rewards_dict.get('__all__', 0.0))
                else:
                    blue_vals = [float(v) for k, v in rewards_dict.items() if isinstance(k, str) and 'blue_agent' in k]
                    rr = float(np.mean(blue_vals)) if blue_vals else 0.0
            else:
                rr = float(rewards_dict) if rewards_dict is not None else 0.0
            ep_ret += rr

            next_raw = _safe_to_dict(obs_dict.get(target_agent, {}))
            ev, risk = _extract_events_and_risk(next_raw)
            obs = {'t': t + 1, 'events': ev, 'risk_proxy': risk, 'region_id': int(args.region_id)}
            target_obs = next_raw

            if ep == 0 and t < 3:
                try:
                    print(
                        f"[DEBUG t={t}] rl={desired_type_rl} final={desired_type} host={chosen_host} "
                        f"reason={decision_reason} top_scores={dict(_rank_hosts(host_scores)[:3])} reward={rr}"
                    )
                except Exception:
                    pass

            if isinstance(terminated, dict):
                done_term = bool(terminated.get(target_agent, False) or terminated.get('__all__', False) or any(bool(v) for v in terminated.values()))
            else:
                done_term = bool(terminated)
            if isinstance(truncated, dict):
                done_trunc = bool(truncated.get(target_agent, False) or truncated.get('__all__', False) or any(bool(v) for v in truncated.values()))
            else:
                done_trunc = bool(truncated)
            done_flag = bool(done_term or done_trunc)

            rec = {
                'episode': ep,
                't': t,
                'region_id': int(args.region_id),
                'choose_plan': choose_plan,
                'action_id': action_id,
                'desired_type_rl': desired_type_rl,
                'desired_type_final': desired_type,
                'decision_reason': decision_reason,
                'chosen_host': chosen_host,
                'executed_label': chosen_labels[target_agent],
                'host_scores_top3': _rank_hosts(host_scores)[:3],
                'official_reward': rr,
                'reward': rr,
                'done': done_flag,
            }
            write_jsonl(out, rec)

            if done_flag:
                break

        episode_returns.append(ep_ret)
        print(f'[EP={ep}] return={ep_ret:.3f}')

    mean_ret = float(np.mean(episode_returns)) if episode_returns else 0.0
    std_ret = float(np.std(episode_returns, ddof=1)) if len(episode_returns) > 1 else 0.0
    print('\n[OFFICIAL-REWARD EVAL - HYBRID BOOST V2]')
    print('  region_id:', args.region_id)
    print('  episodes:', len(episode_returns))
    print('  steps_per_episode:', args.steps)
    print('  mean_reward:', mean_ret)
    print('  std_reward:', std_ret)
    print('  jsonl:', str(out))


if __name__ == '__main__':
    main()
