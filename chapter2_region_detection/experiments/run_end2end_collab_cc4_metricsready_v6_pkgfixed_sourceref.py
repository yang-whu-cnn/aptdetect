
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import types
if "src" not in sys.modules:
    pkg = types.ModuleType("src")
    pkg.__path__ = [str(SRC_DIR)]
    sys.modules["src"] = pkg
else:
    pkg = sys.modules["src"]
    if not hasattr(pkg, "__path__"):
        pkg.__path__ = [str(SRC_DIR)]
    elif str(SRC_DIR) not in list(pkg.__path__):
        pkg.__path__ = [str(SRC_DIR)] + list(pkg.__path__)

CYBORG_ROOT = os.environ.get("CYBORG_ROOT", r"D:\python1\cage-challenge-4")
if CYBORG_ROOT and CYBORG_ROOT not in sys.path:
    sys.path.insert(0, CYBORG_ROOT)

from src.utils import load_yaml, set_seed  # noqa: E402
from src.env_region import build_region_objects, build_evidence  # noqa: E402
from src.collab_mock import End2EndCoordinator, RegionReport  # noqa: E402

SCRIPT_VERSION = "cyborg_v6_metrics_source_ref_2026-03-02"
ACTION_NAMES = {0: "Monitor", 1: "Analyse", 3: "Remove", 4: "Restore"}
ACTION_BUCKETS = [0, 1, 3, 4]


def _safe_to_dict(x: Any) -> Dict[str, Any]:
    if x is None:
        return {}
    if isinstance(x, dict):
        return x
    for attr in ("observation", "data", "raw", "dict"):
        if hasattr(x, attr):
            v = getattr(x, attr)
            if isinstance(v, dict):
                return v
    try:
        return dict(x)
    except Exception:
        return {"_repr": repr(x)}


def _extract_events_and_risk(obs_raw: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], float]:
    events: List[Dict[str, Any]] = []

    for k in ("events", "Events", "event", "Event"):
        if k in obs_raw and isinstance(obs_raw[k], (list, tuple)):
            for e in obs_raw[k]:
                if isinstance(e, dict):
                    et = str(e.get("type", e.get("event_type", "unknown")))
                    sev = float(e.get("severity", e.get("sev", 0.5)) or 0.5)
                    events.append({"type": et, "severity": sev})
            break

    if not events:
        proc_n = sess_n = conn_n = file_n = 0
        max_sev = 0.0
        for hk, hv in obs_raw.items():
            if hk in ("success", "action", "agent", "message", "Messages"):
                continue
            if not isinstance(hv, dict):
                continue
            procs = hv.get("Processes") or hv.get("processes") or []
            sess = hv.get("Sessions") or hv.get("sessions") or []
            conns = hv.get("Connections") or hv.get("connections") or hv.get("NetworkConnections") or []
            files = hv.get("Files") or hv.get("files") or []
            proc_n += len(procs) if isinstance(procs, list) else 0
            sess_n += len(sess) if isinstance(sess, list) else 0
            conn_n += len(conns) if isinstance(conns, list) else 0
            file_n += len(files) if isinstance(files, list) else 0
            alerts = hv.get("Alerts") or hv.get("Host Alerts") or hv.get("alerts") or []
            if isinstance(alerts, list):
                for a in alerts:
                    if isinstance(a, dict):
                        max_sev = max(max_sev, float(a.get("severity", 0.0) or 0.0))
        if conn_n > 0:
            events.append({"type": "outbound_conn", "severity": float(min(1.0, conn_n / 10.0))})
        if proc_n > 0:
            events.append({"type": "proc_spawn", "severity": float(min(1.0, proc_n / 10.0))})
        if sess_n > 0:
            events.append({"type": "suspicious_login", "severity": float(min(1.0, sess_n / 10.0))})
        if file_n > 0:
            events.append({"type": "file_activity", "severity": float(min(1.0, file_n / 10.0))})
        if max_sev > 0:
            events.append({"type": "alert", "severity": float(min(1.0, max_sev))})

    if not events:
        return [], 0.0
    sev_sum = float(sum(float(e.get("severity", 0.0) or 0.0) for e in events))
    risk_proxy = float(np.clip(0.15 * len(events) + 0.35 * (sev_sum / max(1.0, len(events))), 0.0, 1.0))
    return events, risk_proxy


def _count_evidence(obs: Dict[str, Any]) -> Dict[str, float]:
    events = obs.get("events", []) or []
    alert_types = {"port_scan", "proc_spawn", "outbound_conn", "alert"}
    alert_count = 0.0
    ioc_count = 0.0
    for e in events:
        et = str(e.get("type", ""))
        sev = float(e.get("severity", 0.0) or 0.0)
        if et in alert_types:
            alert_count += 1.0
            if et in ("outbound_conn", "alert") and sev >= 0.7:
                ioc_count += 1.0
    risk = float(obs.get("risk_proxy", 0.0) or 0.0)
    return {"alert_count": alert_count, "ioc_count": ioc_count, "risk": risk}


def _uncertainty(obs: Dict[str, Any]) -> float:
    return float(np.clip(float(obs.get("risk_proxy", 0.0) or 0.0), 0.0, 1.0))


def _mean(x: List[float]) -> float:
    return float(np.mean(x)) if x else 0.0


def _std(x: List[float]) -> float:
    return float(np.std(x, ddof=1)) if len(x) > 1 else 0.0


def _valid_indices(env, agent: str) -> List[int]:
    labels = env.action_labels(agent)
    mask = env.action_mask(agent) if hasattr(env, "action_mask") else [True] * len(labels)
    return [i for i in range(len(labels)) if bool(mask[i])]


def _extract_host_tokens(label: str) -> List[str]:
    s = str(label)
    parts = re.split(r"[^A-Za-z0-9_.-]+", s)
    toks: List[str] = []
    for p in parts:
        p = p.strip().lower()
        if not p or p in {"analyse", "remove", "restore", "monitor", "sleep"}:
            continue
        toks.append(p)
        if "_" in p:
            toks.extend([x for x in p.split("_") if x])
        if "-" in p:
            toks.extend([x for x in p.split("-") if x])
    seen = set()
    out = []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _host_score_map(obs_raw: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for hk, hv in obs_raw.items():
        if not isinstance(hv, dict):
            continue
        host_name = str(hk)
        hn = host_name.lower()
        aliases = {hn}
        aliases.update([x for x in re.split(r"[^A-Za-z0-9_.-]+", hn) if x])
        aliases.update([x for x in hn.replace("-", "_").split("_") if x])
        score = 0.0
        ip = None

        if isinstance(hv.get("Interface"), list) and hv["Interface"]:
            it0 = hv["Interface"][0]
            if isinstance(it0, dict):
                ip = it0.get("ip_address")
                if ip is not None:
                    ip = str(ip)
                    aliases.add(ip.lower())

        alerts = hv.get("Alerts") or hv.get("Host Alerts") or hv.get("alerts") or []
        if isinstance(alerts, list):
            for a in alerts:
                if isinstance(a, dict):
                    score += 5.0 + 7.0 * float(a.get("severity", 0.5) or 0.5)
                else:
                    score += 5.0
        conns = hv.get("Connections") or hv.get("connections") or hv.get("NetworkConnections") or []
        if isinstance(conns, list):
            score += min(6.0, 1.0 * len(conns))
        procs = hv.get("Processes") or hv.get("processes") or []
        if isinstance(procs, list):
            score += min(5.0, 0.5 * len(procs))
        sess = hv.get("Sessions") or hv.get("sessions") or []
        if isinstance(sess, list):
            score += min(4.0, 0.7 * len(sess))
        files = hv.get("Files") or hv.get("files") or []
        if isinstance(files, list):
            score += min(3.0, 0.3 * len(files))
        if any(k in hn for k in ["user", "server", "workstation", "host"]):
            score += 0.5

        out[host_name] = {"score": float(score), "aliases": aliases, "ip": ip}
    return out


def _best_host_from_scores(host_scores: Dict[str, Dict[str, Any]]) -> Tuple[Optional[str], float]:
    best_name, best_score = None, -1.0
    for name, meta in host_scores.items():
        sc = float(meta.get("score", 0.0) or 0.0)
        if sc > best_score:
            best_name, best_score = name, sc
    return best_name, float(max(best_score, 0.0))


def _match_label_score(label: str, host_scores: Dict[str, Dict[str, Any]]) -> Tuple[float, Optional[str]]:
    toks = _extract_host_tokens(label)
    if not toks:
        return 0.0, None
    best_score = 0.0
    best_host = None
    for host_name, meta in host_scores.items():
        aliases = meta.get("aliases", set())
        if any(tok in aliases for tok in toks):
            sc = float(meta.get("score", 0.0) or 0.0)
            if sc > best_score:
                best_score = sc
                best_host = host_name
    return best_score, best_host


def _pick_best_action(env, agent: str, action_type: str, host_scores: Dict[str, Dict[str, Any]]) -> Tuple[int, str, Optional[str], float]:
    labels = env.action_labels(agent)
    valid = _valid_indices(env, agent)
    if action_type == "Monitor":
        for i in valid:
            lab = str(labels[i])
            if "Monitor" in lab:
                return i, lab, None, 0.0
        i = valid[0] if valid else 0
        return i, str(labels[i]), None, 0.0

    best_i = best_lab = best_host = None
    best_score = -1.0
    for i in valid:
        lab = str(labels[i])
        if action_type not in lab:
            continue
        sc, host = _match_label_score(lab, host_scores)
        if sc > best_score:
            best_i, best_lab, best_host, best_score = i, lab, host, sc
    if best_i is not None:
        return int(best_i), str(best_lab), best_host, float(max(best_score, 0.0))
    for i in valid:
        lab = str(labels[i])
        if action_type in lab:
            return int(i), lab, None, 0.0
    i = valid[0] if valid else 0
    return int(i), str(labels[i]), None, 0.0


def _label_to_bucket(label: str) -> int:
    if "Restore" in label:
        return 4
    if "Remove" in label:
        return 3
    if "Analyse" in label:
        return 1
    return 0


@dataclass
class BanditChoice:
    ctx_key: Tuple[int, int, int, int, int]
    host_key: str
    action: int
    prev_risk: float
    prev_host_score: float
    same_host_streak: int
    monitor_score: float = 0.0
    chosen_score: float = 0.0


class ConservativeRewardBandit:
    def __init__(self, seed: int = 42, eps_train_start: float = 0.20, eps_train_end: float = 0.03,
                 eps_eval: float = 0.0, ucb_c: float = 0.55, margin_eval: float = 0.60,
                 risk_bonus: float = 10.0, score_bonus: float = 2.0):
        self.rng = random.Random(seed)
        self.eps_train_start = float(eps_train_start)
        self.eps_train_end = float(eps_train_end)
        self.eps_eval = float(eps_eval)
        self.ucb_c = float(ucb_c)
        self.margin_eval = float(margin_eval)
        self.risk_bonus = float(risk_bonus)
        self.score_bonus = float(score_bonus)
        self.ctx_stats = defaultdict(lambda: {a: {"n": 0, "q": 0.0} for a in ACTION_BUCKETS})
        self.host_stats = defaultdict(lambda: {a: {"n": 0, "q": 0.0} for a in ACTION_BUCKETS})

    @staticmethod
    def _risk_bin(x: float) -> int:
        if x < 0.05: return 0
        if x < 0.15: return 1
        if x < 0.30: return 2
        if x < 0.50: return 3
        return 4

    @staticmethod
    def _score_bin(x: float) -> int:
        if x < 0.2: return 0
        if x < 1.0: return 1
        if x < 2.5: return 2
        if x < 5.0: return 3
        return 4

    @staticmethod
    def _small_bin(x: float) -> int:
        if x <= 0: return 0
        if x < 1.0: return 1
        if x < 3.0: return 2
        return 3

    def make_ctx_key(self, risk: float, alert_count: float, ioc_count: float,
                     host_score: float, same_host_streak: int) -> Tuple[int, int, int, int, int]:
        return (
            self._risk_bin(risk),
            self._small_bin(alert_count),
            self._small_bin(ioc_count),
            self._score_bin(host_score),
            min(int(same_host_streak), 3),
        )

    def _ucb(self, stat: Dict[str, float], total_n: int) -> float:
        n = int(stat["n"])
        if n <= 0:
            return 1.8
        return self.ucb_c * math.sqrt(math.log(max(total_n, 2)) / n)

    def _prior(self, action: int, risk: float, alert_count: float, ioc_count: float,
               host_score: float, same_host_streak: int, base_action: int,
               has_verify: bool, has_respond: bool, last_action: int,
               remove_cooldown: int, restore_ready: bool, risk_ema: float) -> float:
        s = 0.0
        if action == 0:
            s += 1.2
            s += 0.8 if risk < 0.10 else -0.2 * max(risk - 0.10, 0)
            if host_score < 0.5:
                s += 0.6
            if ioc_count <= 0 and alert_count <= 0.5:
                s += 0.5
        elif action == 1:
            s += 0.35 * alert_count + 0.60 * ioc_count + 0.35 * min(host_score, 3.0)
            if has_verify or base_action == 1:
                s += 0.45
            if same_host_streak >= 1:
                s += 0.25
            if risk_ema >= 0.20:
                s += 0.30
        elif action == 3:
            s += 0.8 * ioc_count + 0.5 * min(host_score, 4.0) + 0.4 * same_host_streak
            if has_respond or base_action == 3:
                s += 0.55
            if last_action == 1:
                s += 0.25
            if remove_cooldown > 0:
                s -= 1.0
            if host_score < 1.0 and ioc_count <= 0:
                s -= 0.8
            if risk_ema >= 0.25:
                s += 0.25
        elif action == 4:
            s -= 0.3
            if restore_ready:
                s += 0.7
            else:
                s -= 1.5
            if last_action == 3:
                s += 0.45
            if risk < 0.15 and host_score < 0.8:
                s += 0.25
            if has_respond:
                s += 0.15
        return float(s)

    def action_scores(self, *, agent: str, risk: float, alert_count: float, ioc_count: float,
                      host_key: str, host_score: float, same_host_streak: int,
                      available_actions: List[int], base_action: int, has_verify: bool,
                      has_respond: bool, last_action: int, remove_cooldown: int,
                      restore_ready: bool, risk_ema: float) -> Tuple[Tuple[int,int,int,int,int], str, Dict[int, float]]:
        ctx_key = self.make_ctx_key(risk, alert_count, ioc_count, host_score, same_host_streak)
        host_slot = f"{agent}|{host_key or '__none__'}"
        total_n = sum(int(self.ctx_stats[ctx_key][a]["n"]) for a in ACTION_BUCKETS) + 1
        scores = {}
        for a in available_actions:
            ctx_stat = self.ctx_stats[ctx_key][a]
            host_stat = self.host_stats[host_slot][a]
            prior = self._prior(a, risk, alert_count, ioc_count, host_score, same_host_streak,
                                base_action, has_verify, has_respond, last_action,
                                remove_cooldown, restore_ready, risk_ema)
            scores[a] = (
                0.60 * float(ctx_stat["q"]) +
                0.40 * float(host_stat["q"]) +
                self._ucb(ctx_stat, total_n) +
                0.5 * self._ucb(host_stat, total_n) +
                prior
            )
        return ctx_key, host_slot, scores

    def select_action(self, *, agent: str, risk: float, alert_count: float, ioc_count: float,
                      host_key: str, host_score: float, same_host_streak: int,
                      available_actions: List[int], base_action: int, has_verify: bool,
                      has_respond: bool, last_action: int, remove_cooldown: int,
                      restore_ready: bool, risk_ema: float, train_mode: bool,
                      eps_override: Optional[float] = None) -> Tuple[int, BanditChoice]:
        ctx_key, host_slot, scores = self.action_scores(
            agent=agent, risk=risk, alert_count=alert_count, ioc_count=ioc_count,
            host_key=host_key, host_score=host_score, same_host_streak=same_host_streak,
            available_actions=available_actions, base_action=base_action, has_verify=has_verify,
            has_respond=has_respond, last_action=last_action, remove_cooldown=remove_cooldown,
            restore_ready=restore_ready, risk_ema=risk_ema
        )

        monitor_score = float(scores.get(0, -1e9))
        eps = self.eps_eval if not train_mode else (self.eps_train_start if eps_override is None else eps_override)

        if available_actions and self.rng.random() < eps:
            action = self.rng.choice(available_actions)
        else:
            best_action, best_score = max(scores.items(), key=lambda kv: kv[1])
            if not train_mode and 0 in scores and best_action != 0 and (best_score - monitor_score) < self.margin_eval:
                action = 0
                best_score = monitor_score
            else:
                action = best_action

        return action, BanditChoice(
            ctx_key=ctx_key,
            host_key=host_slot,
            action=action,
            prev_risk=float(risk),
            prev_host_score=float(host_score),
            same_host_streak=int(same_host_streak),
            monitor_score=monitor_score,
            chosen_score=float(scores.get(action, 0.0)),
        )

    def update(self, choice: BanditChoice, *, team_reward_mean: float, next_risk: float,
               next_host_score: float, executed_bucket: Optional[int] = None) -> None:
        action = int(executed_bucket if executed_bucket is not None else choice.action)
        risk_drop = float(choice.prev_risk - next_risk)
        host_drop = float(choice.prev_host_score - next_host_score)

        action_cost = {0: 0.0, 1: 0.25, 3: 0.55, 4: 0.45}[action]
        shaped = float(team_reward_mean) + self.risk_bonus * risk_drop + self.score_bonus * host_drop - action_cost

        # conservative shaping: heavy actions need evidence of benefit
        if action == 3 and risk_drop <= 0 and host_drop <= 0:
            shaped -= 0.9
        if action == 4 and risk_drop <= 0:
            shaped -= 1.2
        if action == 0 and risk_drop < -0.05:
            shaped -= 0.7

        ctx_stat = self.ctx_stats[choice.ctx_key][action]
        host_stat = self.host_stats[choice.host_key][action]
        ctx_stat["n"] += 1
        host_stat["n"] += 1
        ctx_stat["q"] += (shaped - ctx_stat["q"]) / ctx_stat["n"]
        host_stat["q"] += (shaped - host_stat["q"]) / host_stat["n"]


def _iter_kv(obj: Any):
    if obj is None:
        return []
    if isinstance(obj, dict):
        return list(obj.items())
    if hasattr(obj, "__dict__"):
        return [(k, v) for k, v in vars(obj).items() if not k.startswith("_")]
    return []


def _get_any(obj: Any, names: List[str], default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        for n in names:
            if n in obj:
                return obj[n]
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return default


def _iter_collection(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, dict):
        return list(x.values())
    if isinstance(x, (list, tuple, set)):
        return list(x)
    return [x]


def _session_owner_strings(sess: Any) -> List[str]:
    vals = []
    for key in ("agent", "agent_name", "username", "user", "name", "hostname", "session_type", "ident"):
        v = _get_any(sess, [key], None)
        if v is not None:
            vals.append(str(v).lower())
    return vals


def _boolish(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "yes", "up", "active", "running"}:
            return True
        if s in {"false", "no", "down", "inactive", "stopped"}:
            return False
    return None



def _extract_true_state_snapshot(cyborg: Any, info_dict: Any = None) -> Dict[str, Any]:
    """Source-aligned host truth extraction.

    Reference logic from the Hierarchical-MARL repo:
    - compromised host: has active red session
    - escalated host: has active red session with privileged access
    - impacted host: host impact_count > 0
    """
    state_obj = None
    if isinstance(info_dict, dict):
        for k in ("true_state", "state", "TrueState"):
            if k in info_dict:
                state_obj = info_dict[k]
                break
    if state_obj is None and hasattr(cyborg, "environment_controller"):
        ec = getattr(cyborg, "environment_controller")
        state_obj = _get_any(ec, ["state", "environment", "env_state"], None)
    if state_obj is None:
        state_obj = _get_any(cyborg, ["state"], None)

    hosts_map = _get_any(state_obj, ["hosts", "Hosts"], {})
    state_sessions = _get_any(state_obj, ["sessions", "Sessions"], {})

    compromised_hosts: set[str] = set()
    escalated_hosts: set[str] = set()
    impacted_hosts: set[str] = set()
    host_truth: Dict[str, Dict[str, Any]] = {}

    host_items = list(hosts_map.items()) if isinstance(hosts_map, dict) else []
    for host_name, host_obj in host_items:
        hn = str(host_name)
        if "router" in hn.lower():
            continue

        host_sess_map = _get_any(host_obj, ["sessions", "Sessions"], {})
        has_session = False
        has_privileged_session = False

        if isinstance(host_sess_map, dict):
            for i in range(6):
                agent_name = f"red_agent_{i}"
                owner_entry = state_sessions.get(agent_name) if isinstance(state_sessions, dict) else None
                sid_list = host_sess_map.get(agent_name, []) if isinstance(host_sess_map, dict) else []
                if not isinstance(sid_list, (list, tuple, set)):
                    sid_list = [sid_list] if sid_list is not None else []
                if not sid_list:
                    continue
                for sid in sid_list:
                    sess_obj = None
                    if isinstance(owner_entry, dict):
                        sess_obj = owner_entry.get(sid)
                        if sess_obj is None:
                            sess_obj = owner_entry.get(str(sid))
                    if sess_obj is None:
                        continue
                    active = _boolish(_get_any(sess_obj, ["active"], None))
                    if active is False:
                        continue
                    has_session = True
                    try:
                        if hasattr(sess_obj, "has_privileged_access") and callable(sess_obj.has_privileged_access):
                            if bool(sess_obj.has_privileged_access()):
                                has_privileged_session = True
                                break
                    except Exception:
                        pass
                    # fallback if method unavailable
                    blob = repr(sess_obj).lower()
                    if any(tok in blob for tok in ["root", "system", "administrator", "admin", "privileged"]):
                        has_privileged_session = True
                        break
                if has_privileged_session:
                    break

        impact_raw = _get_any(host_obj, ["impact_count"], 0)
        try:
            impact_count = float(impact_raw)
        except Exception:
            impact_count = 0.0

        if has_session:
            compromised_hosts.add(hn)
        if has_privileged_session:
            escalated_hosts.add(hn)
        if impact_count > 0:
            impacted_hosts.add(hn)

        host_truth[hn] = {
            "compromised": bool(has_session),
            "escalated": bool(has_privileged_session),
            "impacted": bool(impact_count > 0),
            "impact_count": impact_count,
        }

    total_hosts = max(len(host_truth), 1)
    return {
        "total_hosts_true": int(total_hosts),
        "compromised_hosts": sorted(compromised_hosts),
        "escalated_hosts": sorted(escalated_hosts),
        "impacted_hosts": sorted(impacted_hosts),
        # backward-compatible aliases
        "infected_hosts": sorted(compromised_hosts),
        "root_hosts": sorted(escalated_hosts),
        "clean_host_ratio_true": float((total_hosts - len(compromised_hosts)) / total_hosts),
        "non_escalated_ratio_true": float((total_hosts - len(escalated_hosts)) / total_hosts),
        "host_truth": host_truth,
    }


def _extract_red_last_actions(cyborg: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    agent_names = []
    for obj in (_get_any(cyborg, ["agents"], []), _get_any(_get_any(cyborg, ["environment_controller"], None), ["agents"], [])):
        if isinstance(obj, (list, tuple, set)):
            for a in obj:
                s = str(a)
                if s.startswith("red_agent_"):
                    agent_names.append(s)
    agent_names = sorted(set(agent_names))
    for agent_name in agent_names:
        action_obj = None
        try:
            if hasattr(cyborg, "get_last_action"):
                got = cyborg.get_last_action(agent_name)
                if isinstance(got, (list, tuple)) and got:
                    action_obj = got[0]
                else:
                    action_obj = got
            elif hasattr(cyborg, "environment_controller") and hasattr(cyborg.environment_controller, "get_last_action"):
                got = cyborg.environment_controller.get_last_action(agent_name)
                if isinstance(got, (list, tuple)) and got:
                    action_obj = got[0]
                else:
                    action_obj = got
        except Exception:
            action_obj = None
        if action_obj is None:
            continue
        name = str(_get_any(action_obj, ["name"], type(action_obj).__name__))
        hostname = _get_any(action_obj, ["hostname", "host"], None)
        success = _boolish(_get_any(action_obj, ["success"], None))
        out.append({
            "agent": agent_name,
            "name": name,
            "hostname": None if hostname is None else str(hostname),
            "success": success,
            "repr": repr(action_obj)[:300],
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/cc4_local_online.yaml")
    ap.add_argument("--regions", default="0,1,2,3")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--episodes", type=int, default=20, help="evaluation episodes after training")
    ap.add_argument("--train_episodes", type=int, default=30, help="official-reward training episodes before evaluation")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--force_dummy_prior", action="store_true")
    ap.add_argument("--backend", choices=["local", "cyborg"], default="cyborg")
    ap.add_argument("--budget_per_step", type=int, default=4)
    ap.add_argument("--risk_trigger", type=float, default=0.045)
    ap.add_argument("--verify_u", type=float, default=0.045)
    ap.add_argument("--warmup_steps", type=int, default=2)
    ap.add_argument("--print_every", type=int, default=1)
    ap.add_argument("--out_jsonl", default="")
    ap.add_argument("--eps_eval", type=float, default=0.0)
    ap.add_argument("--eps_train_start", type=float, default=0.18)
    ap.add_argument("--eps_train_end", type=float, default=0.02)
    ap.add_argument("--remove_score_min", type=float, default=1.2)
    ap.add_argument("--restore_score_min", type=float, default=2.0)
    ap.add_argument("--intervene_margin", type=float, default=0.75)
    ap.add_argument("--remove_cooldown_steps", type=int, default=3)
    ap.add_argument("--restore_after_remove_steps", type=int, default=2)
    args = ap.parse_args()

    cfg_path = (ROOT / args.cfg).resolve()
    cfg = load_yaml(str(cfg_path))
    if args.force_dummy_prior:
        cfg.setdefault("llm_prior", {})
        cfg["llm_prior"]["backend"] = "dummy"

    set_seed(int(args.seed))
    print(f"[RUN] {SCRIPT_VERSION} | backend={args.backend} | train={args.train_episodes} | eval={args.episodes} | steps={args.steps}")

    region_ids = [f"region{int(s.strip().replace('region',''))}" for s in str(args.regions).split(',') if s.strip()]

    objs_by_region = {}
    for rid in region_ids:
        cfg_build = dict(cfg)
        cfg_build["mode"] = "online"
        cfg_build.setdefault("online", {})
        cfg_build["online"]["base_url"] = "local"
        cfg_build["online"]["timeout_sec"] = float(cfg_build["online"].get("timeout_sec", 30))
        objs = build_region_objects(cfg_build, region_id=int(rid.replace("region", "")), device=args.device)
        objs_by_region[rid] = objs

    coord = End2EndCoordinator(
        phases=["recon", "initial_access", "persistence", "lateral_movement", "impact"],
        budget_per_step=int(args.budget_per_step),
        risk_trigger=float(args.risk_trigger),
        verify_u=float(args.verify_u),
    )

    bandit = ConservativeRewardBandit(
        seed=int(args.seed),
        eps_train_start=float(args.eps_train_start),
        eps_train_end=float(args.eps_train_end),
        eps_eval=float(args.eps_eval),
        margin_eval=float(args.intervene_margin),
    )

    out_f = None
    if args.out_jsonl:
        out_path = Path(args.out_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_f = open(str(out_path), "w", encoding="utf-8", newline="\n")

    if args.backend == "cyborg":
        if CYBORG_ROOT and CYBORG_ROOT not in sys.path:
            sys.path.insert(0, CYBORG_ROOT)
        from CybORG import CybORG  # type: ignore
        from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator  # type: ignore
        from CybORG.Agents import SleepAgent, FiniteStateRedAgent, EnterpriseGreenAgent  # type: ignore
        from CybORG.Agents.Wrappers.BlueFixedActionWrapper import BlueFixedActionWrapper  # type: ignore
        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=EnterpriseGreenAgent,
            red_agent_class=FiniteStateRedAgent,
            steps=int(args.steps),
        )
        cyborg = CybORG(scenario_generator=sg, seed=int(args.seed))
        env = BlueFixedActionWrapper(cyborg, pad_spaces=True)
        all_blue_agents = sorted([a for a in env.action_spaces().keys() if "blue_agent" in a])
        region_to_agent = {rid: f"blue_agent_{int(rid.replace('region',''))}" for rid in region_ids}
    else:
        raise RuntimeError("This v6 script is intended for backend=cyborg.")

    phase_plan = [("train", int(args.train_episodes)), ("eval", int(args.episodes))]
    eval_returns_sum: List[float] = []
    eval_returns_mean: List[float] = []
    eval_region_returns = {rid: [] for rid in region_ids}

    for phase, n_eps in phase_plan:
        train_mode = phase == "train"
        for ep in range(n_eps):
            obs_dict, info_dict = env.reset(seed=int(args.seed) + (0 if train_mode else 10000) + ep)
            obs_by_region: Dict[str, Dict[str, Any]] = {}
            region_state = {
                rid: {
                    "last_bucket": 0,
                    "last_host": None,
                    "same_host_streak": 0,
                    "risk_ema": 0.0,
                    "remove_cooldown": 0,
                    "recent_removed_host": None,
                    "recent_removed_ttl": 0,
                }
                for rid in region_ids
            }
            for rid in region_ids:
                agent = region_to_agent[rid]
                raw = _safe_to_dict(obs_dict.get(agent, {}))
                ev, risk = _extract_events_and_risk(raw)
                obs_by_region[rid] = {"t": 0, "events": ev, "risk_proxy": risk, "region_id": int(rid.replace("region", ""))}
                objs_by_region[rid].summarizer.reset()
                objs_by_region[rid].summarizer.update(ev)
                region_state[rid]["risk_ema"] = risk

            ep_ret_sum = 0.0
            ep_ret_mean = 0.0
            ep_ret_region = {rid: 0.0 for rid in region_ids}
            ep_action_counts = {rid: {0: 0, 1: 0, 3: 0, 4: 0} for rid in region_ids}

            for t in range(int(args.steps)):
                reports: List[RegionReport] = []
                base_actions: Dict[str, int] = {}
                for rid in region_ids:
                    objs = objs_by_region[rid]
                    obs = obs_by_region[rid]
                    objs.summarizer.update(obs.get("events", []))
                    summary = objs.summarizer.build(risk_proxy=float(obs.get("risk_proxy", 0.0)))
                    prior_out = objs.llm_prior.generate(summary)
                    prior_logits, E, plans, _ = build_evidence(cfg, objs.world_model, summary, prior_out)
                    idx, _, _ = objs.ppo.act(prior_logits=prior_logits, E=E, deterministic=bool(args.deterministic))
                    choose_plan = int(idx)
                    action_id = int(plans[choose_plan][0])
                    base_actions[rid] = action_id
                    reports.append(RegionReport(
                        region_id=rid, t=t,
                        posterior={"choose_plan": choose_plan, "action_id": action_id},
                        evidence=_count_evidence(obs), foresight={}, uncertainty=_uncertainty(obs)
                    ))

                tasks = coord.step(reports)
                task_types_by_region = {rid: [] for rid in region_ids}
                for tt in tasks:
                    if tt.src_region in task_types_by_region:
                        task_types_by_region[tt.src_region].append(tt.task_type)

                actions_idx = {}
                chosen_labels = {}
                chosen_meta: Dict[str, Dict[str, Any]] = {}

                for agent in all_blue_agents:
                    idx0, lab0, _, _ = _pick_best_action(env, agent, "Monitor", {})
                    actions_idx[agent] = idx0
                    chosen_labels[agent] = lab0

                # epsilon decay only in train
                if train_mode and n_eps > 1:
                    frac = ep / max(n_eps - 1, 1)
                    eps_now = float(args.eps_train_start) + (float(args.eps_train_end) - float(args.eps_train_start)) * frac
                else:
                    eps_now = float(args.eps_eval)

                for rid in region_ids:
                    agent = region_to_agent[rid]
                    raw_obs = _safe_to_dict(obs_dict.get(agent, {}))
                    host_scores = _host_score_map(raw_obs)
                    best_host, best_score = _best_host_from_scores(host_scores)
                    last_host = region_state[rid]["last_host"]
                    if best_host is not None and best_host == last_host and best_score >= 0.2:
                        region_state[rid]["same_host_streak"] += 1
                    elif best_score >= 0.2:
                        region_state[rid]["same_host_streak"] = 1
                    else:
                        region_state[rid]["same_host_streak"] = 0

                    risk = float(obs_by_region[rid].get("risk_proxy", 0.0) or 0.0)
                    region_state[rid]["risk_ema"] = 0.65 * float(region_state[rid]["risk_ema"]) + 0.35 * risk
                    risk_ema = float(region_state[rid]["risk_ema"])
                    evc = _count_evidence(obs_by_region[rid])
                    has_verify = "verify" in task_types_by_region[rid]
                    has_respond = "respond" in task_types_by_region[rid]

                    remove_cooldown = int(region_state[rid]["remove_cooldown"])
                    restore_ready = bool(region_state[rid]["recent_removed_ttl"] > 0 and
                                         best_host is not None and
                                         region_state[rid]["recent_removed_host"] == best_host)

                    available = [0]
                    if best_score >= 0.15 or evc["alert_count"] > 0 or has_verify:
                        available.append(1)
                    if (best_score >= float(args.remove_score_min) or evc["ioc_count"] > 0 or has_respond) and remove_cooldown <= 0:
                        available.append(3)
                    if restore_ready and best_score >= float(args.restore_score_min):
                        available.append(4)

                    # warmup: only light analyse on suspicious host
                    if t < int(args.warmup_steps) and best_score > 0.2:
                        selected_bucket = 1
                        choice = BanditChoice(
                            ctx_key=bandit.make_ctx_key(risk, evc["alert_count"], evc["ioc_count"], best_score, region_state[rid]["same_host_streak"]),
                            host_key=f"{agent}|{best_host or '__none__'}",
                            action=1,
                            prev_risk=risk,
                            prev_host_score=best_score,
                            same_host_streak=region_state[rid]["same_host_streak"],
                            monitor_score=0.0,
                            chosen_score=0.0,
                        )
                    else:
                        selected_bucket, choice = bandit.select_action(
                            agent=agent,
                            risk=risk,
                            alert_count=evc["alert_count"],
                            ioc_count=evc["ioc_count"],
                            host_key=(best_host or "__none__"),
                            host_score=best_score,
                            same_host_streak=region_state[rid]["same_host_streak"],
                            available_actions=available,
                            base_action=base_actions[rid],
                            has_verify=has_verify,
                            has_respond=has_respond,
                            last_action=int(region_state[rid]["last_bucket"]),
                            remove_cooldown=remove_cooldown,
                            restore_ready=restore_ready,
                            risk_ema=risk_ema,
                            train_mode=train_mode,
                            eps_override=eps_now if train_mode else float(args.eps_eval),
                        )

                    desired_type = ACTION_NAMES[selected_bucket]
                    idx_a, lab_a, host_a, sc_a = _pick_best_action(env, agent, desired_type, host_scores)
                    actual_bucket = _label_to_bucket(lab_a)
                    actions_idx[agent] = idx_a
                    chosen_labels[agent] = lab_a
                    chosen_meta[rid] = {
                        "choice": choice,
                        "desired_bucket": selected_bucket,
                        "executed_bucket": actual_bucket,
                        "best_host": best_host,
                        "best_score": best_score,
                        "matched_host": host_a,
                    }
                    ep_action_counts[rid][actual_bucket] = ep_action_counts[rid].get(actual_bucket, 0) + 1
                    region_state[rid]["last_bucket"] = actual_bucket
                    region_state[rid]["last_host"] = host_a if host_a is not None else best_host
                    if actual_bucket == 3:
                        region_state[rid]["remove_cooldown"] = int(args.remove_cooldown_steps)
                        region_state[rid]["recent_removed_host"] = host_a if host_a is not None else best_host
                        region_state[rid]["recent_removed_ttl"] = int(args.restore_after_remove_steps)

                truth_before = _extract_true_state_snapshot(cyborg, None)
                step_t0 = time.perf_counter()
                obs_dict, rewards_dict, terminated, truncated, info_dict = env.step(actions_idx)
                red_last_actions = _extract_red_last_actions(cyborg)
                truth_after = _extract_true_state_snapshot(cyborg, info_dict)
                if ep == 0 and t in [0, 5, 10, 20]:
                    print("\n===== DEBUG TRUE STATE START =====")
                    print("debug_ep =", ep, "debug_t =", t)
                    print("type(info_dict) =", type(info_dict))
                    if isinstance(info_dict, dict):
                        print("info_dict.keys() =", list(info_dict.keys())[:50])

                    print("hasattr(cyborg, 'environment_controller') =", hasattr(cyborg, "environment_controller"))
                    if hasattr(cyborg, "environment_controller"):
                        ec = cyborg.environment_controller
                        print("type(environment_controller) =", type(ec))
                        print("dir(environment_controller) sample =", [x for x in dir(ec) if not x.startswith('_')][:50])

                        if hasattr(ec, "state"):
                            st = ec.state
                            print("type(environment_controller.state) =", type(st))
                            print("dir(state) sample =", [x for x in dir(st) if not x.startswith('_')][:80])

                            for name in ["hosts", "Hosts", "sessions", "Sessions", "subnets", "Subnets", "ip_addresses", "mission_phase", "time"]:
                                if hasattr(st, name):
                                    val = getattr(st, name)
                                    print(f"state.{name} type =", type(val))
                                    try:
                                        if isinstance(val, dict):
                                            print(f"state.{name}.keys sample =", list(val.keys())[:10])
                                        elif isinstance(val, list):
                                            print(f"state.{name} sample =", val[:3])
                                        else:
                                            print(f"state.{name} =", val)
                                    except Exception as e:
                                        print(f"print state.{name} failed:", e)

                            # ===== 新增：打印 sessions 详细结构 =====
                            try:
                                sess_map = getattr(st, "sessions", None)
                                if isinstance(sess_map, dict):
                                    print("---- state.sessions detail sample ----")
                                    for k in list(sess_map.keys())[:8]:
                                        v = sess_map[k]
                                        print(f"[sessions key] {k} -> type={type(v)}")
                                        print(f"repr(session_entry)={repr(v)[:800]}")
                                        if isinstance(v, dict):
                                            print(f"session_entry.keys sample={list(v.keys())[:20]}")
                            except Exception as e:
                                print("print state.sessions detail failed:", e)

                            # ===== 新增：打印 hosts 详细结构 =====
                            try:
                                hosts_map = getattr(st, "hosts", None)
                                if isinstance(hosts_map, dict):
                                    print("---- state.hosts detail sample ----")
                                    for hk in list(hosts_map.keys())[:5]:
                                        hv = hosts_map[hk]
                                        print(f"[host key] {hk} -> type={type(hv)}")
                                        print(f"repr(host)={repr(hv)[:800]}")
                                        print(f"dir(host) sample={[x for x in dir(hv) if not x.startswith('_')][:60]}")

                                        for attr in ["sessions", "processes", "files", "services"]:
                                            if hasattr(hv, attr):
                                                vv = getattr(hv, attr)
                                                print(f"host.{attr} type={type(vv)} repr={repr(vv)[:500]}")
                            except Exception as e:
                                print("print state.hosts detail failed:", e)

                    print("===== DEBUG TRUE STATE END =====\n")

                step_runtime_sec = float(time.perf_counter() - step_t0)

                step_rewards = []
                next_obs_by_region = {}
                for rid in region_ids:
                    agent = region_to_agent[rid]
                    rr = float(rewards_dict.get(agent, 0.0))
                    step_rewards.append(rr)
                    ep_ret_region[rid] += rr
                    raw = _safe_to_dict(obs_dict.get(agent, {}))
                    ev, risk = _extract_events_and_risk(raw)
                    next_obs_by_region[rid] = {"t": t+1, "events": ev, "risk_proxy": risk, "region_id": int(rid.replace("region", ""))}
                team_reward_mean = float(np.mean(step_rewards)) if step_rewards else 0.0
                obs_by_region = next_obs_by_region

                if train_mode:
                    for rid in region_ids:
                        meta = chosen_meta[rid]
                        raw_next = _safe_to_dict(obs_dict.get(region_to_agent[rid], {}))
                        next_host_scores = _host_score_map(raw_next)
                        _, next_best_score = _best_host_from_scores(next_host_scores)
                        bandit.update(
                            meta["choice"],
                            team_reward_mean=team_reward_mean,
                            next_risk=float(obs_by_region[rid].get("risk_proxy", 0.0) or 0.0),
                            next_host_score=float(next_best_score),
                            executed_bucket=int(meta["executed_bucket"]),
                        )

                for rid in region_ids:
                    if region_state[rid]["remove_cooldown"] > 0:
                        region_state[rid]["remove_cooldown"] -= 1
                    if region_state[rid]["recent_removed_ttl"] > 0:
                        region_state[rid]["recent_removed_ttl"] -= 1
                    if region_state[rid]["recent_removed_ttl"] <= 0:
                        region_state[rid]["recent_removed_host"] = None

                step_sum = float(np.sum(step_rewards))
                step_mean = team_reward_mean
                ep_ret_sum += step_sum
                ep_ret_mean += step_mean

                if out_f is not None:
                    blue_action_family = {rid: ACTION_NAMES.get(int(chosen_meta[rid]["executed_bucket"]), "Monitor") for rid in region_ids if rid in chosen_meta}
                    blue_target_host = {rid: (chosen_meta[rid].get("matched_host") or chosen_meta[rid].get("best_host")) for rid in region_ids if rid in chosen_meta}
                    risk_proxy = {rid: float(obs_by_region[rid].get("risk_proxy", 0.0) or 0.0) for rid in region_ids}
                    evidence = {rid: _count_evidence(obs_by_region[rid]) for rid in region_ids}
                    best_host_score = {rid: float(chosen_meta[rid].get("best_score", 0.0) or 0.0) for rid in region_ids if rid in chosen_meta}

                    out_f.write(json.dumps({
                        "script_version": SCRIPT_VERSION,
                        "phase": phase,
                        "episode": ep,
                        "t": t,
                        "step_runtime_sec": step_runtime_sec,
                        "base_actions": base_actions,
                        "actual_counts_so_far": ep_action_counts,
                        "step_rewards": step_rewards,
                        "tasks": [tt.__dict__ for tt in tasks],
                        "chosen_labels": chosen_labels,
                        "blue_action_family": blue_action_family,
                        "blue_target_host": blue_target_host,
                        "risk_proxy": risk_proxy,
                        "alert_count": {rid: float(evidence[rid].get("alert_count", 0.0)) for rid in region_ids},
                        "ioc_count": {rid: float(evidence[rid].get("ioc_count", 0.0)) for rid in region_ids},
                        "best_host_score": best_host_score,

                        "compromised_hosts_before": truth_before.get("compromised_hosts", []),
                        "escalated_hosts_before": truth_before.get("escalated_hosts", []),
                        "compromised_hosts_after": truth_after.get("compromised_hosts", []),
                        "escalated_hosts_after": truth_after.get("escalated_hosts", []),
                        "infected_hosts": truth_after.get("infected_hosts", []),
                        "root_hosts": truth_after.get("root_hosts", []),
                        "impacted_hosts": truth_after.get("impacted_hosts", []),
                        "clean_host_ratio_true": truth_after.get("clean_host_ratio_true", None),
                        "non_escalated_ratio_true": truth_after.get("non_escalated_ratio_true", None),
                        "red_last_actions": red_last_actions,
                        "total_hosts_true": truth_after.get("total_hosts_true", None),
                    }, ensure_ascii=False) + "\n")

            if phase == "eval":
                eval_returns_sum.append(ep_ret_sum)
                eval_returns_mean.append(ep_ret_mean)
                for rid in region_ids:
                    eval_region_returns[rid].append(ep_ret_region[rid])

            if int(args.print_every) > 0 and ((ep + 1) % int(args.print_every) == 0):
                action_brief = []
                for rid in region_ids:
                    c = ep_action_counts[rid]
                    action_brief.append(
                        f"{rid}:M{c.get(0,0)}/A{c.get(1,0)}/Rm{c.get(3,0)}/Rs{c.get(4,0)}/Hs{region_state[rid]['same_host_streak']}"
                    )
                if phase == "eval":
                    run_mean = _mean(eval_returns_mean)
                    run_std = _std(eval_returns_mean)
                else:
                    run_mean = 0.0
                    run_std = 0.0
                print(
                    f"[{phase.upper()} EP {ep+1:03d}/{n_eps:03d}] return_sum={ep_ret_sum:.3f} return_mean={ep_ret_mean:.3f} "
                    f"running_mean={run_mean:.3f} running_std={run_std:.3f} | "
                    + " ; ".join(action_brief)
                )

    if out_f is not None:
        out_f.close()

    print("\n[CC4-STYLE EVAL | OVERALL]")
    print("  backend:", args.backend)
    print("  train_episodes:", int(args.train_episodes))
    print("  eval_episodes:", int(args.episodes))
    print("  steps_per_episode:", int(args.steps))
    print("  mean_reward_sum:", _mean(eval_returns_sum))
    print("  std_reward_sum :", _std(eval_returns_sum))
    print("  mean_reward_mean:", _mean(eval_returns_mean))
    print("  std_reward_mean :", _std(eval_returns_mean))

    print("\n[CC4-STYLE EVAL | PER-REGION]")
    for rid in region_ids:
        arr = eval_region_returns[rid]
        print(f"  {rid}: mean_reward={_mean(arr):.6f} std_reward={_std(arr):.6f}")


if __name__ == "__main__":
    main()
