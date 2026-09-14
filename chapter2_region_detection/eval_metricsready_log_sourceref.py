
from __future__ import annotations
import argparse, json, statistics
from collections import defaultdict
from typing import Dict, Any, List, Set


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def std(xs: List[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    return statistics.stdev(xs)


def _sum_map(d: Any) -> float:
    if not isinstance(d, dict):
        return 0.0
    s = 0.0
    for v in d.values():
        try:
            s += float(v or 0.0)
        except Exception:
            pass
    return s


def _first_alert_step(rec: Dict[str, Any], risk_threshold: float = 0.5) -> bool:
    # User-requested lead-time metric: use a meaningful alert threshold, not any tiny risk.
    if _sum_map(rec.get("alert_count")) > 0:
        return True
    if _sum_map(rec.get("ioc_count")) > 0:
        return True
    rp = rec.get("risk_proxy")
    if isinstance(rp, dict):
        for v in rp.values():
            try:
                if float(v or 0.0) >= risk_threshold:
                    return True
            except Exception:
                pass
    return False


def _is_recovery_family(name: Any) -> bool:
    s = str(name or "")
    return ("Remove" in s) or ("Restore" in s)


def _is_remove(name: Any) -> bool:
    return "Remove" in str(name or "")


def _iter_targets(target_map: Any) -> List[str]:
    out: List[str] = []
    if isinstance(target_map, dict):
        for v in target_map.values():
            if isinstance(v, str) and v:
                out.append(v)
    return out


def _impact_events_from_red_actions(red_last_actions: Any) -> int:
    count = 0
    if isinstance(red_last_actions, list):
        for a in red_last_actions:
            if not isinstance(a, dict):
                continue
            if str(a.get("name", "")) != "Impact":
                continue
            hostname = str(a.get("hostname", "") or "")
            if "operational" in hostname:
                count += 1
    return count


def eval_metricsready_log(path: str, out_json: str | None = None, risk_threshold: float = 0.5):
    rows = [r for r in load_jsonl(path) if r.get("phase") == "eval"]
    buckets: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        buckets[int(r.get("episode", 0))].append(r)
    for ep in buckets:
        buckets[ep].sort(key=lambda r: int(r.get("t", 0)))

    per_episode = []
    for ep, recs in sorted(buckets.items()):
        reward_trace: List[float] = []
        clean_frac_trace: List[float] = []
        nonesc_frac_trace: List[float] = []
        runtimes: List[float] = []

        true_pos = 0
        false_pos = 0
        impact_count = 0

        compromised_length: Dict[str, int] = {}
        compromised_lengths_list: List[int] = []

        first_attack = None
        first_escalation = None
        first_impact = None
        first_detection = None
        attack_end = None

        for rec in recs:
            t = int(rec.get("t", 0))

            srs = rec.get("step_rewards") or []
            if isinstance(srs, list) and srs:
                step_mean = sum(float(x or 0.0) for x in srs) / len(srs)
                reward_trace.append(step_mean)

            rt = rec.get("step_runtime_sec")
            if rt is not None:
                try:
                    runtimes.append(float(rt))
                except Exception:
                    pass

            before_comp = set(rec.get("compromised_hosts_before") or [])
            before_esc = set(rec.get("escalated_hosts_before") or [])
            after_comp = set(rec.get("compromised_hosts_after") or rec.get("infected_hosts") or [])
            after_esc = set(rec.get("escalated_hosts_after") or rec.get("root_hosts") or [])
            after_imp = set(rec.get("impacted_hosts") or [])

            total_hosts = int(rec.get("total_hosts_true") or max(len(after_comp | after_esc | after_imp), 1))
            total_hosts = max(total_hosts, 1)

            clean_frac_trace.append((total_hosts - len(after_comp)) / total_hosts)
            nonesc_frac_trace.append((total_hosts - len(after_esc)) / total_hosts)

            # compromise / escalation / impact timing
            if after_comp and first_attack is None:
                first_attack = t
            if after_esc and first_escalation is None:
                first_escalation = t
            if after_imp and first_impact is None:
                first_impact = t
            if after_comp or after_esc or after_imp or _impact_events_from_red_actions(rec.get("red_last_actions")) > 0:
                attack_end = t

            if first_detection is None and _first_alert_step(rec, risk_threshold=risk_threshold):
                first_detection = t

            # Source-aligned TP/FP
            fam = rec.get("blue_action_family") or {}
            targets = _iter_targets(rec.get("blue_target_host") or {})
            if isinstance(fam, dict):
                for rid, action_name in fam.items():
                    if not _is_recovery_family(action_name):
                        continue
                    host = None
                    target_map = rec.get("blue_target_host") or {}
                    if isinstance(target_map, dict):
                        host = target_map.get(rid)
                    if not host:
                        false_pos += 1
                        continue
                    if host not in before_comp:
                        false_pos += 1
                    elif _is_remove(action_name) and host in before_esc:
                        false_pos += 1
                    else:
                        true_pos += 1

            # Source-aligned impact count
            impact_count += _impact_events_from_red_actions(rec.get("red_last_actions"))

            # Source-aligned mean time to recover
            all_hosts = before_comp | after_comp | set(compromised_length.keys())
            for h in all_hosts:
                compromised_before = h in before_comp
                compromised_after = h in after_comp
                if compromised_after:
                    if h not in compromised_length:
                        compromised_length[h] = 0
                    compromised_length[h] += 1
                elif (not compromised_after) and compromised_before:
                    compromised_lengths_list.append(compromised_length.get(h, 0))
                    compromised_length[h] = 0

        # episode-end aggregate per source style
        denom = true_pos + false_pos
        recovery_err = (false_pos / denom) if denom else None
        recovery_precision = (1.0 - recovery_err) if recovery_err is not None else None

        lead_es = (first_escalation - first_detection) if (first_escalation is not None and first_detection is not None) else None
        lead_im = (first_impact - first_detection) if (first_impact is not None and first_detection is not None) else None
        lead_end = (attack_end - first_detection) if (attack_end is not None and first_detection is not None) else None
        det_delay = (first_detection - first_attack) if (first_attack is not None and first_detection is not None) else None

        per_episode.append({
            "episode": ep,
            "steps": len(recs),
            "reward_mean": mean(reward_trace),
            "reward_return": sum(reward_trace),
            "clean_hosts_ratio": mean(clean_frac_trace),
            "non_escalated_hosts_ratio": mean(nonesc_frac_trace),
            "mean_time_to_recover": mean(compromised_lengths_list),
            "useful_recoveries_tp": true_pos,
            "wasted_recoveries_fp": false_pos,
            "recovery_precision": recovery_precision,
            "recovery_error": recovery_err,
            "red_impact_count": impact_count,
            "running_time_sec": sum(runtimes) if runtimes else None,
            "first_attack_step": first_attack,
            "first_detection_step": first_detection,
            "first_escalation_step": first_escalation,
            "first_impact_step": first_impact,
            "attack_end_step": attack_end,
            "lead_time_to_escalation": lead_es,
            "lead_time_to_impact": lead_im,
            "lead_time_to_attack_end": lead_end,
            "detection_delay_from_attack_start": det_delay,
        })

    def pick(name: str) -> List[float]:
        vals = [ep[name] for ep in per_episode if ep.get(name) is not None]
        return vals

    overall = {
        "episodes_eval": len(per_episode),
        "reward_mean": mean(pick("reward_mean")),
        "reward_std": std(pick("reward_mean")),
        "return_mean": mean(pick("reward_return")),
        "return_std": std(pick("reward_return")),
        "clean_hosts_ratio": mean(pick("clean_hosts_ratio")),
        "non_escalated_hosts_ratio": mean(pick("non_escalated_hosts_ratio")),
        "mean_time_to_recover": mean(pick("mean_time_to_recover")),
        "useful_recoveries_tp_total": int(sum(pick("useful_recoveries_tp"))),
        "useful_recoveries_tp_per_episode": mean(pick("useful_recoveries_tp")),
        "wasted_recoveries_fp_total": int(sum(pick("wasted_recoveries_fp"))),
        "wasted_recoveries_fp_per_episode": mean(pick("wasted_recoveries_fp")),
        "recovery_precision": mean(pick("recovery_precision")),
        "recovery_error": mean(pick("recovery_error")),
        "red_impact_count_total": int(sum(pick("red_impact_count"))),
        "red_impact_count_per_episode": mean(pick("red_impact_count")),
        "running_time_sec_total": sum(v for v in pick("running_time_sec") if v is not None),
        "running_time_sec_per_episode": mean([v for v in pick("running_time_sec") if v is not None]),
        # user-defined early detection metrics
        "mean_lead_time_to_escalation": mean(pick("lead_time_to_escalation")),
        "mean_lead_time_to_impact": mean(pick("lead_time_to_impact")),
        "mean_lead_time_to_attack_end": mean(pick("lead_time_to_attack_end")),
        "mean_detection_delay_from_attack_start": mean(pick("detection_delay_from_attack_start")),
        "lead_time_alert_risk_threshold": risk_threshold,
    }

    payload = {"overall": overall, "per_episode": per_episode}
    if out_json:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(overall, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--risk_threshold", type=float, default=0.5)
    args = ap.parse_args()
    eval_metricsready_log(args.jsonl, args.out_json, risk_threshold=args.risk_threshold)
