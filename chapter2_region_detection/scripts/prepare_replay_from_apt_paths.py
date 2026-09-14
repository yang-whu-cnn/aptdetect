# scripts/prepare_replay_from_apt_paths.py
# Purpose: Convert APT path json files (apt_attack_path_*.json) into per-region replay jsonl files
#          that can be directly consumed by ReplayCC4Client:
#          data/raw/cc4_export_region{region_id}.jsonl
#
# Usage (run at project root):
#   python scripts/prepare_replay_from_apt_paths.py --apt_dir "<dir_with_apt_attack_path_json>" --subset success
#   python scripts/prepare_replay_from_apt_paths.py --apt_dir "<dir_with_apt_attack_path_json>" --subset all
#
# After conversion, set configs/cc4.yaml: mode: replay
# Then run:
#   python experiments/run_deploy_region.py --region_id 0

from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any


# --------- 1) Zone/Subnet -> region_id mapping (edit if your thesis uses a different region ordering) ----------
ZONE2RID_ZONE: Dict[str, int] = {
    # Fine-grained (8 security zones)
    "contractor_network_subnet": 0,
    "public_access_zone_subnet": 1,
    "office_network_subnet": 2,
    "admin_network_subnet": 3,
    "restricted_zone_a_subnet": 4,
    "operational_zone_a_subnet": 5,
    "restricted_zone_b_subnet": 6,
    "operational_zone_b_subnet": 7,
}

ZONE2RID_SUBNET: Dict[str, int] = {
    # Coarse-grained (4 subnets / macro regions in CC4)
    # 0: HQ network (public access + office + admin)
    # 1: Deployment network A (restricted A + operational A)
    # 2: Deployment network B (restricted B + operational B)
    # 3: Contractor network
    "public_access_zone_subnet": 0,
    "office_network_subnet": 0,
    "admin_network_subnet": 0,
    "restricted_zone_a_subnet": 1,
    "operational_zone_a_subnet": 1,
    "restricted_zone_b_subnet": 2,
    "operational_zone_b_subnet": 2,
    "contractor_network_subnet": 3,
}

# rid2zone is built per granularity inside convert()

ZONE_RE = re.compile(r"(.+?_subnet)")


# --------- 2) Minimal action -> unified events mapping (for pipeline test; real CC4 online should use true alerts) ----------
def action_to_events(action: str, success: str, agent: str, target_host: str, ts: int) -> List[Dict[str, Any]]:
    """
    Output events using the unified schema required by src/state_summary.py:
      {ts, type in [auth_fail, port_scan, proc_spawn, outbound_conn], severity, src, dst}
    """
    a = (action or "").strip()
    s = (success or "UNKNOWN").strip().upper()
    src = agent or "red"
    dst = target_host or ""

    evs: List[Dict[str, Any]] = []
    def add(t: str, sev: float):
        evs.append({"ts": ts, "type": t, "severity": float(sev), "src": src, "dst": dst})

    if a == "DiscoverRemoteSystems":
        add("port_scan", 0.30)
    elif a == "AggressiveServiceDiscovery":
        add("port_scan", 0.40)
    elif a == "ExploitRemoteService":
        add("proc_spawn", 0.75 if s == "TRUE" else 0.55)
    elif a == "PrivilegeEscalate":
        # treat privilege escalation as strong auth anomaly
        add("auth_fail", 0.85 if s == "TRUE" else 0.65)
    elif a == "Impact":
        # strong impact: process anomaly + network burst
        add("proc_spawn", 1.00)
        add("outbound_conn", 0.90)
    elif a == "DegradeServices":
        add("outbound_conn", 0.90)
    else:
        # Sleep / Withdraw / InvalidAction / others: no new event (keeps the window from being polluted)
        pass

    return evs


# --------- 3) A simple risk proxy (for replay test) ----------
def update_risk(prev: float, action: str, success: str) -> float:
    a = (action or "").strip()
    s = (success or "UNKNOWN").strip().upper()
    r = float(prev)

    if a == "DiscoverRemoteSystems":
        r += 0.02
    elif a == "AggressiveServiceDiscovery":
        r += 0.03
    elif a == "ExploitRemoteService":
        r += 0.08 if s == "TRUE" else 0.04
    elif a == "PrivilegeEscalate":
        r += 0.06 if s == "TRUE" else 0.03
    elif a in ("Impact", "DegradeServices"):
        r += 0.15
    elif a == "Withdraw":
        r -= 0.10
    elif a == "Sleep":
        r -= 0.02
    else:
        # InvalidAction / others: no change
        pass

    if r < 0.0:
        r = 0.0
    if r > 1.0:
        r = 1.0
    return r


def extract_zone(target_host: str) -> str:
    if not target_host:
        return ""
    m = ZONE_RE.match(target_host)
    return m.group(1) if m else ""


def iter_apt_files(apt_dir: Path, subset: str) -> List[Path]:
    files = sorted(apt_dir.glob("apt_attack_path_*.json"))
    if subset == "all":
        return files
    if subset == "failure":
        return [p for p in files if re.search(r"_\d+_1\.json$", p.name)]
    # subset == "success"
    return [p for p in files if not re.search(r"_\d+_1\.json$", p.name)]


def convert(apt_dir: Path, out_raw_dir: Path, subset: str, granularity: str) -> Tuple[int, List[Path]]:
    out_raw_dir.mkdir(parents=True, exist_ok=True)    # choose granularity: subnet=4 macro regions (HQ/DeployA/DeployB/Contractor), zone=8 security zones
    if granularity == "zone":
        zone2rid = dict(ZONE2RID_ZONE)
        rid2name = {rid: zone for zone, rid in zone2rid.items()}  # mostly for debugging
    else:
        zone2rid = dict(ZONE2RID_SUBNET)
        rid2name = {0: "hq_network", 1: "deployment_a", 2: "deployment_b", 3: "contractor"}

    region_ids = sorted(set(zone2rid.values()))
    buffers: Dict[int, List[Dict[str, Any]]] = {rid: [] for rid in region_ids}
    risk_state: Dict[int, float] = {rid: 0.05 for rid in region_ids}


    files = iter_apt_files(apt_dir, subset=subset)
    used_files: List[Path] = []

    for fp in files:
        try:
            recs = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] skip {fp.name}: {e}")
            continue

        if not isinstance(recs, list) or len(recs) == 0:
            print(f"[WARN] empty or invalid list: {fp.name}")
            continue

        used_files.append(fp)

        for rec in recs:
            # Some files have a first "reset record" with only "observation"
            action = rec.get("action")
            if not action:
                continue

            target_host = rec.get("target_host", "")
            zone = extract_zone(target_host)
            if zone not in zone2rid:
                # fallback: try to find any host key in observation to infer zone
                obs = rec.get("observation", {})
                if isinstance(obs, dict):
                    for hk in obs.keys():
                        z2 = extract_zone(hk)
                        if z2 in zone2rid:
                            zone = z2
                            break

            if zone not in zone2rid:
                continue

            rid = zone2rid[zone]
            t_local = len(buffers[rid])  # local time index for this region

            agent = rec.get("agent", "red")
            success = str(rec.get("success", "UNKNOWN"))

            # update risk proxy
            risk_state[rid] = update_risk(risk_state[rid], action, success)

            events = action_to_events(action, success, str(agent), target_host, ts=t_local)
            obs = {
                "region_id": rid,
                "t": t_local,
                "events": events,
                "risk_proxy": float(risk_state[rid]),
            }
            info = {
                "src_file": fp.name,
                "orig_step": rec.get("step"),
                "zone": zone,
                "action": action,
                "success": success,
                "target_host": target_host,
                "risk": float(risk_state[rid]),
            }
            buffers[rid].append({"obs": obs, "reward": 0.0, "done": False, "info": info})

    # write jsonl per region
    total_lines = 0
    for rid, rows in buffers.items():
        out_fp = out_raw_dir / f"cc4_export_region{rid}.jsonl"
        if len(rows) == 0:
            # still create an empty file for consistency
            out_fp.write_text("", encoding="utf-8")
            continue
        rows[-1]["done"] = True
        out_fp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
        total_lines += len(rows)

    return total_lines, used_files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apt_dir", type=str, required=True, help="Directory containing apt_attack_path_*.json")
    ap.add_argument("--out_raw_dir", type=str, default="data/raw", help="Output directory for replay jsonl files")
    ap.add_argument("--subset", type=str, default="success", choices=["success", "failure", "all"],
                    help="Which files to convert: success (default), failure (*_1.json), or all")
    ap.add_argument("--granularity", type=str, default="subnet", choices=["subnet", "zone"],
                    help="How to split regions: subnet=4 macro regions (HQ/DeployA/DeployB/Contractor), zone=8 security zones")
    args = ap.parse_args()

    apt_dir = Path(args.apt_dir)
    if not apt_dir.exists():
        raise FileNotFoundError(f"apt_dir not found: {apt_dir}")

    out_raw_dir = Path(args.out_raw_dir)
    total_lines, used_files = convert(apt_dir, out_raw_dir, subset=args.subset, granularity=args.granularity)
    print(f"[OK] converted {len(used_files)} files -> {out_raw_dir} ; total replay steps={total_lines}")
    print("[OK] generated:", ", ".join(sorted([p.name for p in out_raw_dir.glob('cc4_export_region*.jsonl')])))


if __name__ == "__main__":
    main()
