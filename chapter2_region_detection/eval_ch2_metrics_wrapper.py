from __future__ import annotations
import argparse
import json
from pathlib import Path
from cc4_trace_metrics import eval_ch2_region_jsonl, _aggregate_episode_summaries, _parse_hosts_per_region


def _strip_detail(obj):
    if isinstance(obj, dict):
        obj = dict(obj)
        obj.pop("episodes_detail", None)
        for k, v in list(obj.items()):
            if isinstance(v, dict):
                obj[k] = _strip_detail(v)
        return obj
    return obj


def main() -> None:
    ap = argparse.ArgumentParser(description="第二章扩展指标评测：仅输出聚合平均结果")
    ap.add_argument("--jsonl", nargs="+", required=True, help="各区域 deploy/log jsonl")
    ap.add_argument("--regions", default="", help="与 --jsonl 顺序对应，例如 0,1,2,3")
    ap.add_argument("--total_hosts", type=int, default=None)
    ap.add_argument("--hosts_per_region", default=None)
    ap.add_argument("--out_json", default="")
    args = ap.parse_args()

    hosts_per_region = _parse_hosts_per_region(args.hosts_per_region)
    regions = [x.strip() for x in str(args.regions).split(",") if x.strip()]
    if regions and len(regions) != len(args.jsonl):
        raise ValueError("--regions 个数必须与 --jsonl 一致")

    per_region = {}
    all_eps = []
    for i, path in enumerate(args.jsonl):
        rid = f"region{int(regions[i].replace('region', ''))}" if regions else None
        res = eval_ch2_region_jsonl(path, region_id=rid, total_hosts=args.total_hosts, hosts_per_region=hosts_per_region)
        per_region[res["data_stats"]["region"]] = _strip_detail(res)
        all_eps.extend(res.get("episodes_detail", []))

    overall = _aggregate_episode_summaries(all_eps, runtime_sec=None, data_stats={
        "schema": "chapter2_multi_region",
        "jsonl_files": args.jsonl,
        "regions": sorted(per_region.keys()),
    }, include_detail=False)
    out = {
        "mode": "chapter2",
        "per_region": per_region,
        "overall": overall,
    }
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
