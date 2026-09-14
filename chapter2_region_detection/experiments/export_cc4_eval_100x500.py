# -*- coding: utf-8 -*-
"""
批量导出第二章部署日志为 CC4 风格：100 episodes × 500 steps
会生成第三章可直接读取的 jsonl：deploy/random/noop_region*.jsonl
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]):
    print("\n[RUN]", " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/cc4_local_online_qwen.yaml")
    ap.add_argument("--regions", default="0,1,2,3")
    ap.add_argument("--out_dir", default="outputs/local_online_deploy")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--skip_baselines", action="store_true")
    args = ap.parse_args()

    out_dir = (ROOT / args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    regions = [int(x) for x in str(args.regions).split(",") if x.strip()]

    # deploy（LLM+WM+PPO）
    for rid in regions:
        run([
            sys.executable, "-m", "experiments.run_deploy_region",
            "--cfg", args.cfg,
            "--region_id", str(rid),
            "--device", args.device,
            "--seed", str(args.seed),
            "--jsonl_path", str(out_dir / f"deploy_region{rid}.jsonl"),
            "--log_path", str(out_dir / f"deploy_region{rid}.log"),
            "--cc4_eval",
            "--deterministic",
        ])

    if args.skip_baselines:
        return

    # baselines（random / noop）
    for policy in ("random", "noop"):
        for rid in regions:
            run([
                sys.executable, "-m", "experiments.run_deploy_baseline",
                "--cfg", args.cfg,
                "--region_id", str(rid),
                "--device", args.device,
                "--seed", str(args.seed),
                "--policy", policy,
                "--out_jsonl", str(out_dir / f"{policy}_region{rid}.jsonl"),
                "--cc4_eval",
            ])


if __name__ == "__main__":
    main()
