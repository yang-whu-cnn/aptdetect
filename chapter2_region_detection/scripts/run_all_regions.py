# scripts/run_all_regions.py
# 作用：批量跑多个区域（单区域代码复用）。默认串行，必要时你可以改为 multiprocessing。
import subprocess
import sys
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/cc4.yaml")
    ap.add_argument("--regions", type=str, default="0,1,2")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    regions = [int(x) for x in args.regions.split(",") if x.strip()]

    for rid in regions:
        cmd = [
            sys.executable,
            str(ROOT / "experiments" / "run_train_posterior.py"),
            "--cfg", args.cfg,
            "--region_id", str(rid),
            "--device", args.device,
        ]
        print("\n[RUN]", " ".join(cmd))
        subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
