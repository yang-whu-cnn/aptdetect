# experiments/run_train_world_model.py
# 作用：用 data/replays/replay_region{rid}.npz 训练世界模型，并保存到 checkpoints/
import sys
from pathlib import Path
import argparse
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.utils import ensure_dir, load_yaml, set_seed
from src.env_region import build_region_objects


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/cc4.yaml")
    ap.add_argument("--region_id", type=int, required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = load_yaml(str((ROOT / args.cfg).resolve()))
    set_seed(cfg.get("seed", 42))

    objs = build_region_objects(cfg, region_id=args.region_id, device=args.device)

    replay_dir = (ROOT / cfg["paths"]["replay_dir"]).resolve()
    in_path = replay_dir / f"replay_region{args.region_id}.npz"
    if not in_path.exists():
        raise FileNotFoundError(f"missing replay: {in_path}. 先运行 run_collect_random.py")

    data = np.load(in_path)
    s, a, s2 = data["s"], data["a"], data["s2"]
    print("[OK] loaded replay:", s.shape, a.shape, s2.shape)

    objs.world_model.train(s=s, a=a, s2=s2)

    ckpt_dir = (ROOT / cfg["paths"]["ckpt_dir"]).resolve()
    ensure_dir(str(ckpt_dir))
    out = ckpt_dir / f"world_model_region{args.region_id}.pt"

    import torch
    torch.save([m.state_dict() for m in objs.world_model.models], out)
    print("[OK] saved:", out)


if __name__ == "__main__":
    main()
