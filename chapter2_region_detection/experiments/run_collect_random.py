# experiments/run_collect_random.py
# 作用：为世界模型收集初始交互轨迹（random policy），保存到 data/replays/
import sys
from pathlib import Path
import argparse
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.utils import ensure_dir, load_yaml, set_seed
from src.env_region import build_region_objects
from src.replay_buffer import ReplayBuffer, Transition


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/cc4.yaml")
    ap.add_argument("--region_id", type=int, required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = load_yaml(str((ROOT / args.cfg).resolve()))
    set_seed(cfg.get("seed", 42))

    objs = build_region_objects(cfg, region_id=args.region_id, device=args.device)

    replay_dir = (ROOT / cfg["paths"]["replay_dir"]).resolve()
    ensure_dir(str(replay_dir))
    out_path = replay_dir / f"replay_region{args.region_id}.npz"

    buf = ReplayBuffer(capacity=500000)

    obs = objs.client.reset()
    objs.summarizer.reset()

    for _ in range(args.steps):
        events = obs.get("events", [])
        risk_proxy = float(obs.get("risk_proxy", 0.0))
        objs.summarizer.update(events)
        summary = objs.summarizer.build(risk_proxy=risk_proxy)

        a = int(np.random.randint(0, 5))
        obs2, r, done, _ = objs.client.step(a)

        objs.summarizer.update(obs2.get("events", []))
        summary2 = objs.summarizer.build(risk_proxy=float(obs2.get("risk_proxy", 0.0)))

        buf.add(Transition(s=summary.vec, a=a, r=r, s2=summary2.vec, done=done))

        obs = obs2
        if done:
            obs = objs.client.reset()
            objs.summarizer.reset()

    s = np.stack([tr.s for tr in buf.buf]).astype(np.float32)
    a = np.array([tr.a for tr in buf.buf], dtype=np.int64)
    s2 = np.stack([tr.s2 for tr in buf.buf]).astype(np.float32)
    np.savez(out_path, s=s, a=a, s2=s2)
    print("[OK] replay saved:", out_path, "n=", len(buf))


if __name__ == "__main__":
    main()
