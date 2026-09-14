# experiments/train_world_model_local.py
import os
import sys
import argparse
import numpy as np

# --- make project root importable so "import src.xxx" works ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import load_yaml, set_seed
from src.env_region import build_region_objects


def load_npz_triplet(npz_path: str):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Replay npz not found: {npz_path}")

    d = np.load(npz_path, allow_pickle=True)
    try:
        # 必须包含 s/a/s2
        for k in ("s", "a", "s2"):
            if k not in d.files:
                raise KeyError(f"npz missing key '{k}', got keys={d.files}")

        s = d["s"]
        a = d["a"]
        s2 = d["s2"]
    finally:
        d.close()

    # 类型/形状做一次“稳健化”
    s = np.asarray(s, dtype=np.float32)
    s2 = np.asarray(s2, dtype=np.float32)
    a = np.asarray(a)  # a 通常是 int/float 都行，交给模型内部处理
    if a.dtype.kind not in ("i", "u"):
        # 若动作不是整数，转一下（更符合离散动作）
        a = a.astype(np.int64)

    if len(s) != len(a) or len(s2) != len(a):
        raise ValueError(f"Length mismatch: len(s)={len(s)}, len(a)={len(a)}, len(s2)={len(s2)}")

    return s, a, s2


def pick_world_model(objs):
    # 兼容不同命名
    for name in ("world_model", "wm", "dynamics", "model"):
        if hasattr(objs, name):
            return getattr(objs, name)
    raise RuntimeError(
        "Cannot find world model object in build_region_objects() return. "
        "Tried attributes: world_model / wm / dynamics / model"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--region_id", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--replay", required=True)
    ap.add_argument("--outdir", default="outputs/local_online_wm")
    args = ap.parse_args()

    cfg = load_yaml(args.cfg)
    os.makedirs(args.outdir, exist_ok=True)

    # seed（没有就用0）
    set_seed(int(cfg.get("seed", 0)))

    # 构建 region 相关对象（含 world model）
    objs = build_region_objects(cfg, region_id=args.region_id, device=args.device)
    wm = pick_world_model(objs)

    # 从 npz 里读取 (s, a, s2)
    s, a, s2 = load_npz_triplet(args.replay)
    print(f"[INFO] loaded replay: s={s.shape} a={a.shape} s2={s2.shape} dtype(s)={s.dtype} dtype(a)={a.dtype}")

    # 关键：你的 train() 签名就是 (s, a, s2)
    # 日志里已经确认：[INFO] train() signature: (s: np.ndarray, a: np.ndarray, s2: np.ndarray)
    print("[START] world_model.train(s, a, s2) ...")
    ret = wm.train(s, a, s2)
    print("[DONE] world model training finished. return =", ret)

    # 可选：落个标记文件，方便你确认 outdir 不是空的
    mark = os.path.join(args.outdir, "DONE.txt")
    with open(mark, "w", encoding="utf-8") as f:
        f.write("world model trained OK\n")
        f.write(f"replay={args.replay}\n")
        f.write(f"s={s.shape} a={a.shape} s2={s2.shape}\n")
    print("[DONE] wrote:", mark)


if __name__ == "__main__":
    main()
