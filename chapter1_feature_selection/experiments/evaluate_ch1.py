import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import argparse
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import torch

torch.set_num_threads(1)

from src.utils import load_yaml, load_json, save_json, ensure_dir, set_seed
from src.detector import train_logreg_detector
from src.ppo_agent import ActorNet
from src.sensitivity import compute_field_utility_scores


def map_fields_to_cols(feature_meta: dict, candidate_fields: list) -> np.ndarray:
    cols = []
    for f in candidate_fields:
        if f not in feature_meta["field_slices"]:
            continue
        s, e = feature_meta["field_slices"][f]
        cols.extend(range(s, e))
    return np.array(sorted(set(cols)), dtype=np.int64)


def run_dynamic_prfs(XF_test, detector, actor, k_flip, tau, device):
    T, D = XF_test.shape
    mask_prob = np.ones(D, dtype=np.float32)
    mask_hard = np.ones(D, dtype=np.float32)
    keep_dims = []

    for t in range(T):
        x_now = XF_test[t] * mask_hard
        eps = float(detector.uncertainty(x_now.reshape(1, -1))[0])

        state = np.concatenate([[eps], mask_prob], axis=0).astype(np.float32)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            logits = actor(s).squeeze(0)
            probs = torch.sigmoid(logits)
            topk = torch.topk(probs, k=min(k_flip, D)).indices.cpu().numpy()

        hard = (mask_prob >= 0.5).astype(np.float32)
        for idx in topk:
            if 0 <= idx < D:
                hard[idx] = 1.0 - hard[idx]

        mask_prob = (1 - tau) * mask_prob + tau * hard
        mask_hard = (mask_prob >= 0.5).astype(np.float32)

        keep_dims.append(int(mask_hard.sum()))

    return np.array(keep_dims, dtype=int)


def main(cfg_path: str = "configs/cc4.yaml", split_ratio: float = 0.7):
    set_seed(42)
    cfg = load_yaml(str((ROOT / cfg_path).resolve()))
    data_dir = ROOT / "data"
    fig_dir = ROOT / "figures"
    ensure_dir(str(fig_dir))

    print("[EVAL] loading arrays...")
    X = np.load(data_dir / "X.npy").astype(np.float32)
    y = np.load(data_dir / "y.npy").astype(np.int64)
    meta = load_json(str(data_dir / "feature_meta.json"))
    cand = load_json(str(data_dir / "candidate_fields.json"))["candidate_fields"]
    raw_scores = load_json(str(data_dir / "field_sensitivity.json"))

    utility_scores = compute_field_utility_scores(
        raw_scores=raw_scores,
        field_slices={k: tuple(v) for k, v in meta["field_slices"].items()},
        score_norm=str(cfg["candidate_pool"].get("score_norm", "none")),
    )

    cand_cols = map_fields_to_cols(meta, cand)
    XF = X[:, cand_cols]

    n = len(X)
    split = int(n * split_ratio)
    XF_tr, XF_te = XF[:split], XF[split:]
    y_tr, y_te = y[:split], y[split:]

    # 这里 detector 只用于 PRF-S 回放环境/状态摘要，不作为正文主指标输出
    det_cfg = cfg.get("detector", {})
    detector = train_logreg_detector(
        XF_tr,
        y_tr,
        threshold=float(det_cfg.get("threshold", 0.5)),
        class_weight=det_cfg.get("class_weight", None),
        C=float(det_cfg.get("C", 1.0)),
        max_iter=int(det_cfg.get("max_iter", 1000)),
    )

    keep_dims = None
    actor_path = ROOT / "checkpoints" / "ppo_actor.pt"
    if actor_path.exists():
        print("[EVAL] loading best PRF-S actor and simulating...")
        device = cfg.get("clfs", {}).get("device", "cpu")
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        actor = ActorNet(state_dim=1 + XF.shape[1], action_dim=XF.shape[1]).to(device)
        actor.load_state_dict(torch.load(actor_path, map_location=device))
        actor.eval()

        prcfg = cfg["prfs"]
        keep_dims = run_dynamic_prfs(
            XF_te,
            detector,
            actor,
            k_flip=int(prcfg["k_flip"]),
            tau=float(prcfg["tau_soft_update"]),
            device=device,
        )

    orig_dim = int(X.shape[1])
    candidate_dim = int(XF.shape[1])
    avg_keep_dim = float(np.mean(keep_dims)) if keep_dims is not None else None

    results = {
        "split_ratio": float(split_ratio),
        "num_train": int(split),
        "num_test": int(n - split),
        "logical_field_count": int(len(meta["fields_order"])),
        "window_feature_dim": orig_dim,
        "candidate_field_count": int(len(cand)),
        "candidate_feature_dim": candidate_dim,
        "avg_keep_dim_prfs": avg_keep_dim,
        "avg_keep_ratio_prfs": float(avg_keep_dim / candidate_dim) if avg_keep_dim is not None else None,
        "compression_rate_clfs": float(1.0 - candidate_dim / max(orig_dim, 1)),
        "compression_rate_prfs_from_full": float(1.0 - avg_keep_dim / max(orig_dim, 1)) if avg_keep_dim is not None else None,
        "compression_rate_prfs_from_candidate": float(1.0 - avg_keep_dim / max(candidate_dim, 1)) if avg_keep_dim is not None else None,
        "candidate_fields": cand,
        "kept_dim_stats": {
            "min": int(np.min(keep_dims)) if keep_dims is not None else None,
            "max": int(np.max(keep_dims)) if keep_dims is not None else None,
            "mean": float(np.mean(keep_dims)) if keep_dims is not None else None,
            "p10": float(np.percentile(keep_dims, 10)) if keep_dims is not None else None,
            "p50": float(np.percentile(keep_dims, 50)) if keep_dims is not None else None,
            "p90": float(np.percentile(keep_dims, 90)) if keep_dims is not None else None,
        },
        "note": "本文件以特征选择指标为主；代理分类指标不再作为正文主结果输出。",
    }
    save_json(results, str(data_dir / "evaluation_summary.json"))

    # 图1：字段“效用”排序图（不是原始 sensitivity）
    utility_sorted = sorted(utility_scores.items(), key=lambda x: float(x[1]), reverse=True)
    labels = [k for k, _ in utility_sorted]
    vals = [float(v) for _, v in utility_sorted]
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(labels)), vals)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel("Utility score")
    plt.title("Field utility ranking")
    plt.tight_layout()
    plt.savefig(fig_dir / "field_utility_ranking.png", dpi=200)
    plt.close()

    # 图2：维度压缩汇总图
    names = ["Window features", "CL-FS candidate"]
    dims = [orig_dim, candidate_dim]
    if avg_keep_dim is not None:
        names.append("PRF-S avg kept")
        dims.append(avg_keep_dim)

    plt.figure(figsize=(7, 5))
    plt.bar(range(len(names)), dims)
    plt.xticks(range(len(names)), names, rotation=15)
    plt.ylabel("Dimension")
    plt.title("Dimension reduction summary")
    for i, v in enumerate(dims):
        plt.text(i, v, f"{v:.1f}" if isinstance(v, float) else str(v), ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(fig_dir / "dimension_reduction_summary.png", dpi=200)
    plt.close()

    # 图3：压缩率对比图
    rate_names = ["CL-FS/full", "PRF-S/full", "PRF-S/candidate"]
    rate_vals = [
        results["compression_rate_clfs"],
        results["compression_rate_prfs_from_full"] if results["compression_rate_prfs_from_full"] is not None else 0.0,
        results["compression_rate_prfs_from_candidate"] if results["compression_rate_prfs_from_candidate"] is not None else 0.0,
    ]
    plt.figure(figsize=(7, 5))
    plt.bar(range(len(rate_names)), rate_vals)
    plt.xticks(range(len(rate_names)), rate_names, rotation=15)
    plt.ylabel("Compression rate")
    plt.title("Compression rate comparison")
    for i, v in enumerate(rate_vals):
        plt.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(fig_dir / "compression_rate_summary.png", dpi=200)
    plt.close()

    # 图4：PRF-S 动态保留维度曲线
    if keep_dims is not None:
        show_n = min(300, len(keep_dims))
        plt.figure(figsize=(9, 4))
        plt.plot(np.arange(show_n), keep_dims[:show_n])
        plt.xlabel("Test window index")
        plt.ylabel("Kept dimensions")
        plt.title("PRF-S dynamic kept dimensions (first 300 test windows)")
        plt.tight_layout()
        plt.savefig(fig_dir / "prfs_kept_dimensions_curve.png", dpi=200)
        plt.close()

    print(f"[OK] evaluation saved -> {data_dir / 'evaluation_summary.json'}")
    print(f"[OK] figures saved -> {fig_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/cc4.yaml")
    parser.add_argument("--split-ratio", type=float, default=0.7)
    args = parser.parse_args()
    main(cfg_path=args.cfg, split_ratio=args.split_ratio)