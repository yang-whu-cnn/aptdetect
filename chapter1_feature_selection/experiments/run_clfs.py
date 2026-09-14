import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import os
import numpy as np
import torch

from src.utils import load_yaml, ensure_dir, save_json, set_seed
from src.dataset import load_events_as_df, build_windows, build_labels, save_windows_table, load_windows_table
from src.preprocess import WindowFeaturizer
from src.clfs import train_clfs, encode_windows
from src.sensitivity import compute_field_sensitivity, build_candidate_pool, compute_field_utility_scores


def main(cfg_path: str = "configs/cc4.yaml"):
    set_seed(42)
    cfg_path = (ROOT / cfg_path).resolve()
    cfg = load_yaml(str(cfg_path))

    dcfg = cfg["data"]
    dcfg["raw_json_path"] = str((ROOT / dcfg["raw_json_path"]).resolve())
    dcfg["windows_path"] = str((ROOT / dcfg["windows_path"]).resolve())

    print("[PATH CHECK] raw_json_path =", dcfg["raw_json_path"])
    print("[PATH CHECK] windows_path  =", dcfg["windows_path"])

    events = load_events_as_df(
        dcfg["raw_json_path"],
        timestamp_key=dcfg["timestamp_key"],
        agent_id_key=dcfg.get("agent_id_key", None),
        event_type_key=dcfg.get("event_type_key", None),
        label_key=dcfg.get("label_key", None),
        attack_stage_key=dcfg.get("attack_stage_key", None),
    )

    windows = build_windows(events, dcfg["window_size_sec"], dcfg["window_step_sec"])
    y = build_labels(
        windows,
        label_mode=dcfg["label_mode"],
        early_warning_horizon_sec=dcfg["early_warning_horizon_sec"],
        window_size_sec=dcfg["window_size_sec"],
    )

    ensure_dir(os.path.dirname(dcfg["windows_path"]))
    windows_saved_path = save_windows_table(windows, y, dcfg["windows_path"])
    print(f"[OK] windows saved -> {windows_saved_path}")

    wdf = load_windows_table(windows_saved_path)
    fcfg = cfg["features"]
    featurizer = WindowFeaturizer(
        numeric_fields=fcfg["numeric_fields"],
        categorical_fields=fcfg["categorical_fields"],
        text_fields=fcfg["text_fields"],
        hash_dim_cat=fcfg["hash_dim_cat"],
        hash_dim_text=fcfg["hash_dim_text"],
    )

    X, y_arr, meta = featurizer.fit_transform(wdf)
    data_dir = ROOT / "data"
    ckpt_dir = ROOT / "checkpoints"
    ensure_dir(str(data_dir))
    ensure_dir(str(ckpt_dir))

    np.save(str(data_dir / "X.npy"), X)
    np.save(str(data_dir / "y.npy"), y_arr)
    save_json(
        {
            "feature_dim": meta.feature_dim,
            "field_slices": {k: [v[0], v[1]] for k, v in meta.field_slices.items()},
            "fields_order": meta.fields_order,
        },
        str(data_dir / "feature_meta.json"),
    )

    pos = int(y_arr.sum())
    neg = int(len(y_arr) - pos)
    print(f"[OK] X,y saved -> {data_dir / 'X.npy'}, {data_dir / 'y.npy'} (T={len(X)}, D={X.shape[1]})")
    print(f"[STATS] y: pos={pos}, neg={neg}, pos_rate={pos / max(len(y_arr), 1):.4f}")

    clcfg = cfg["clfs"]
    device = clcfg.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA不可用，CL-FS 自动切换到 CPU")
        device = "cpu"

    bundle = train_clfs(
        X,
        embed_dim=int(clcfg["embed_dim"]),
        proj_dim=int(clcfg["proj_dim"]),
        lr=float(clcfg["lr"]),
        batch_size=int(clcfg["batch_size"]),
        epochs=int(clcfg["epochs"]),
        temperature=float(clcfg["temperature"]),
        device=device,
    )

    torch.save(bundle.encoder.state_dict(), str(ckpt_dir / "encoder.pt"))
    print(f"[OK] encoder saved -> {ckpt_dir / 'encoder.pt'}")

    Z = encode_windows(bundle.encoder, X, device=device)
    np.save(str(data_dir / "Z.npy"), Z)

    cpcfg = cfg["candidate_pool"]
    raw_scores = compute_field_sensitivity(
        bundle.encoder,
        X,
        meta,
        max_windows=int(cpcfg["sensitivity_max_windows"]),
        device=device,
    )
    save_json(raw_scores, str(data_dir / "field_sensitivity.json"))

    utility_scores = compute_field_utility_scores(
        raw_scores=raw_scores,
        field_slices=meta.field_slices,
        score_norm=str(cpcfg.get("score_norm", "none")),
    )
    save_json(utility_scores, str(data_dir / "field_utility.json"))

    cand = build_candidate_pool(
        scores=raw_scores,
        field_slices=meta.field_slices,
        top_ratio=float(cpcfg.get("top_ratio", 0.1)),
        min_fields=int(cpcfg.get("min_fields", 5)),
        max_candidate_dim=int(cpcfg.get("max_candidate_dim", 0)) if cpcfg.get("max_candidate_dim", None) is not None else None,
        score_norm=str(cpcfg.get("score_norm", "none")),
        min_raw_score_ratio=float(cpcfg.get("min_raw_score_ratio", 0.0)),
    )

    save_json({"candidate_fields": cand}, str(data_dir / "candidate_fields.json"))
    print(f"[OK] candidate pool size={len(cand)} saved -> {data_dir / 'candidate_fields.json'}")

    candidate_dim = 0
    field_dims = {}
    for f in cand:
        s, e = meta.field_slices[f]
        field_dims[f] = int(e - s)
        candidate_dim += (e - s)

    print(f"[OK] candidate fields={cand}")
    print(f"[OK] candidate_dim={candidate_dim} / full_dim={X.shape[1]} => compression={1 - candidate_dim / X.shape[1]:.4f}")

    save_json(
        {
            "windows_path": windows_saved_path,
            "num_windows": int(len(X)),
            "feature_dim": int(X.shape[1]),
            "embed_dim": int(clcfg["embed_dim"]),
            "candidate_fields": cand,
            "candidate_field_dims": field_dims,
            "candidate_field_count": int(len(cand)),
            "candidate_dim": int(candidate_dim),
            "compression_rate_clfs": float(1.0 - candidate_dim / max(int(X.shape[1]), 1)),
            "y_pos": pos,
            "y_neg": neg,
            "y_pos_rate": float(pos / max(len(y_arr), 1)),
        },
        str(data_dir / "clfs_summary.json"),
    )
    print(f"[OK] summary saved -> {data_dir / 'clfs_summary.json'}")


if __name__ == "__main__":
    main()