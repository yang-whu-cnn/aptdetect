import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import numpy as np
import torch

torch.set_num_threads(1)
from tqdm import tqdm

from src.utils import load_yaml, load_json, set_seed, ensure_dir, save_json
from src.detector import train_logreg_detector
from src.env_mask import PRFSEnv, PRFSEnvConfig
from src.ppo_agent import PPOAgent, PPOConfig, ActorNet


def map_fields_to_cols(feature_meta: dict, candidate_fields: list) -> np.ndarray:
    slices = {k: tuple(v) for k, v in feature_meta["field_slices"].items()}
    cols = []
    for f in candidate_fields:
        if f not in slices:
            continue
        s, e = slices[f]
        cols.extend(list(range(s, e)))
    return np.array(sorted(set(cols)), dtype=np.int64)


def replay_keep_dims(XF_eval, detector, actor, k_flip, tau, device):
    """
    用当前 actor 在给定数据上做一次确定性回放，
    只统计 kept dimensions，用于选 best checkpoint。
    """
    T, D = XF_eval.shape
    mask_prob = np.ones(D, dtype=np.float32)
    mask_hard = np.ones(D, dtype=np.float32)
    keep_dims = []

    for t in range(T):
        x_now = XF_eval[t] * mask_hard
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

    keep_dims = np.array(keep_dims, dtype=np.int64)
    return {
        "mean": float(np.mean(keep_dims)),
        "min": int(np.min(keep_dims)),
        "max": int(np.max(keep_dims)),
        "p50": float(np.percentile(keep_dims, 50)),
        "p90": float(np.percentile(keep_dims, 90)),
    }


def model_select_score(val_stats: dict) -> float:
    """
    越小越好：
    - 优先更小平均保留维度
    - 次优先更小 p90，避免“平均看着不大但高位又反弹”
    """
    return float(val_stats["mean"]) + 0.15 * float(val_stats["p90"])


def main(cfg_path: str = "configs/cc4.yaml"):
    set_seed(42)
    cfg_path = (ROOT / cfg_path).resolve()
    cfg = load_yaml(str(cfg_path))

    data_dir = (ROOT / "data").resolve()
    ckpt_dir = (ROOT / "checkpoints").resolve()
    ensure_dir(str(ckpt_dir))

    X = np.load(str(data_dir / "X.npy")).astype(np.float32)
    y = np.load(str(data_dir / "y.npy")).astype(np.int64)
    meta = load_json(str(data_dir / "feature_meta.json"))
    candidate_fields = load_json(str(data_dir / "candidate_fields.json"))["candidate_fields"]

    device = cfg.get("clfs", {}).get("device", "cpu")
    if device == "cuda" and (not torch.cuda.is_available()):
        print("[WARN] CUDA不可用，PRF-S 自动切换到 CPU")
        device = "cpu"

    cand_cols = map_fields_to_cols(meta, candidate_fields)
    if cand_cols.size == 0:
        raise RuntimeError("[ERROR] cand_cols 为空：candidate_fields 与 feature_meta.json 不匹配")

    X_F = X[:, cand_cols]
    print(f"[OK] Candidate feature matrix: X_F shape={X_F.shape} (fields={len(candidate_fields)}, cols={len(cand_cols)})")

    # 训练/验证切分：这里只用于选 best checkpoint，不作为正文代理指标
    prcfg = cfg["prfs"]
    val_ratio = float(prcfg.get("val_ratio", 0.15))
    n = len(X_F)
    val_n = max(256, int(n * val_ratio))
    train_n = n - val_n
    X_F_train, X_F_val = X_F[:train_n], X_F[train_n:]
    y_train, y_val = y[:train_n], y[train_n:]

    det_cfg = cfg["detector"]
    detector = train_logreg_detector(
        X_F_train,
        y_train,
        threshold=float(det_cfg.get("threshold", 0.5)),
        class_weight=det_cfg.get("class_weight", None),
        C=float(det_cfg.get("C", 1.0)),
        max_iter=int(det_cfg.get("max_iter", 1000)),
    )
    print("[OK] detector trained")

    dcfg = cfg["data"]
    rwcfg = cfg["reward"]
    early_steps = max(1, int(dcfg["early_warning_horizon_sec"] // dcfg["window_size_sec"]))
    hour_steps = max(1, int(rwcfg["hour_window_sec"] // dcfg["window_size_sec"]))

    env_cfg = PRFSEnvConfig(
        window_size_sec=int(dcfg["window_size_sec"]),
        k_flip=int(prcfg["k_flip"]),
        tau_soft_update=float(prcfg["tau_soft_update"]),
        early_horizon_steps=early_steps,
        lambda_sparse=float(rwcfg["lambda_sparse"]),
        target_keep_ratio=float(rwcfg.get("target_keep_ratio", 0.5)),
        lambda_fp=float(rwcfg["lambda_fp"]),
        lambda_lag=float(rwcfg["lambda_lag"]),
        hour_window_steps=hour_steps,
    )

    env = PRFSEnv(X_F_train, y_train, detector, env_cfg)
    print("[OK] PRFS env ready")

    state_dim = 1 + X_F.shape[1]
    action_dim = X_F.shape[1]
    ppo_cfg = PPOConfig(
        gamma=float(prcfg["gamma"]),
        gae_lambda=float(prcfg["gae_lambda"]),
        clip_eps=float(prcfg["clip_eps"]),
        actor_lr=float(prcfg["actor_lr"]),
        critic_lr=float(prcfg["critic_lr"]),
        ppo_epochs=int(prcfg["ppo_epochs"]),
        device=device,
        k_flip=min(int(prcfg["k_flip"]), int(X_F.shape[1])),
    )

    agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, cfg=ppo_cfg)
    print("[OK] PPO agent ready")

    rollout_len = int(prcfg["rollout_len"])
    iters = int(prcfg.get("iters", 10))
    s = env.reset()
    history = []

    best_score = float("inf")
    best_iter = -1

    best_actor_path = ckpt_dir / "ppo_actor.pt"
    best_critic_path = ckpt_dir / "ppo_critic.pt"
    last_actor_path = ckpt_dir / "ppo_actor_last.pt"
    last_critic_path = ckpt_dir / "ppo_critic_last.pt"

    for it in range(iters):
        traj = {"states": [], "actions": [], "old_logp": [], "rewards": [], "dones": []}
        infos = []

        for _ in tqdm(range(rollout_len), desc=f"PPO rollout iter {it + 1}"):
            a_idx, logp = agent.act(s)
            s2, r, done, info = env.step(a_idx)
            traj["states"].append(s)
            traj["actions"].append(a_idx)
            traj["old_logp"].append(logp)
            traj["rewards"].append(r)
            traj["dones"].append(float(done))
            infos.append(info)
            s = s2
            if done:
                s = env.reset()

        traj["states"] = np.stack(traj["states"], axis=0).astype(np.float32)
        traj["actions"] = np.stack(traj["actions"], axis=0).astype(np.int64)
        traj["old_logp"] = np.array(traj["old_logp"], dtype=np.float32)
        traj["rewards"] = np.array(traj["rewards"], dtype=np.float32)
        traj["dones"] = np.array(traj["dones"], dtype=np.float32)

        flat = traj["actions"].reshape(-1).tolist()
        cnt = Counter(flat)
        print(f"[DEBUG Iter {it + 1}] unique_actions={len(cnt)} top10={cnt.most_common(10)}")

        out = agent.update(traj)

        def safe_mean(key, default=0.0):
            vals = [x.get(key, default) for x in infos]
            return float(np.mean(vals)) if len(vals) > 0 else float(default)

        row = {
            "iter": int(it + 1),
            "loss_actor": float(out.get("loss_actor", 0.0)),
            "loss_critic": float(out.get("loss_critic", 0.0)),
            "delta_auprc": safe_mean("delta_auprc", 0.0),
            "fp_per_hour": safe_mean("fp_per_hour", 0.0),
            "keep_ratio_train": safe_mean("sparsity", 1.0),
            "kept_dim_train": safe_mean("kept_dim", float(X_F_train.shape[1])),
        }

        # 用验证集真正回放一次，作为 best checkpoint 选择依据
        val_stats = replay_keep_dims(
            X_F_val,
            detector,
            agent.actor,
            k_flip=int(prcfg["k_flip"]),
            tau=float(prcfg["tau_soft_update"]),
            device=device,
        )
        row["val_keep_mean"] = float(val_stats["mean"])
        row["val_keep_p90"] = float(val_stats["p90"])
        row["model_select_score"] = float(model_select_score(val_stats))
        history.append(row)

        print(
            f"[Iter {it + 1}] lossA={row['loss_actor']:.3f} "
            f"lossC={row['loss_critic']:.3f} "
            f"train_keep={row['kept_dim_train']:.1f} "
            f"val_keep_mean={row['val_keep_mean']:.1f} "
            f"val_keep_p90={row['val_keep_p90']:.1f} "
            f"score={row['model_select_score']:.2f}"
        )

        if row["model_select_score"] < best_score:
            best_score = row["model_select_score"]
            best_iter = int(it + 1)
            torch.save(agent.actor.state_dict(), str(best_actor_path))
            torch.save(agent.critic.state_dict(), str(best_critic_path))
            print(f"[BEST] iter={best_iter}, score={best_score:.2f} -> saved best checkpoint")

    torch.save(agent.actor.state_dict(), str(last_actor_path))
    torch.save(agent.critic.state_dict(), str(last_critic_path))

    save_json(
        {
            "history": history,
            "best_iter": int(best_iter),
            "best_score": float(best_score),
            "best_actor_path": str(best_actor_path),
            "best_critic_path": str(best_critic_path),
            "last_actor_path": str(last_actor_path),
            "last_critic_path": str(last_critic_path),
        },
        str(data_dir / "prfs_history.json"),
    )

    print(f"[OK] best PPO saved -> {best_actor_path} / {best_critic_path}")
    print(f"[OK] last PPO saved -> {last_actor_path} / {last_critic_path}")
    print(f"[OK] history saved -> {data_dir / 'prfs_history.json'}")


if __name__ == "__main__":
    main()