# experiments/run_train_posterior.py
# 作用：串起第二章闭环：LLM先验 -> 世界模型评估 -> PPO后验选择（单区域）
# 本版新增：LLM先验缓存与降频调用（refresh_every），显著缩短训练时间
import sys
from pathlib import Path
import argparse
import time
import numpy as np
from tqdm import trange

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.utils import ensure_dir, load_yaml, set_seed
from src.env_region import build_region_objects, build_evidence
from src.metrics import episode_stats, plan_rank_stats


def load_world_model(objs, path: Path):
    import torch
    sd_list = torch.load(path, map_location=objs.world_model.device)
    for m, sd in zip(objs.world_model.models, sd_list):
        m.load_state_dict(sd)
        m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/cc4.yaml")
    ap.add_argument("--region_id", type=int, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--use_trained_world_model", action="store_true")
    # 可选覆盖：不传就读 cfg.llm_prior.refresh_every（默认1）
    ap.add_argument("--llm_refresh_every", type=int, default=None,
                    help="每N步调用一次LLM，其余步复用上次先验；不传则读取cfg.llm_prior.refresh_every")
    args = ap.parse_args()

    cfg = load_yaml(str((ROOT / args.cfg).resolve()))
    set_seed(cfg.get("seed", 42))

    objs = build_region_objects(cfg, region_id=args.region_id, device=args.device)

    ckpt_dir = (ROOT / cfg["paths"]["ckpt_dir"]).resolve()
    ensure_dir(str(ckpt_dir))

    wm_path = ckpt_dir / f"world_model_region{args.region_id}.pt"
    if args.use_trained_world_model and wm_path.exists():
        load_world_model(objs, wm_path)
        print("[OK] loaded world model:", wm_path)
    else:
        print("[WARN] world model not loaded (mock模式下仍可跑通流程；正式训练建议先训world model)。")

    ppo_steps = int(cfg["ppo"]["train_steps"])
    rollout_len = int(cfg["ppo"]["rollout_len"])
    eval_every = int(cfg["train"]["eval_every"])
    N = int(cfg["llm_prior"]["k_candidates"])

    # 读取 LLM 降频调用间隔（默认1=每步都调）
    if args.llm_refresh_every is not None:
        llm_refresh_every = max(1, int(args.llm_refresh_every))
    else:
        llm_refresh_every = max(1, int(cfg.get("llm_prior", {}).get("refresh_every", 1)))

    print(f"[INFO] llm_refresh_every={llm_refresh_every} "
          f"(每{llm_refresh_every}步调用一次LLM，其余步复用上次先验)")

    obs = objs.client.reset()
    objs.summarizer.reset()

    rewards_hist, infos_hist = [], []
    plan_rank_hist = []

    def empty_traj():
        return {
            "prior_logits": np.zeros((rollout_len, N), dtype=np.float32),
            "E": np.zeros((rollout_len, N, 4), dtype=np.float32),
            "actions": np.zeros((rollout_len,), dtype=np.int64),
            "logp_old": np.zeros((rollout_len,), dtype=np.float32),
            "rewards": np.zeros((rollout_len,), dtype=np.float32),
            "dones": np.zeros((rollout_len,), dtype=np.float32),
        }

    traj = empty_traj()
    t_in_roll = 0

    # LLM 先验缓存（用于降频调用）
    cached_prior_out = None
    llm_call_count = 0
    llm_time_total = 0.0
    just_reset = True  # 新episode开始时强制刷新一次LLM

    for t in trange(ppo_steps, desc=f"PPO train region {args.region_id}"):
        objs.summarizer.update(obs.get("events", []))
        summary = objs.summarizer.build(risk_proxy=float(obs.get("risk_proxy", 0.0)))

        # 1) LLM先验：候选计划/评分/检查点（支持降频缓存）
        need_refresh = (cached_prior_out is None) or (t % llm_refresh_every == 0) or just_reset
        if need_refresh:
            t0 = time.perf_counter()
            prior_out = objs.llm_prior.generate(summary)
            llm_time_total += (time.perf_counter() - t0)
            llm_call_count += 1
            cached_prior_out = prior_out
            just_reset = False
        else:
            prior_out = cached_prior_out

        # 2) 世界模型前瞻评估：证据向量 E
        prior_logits, E, plans, plan_ranks = build_evidence(cfg, objs.world_model, summary, prior_out)

        # 3) PPO后验选择：选择候选索引，并执行首步动作
        idx, logp, _ = objs.ppo.act(prior_logits=prior_logits, E=E, deterministic=False)
        action_id = int(plans[idx][0])

        obs2, r, done, info = objs.client.step(action_id)

        rewards_hist.append(float(r))
        infos_hist.append(info)
        plan_rank_hist.append(int(plan_ranks[idx]))

        traj["prior_logits"][t_in_roll] = prior_logits
        traj["E"][t_in_roll] = E
        traj["actions"][t_in_roll] = idx
        traj["logp_old"][t_in_roll] = logp
        traj["rewards"][t_in_roll] = float(r)
        traj["dones"][t_in_roll] = float(done)

        t_in_roll += 1
        obs = obs2

        if done:
            obs = objs.client.reset()
            objs.summarizer.reset()
            just_reset = True  # 新episode首步强制刷新LLM

        if t_in_roll >= rollout_len:
            objs.ppo.update(traj)
            traj = empty_traj()
            t_in_roll = 0

        if (t + 1) % eval_every == 0:
            st = episode_stats(rewards_hist, infos_hist)
            llm_avg = (llm_time_total / llm_call_count) if llm_call_count > 0 else 0.0

            rank_counts = np.bincount(plan_rank_hist, minlength=N) if plan_rank_hist else np.zeros(N)
            rank_total = len(plan_rank_hist) or 1
            rank_probs = [f"{(rank_counts[i] / rank_total) * 100:.1f}%" for i in range(N)]

            rank_stats = plan_rank_stats(plan_rank_hist, N)

            print("\n[STAT]", st,
                  "| llm_calls:", llm_call_count,
                  "| llm_avg_sec:", round(llm_avg, 3),
                  "| plan_rank_dist:", rank_probs,
                  "| top1_ratio:", f"{rank_stats['top1_ratio'] * 100:.1f}%",
                  "| avg_rank:", f"{rank_stats['avg_rank']:.2f}")

    # 如果末尾有未满 rollout 的数据，这里不强制 update（避免改动现有 PPO.update 接口约定）
    if t_in_roll > 0:
        print(f"[INFO] leftover rollout not updated: {t_in_roll}/{rollout_len} "
              f"(快速测试可忽略；正式训练可设置train_steps为rollout_len的整数倍)")

    out = ckpt_dir / f"ppo_posterior_region{args.region_id}.pt"
    objs.ppo.save(str(out))
    print("[OK] saved:", out)


if __name__ == "__main__":
    main()