# experiments/collect_local_online.py
import os, json, argparse
import yaml
from src.cc4_client import OnlineCC4Client


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/cc4_local_online.yaml")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--out", default="data/replays/local_online.jsonl")
    args = ap.parse_args()

    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    client = OnlineCC4Client(
        base_url=cfg["online"]["base_url"],
        api_key=cfg["online"].get("api_key", ""),
        timeout_sec=cfg["online"].get("timeout_sec", 30),
        max_steps_local=cfg["train"].get("max_episode_steps", 100),
        seed_local=cfg.get("seed", 0),
    )

    n_actions = cfg.get("world_model", {}).get("action_dim", 5)

    with open(args.out, "w", encoding="utf-8") as wf:
        for ep in range(args.episodes):
            obs = client.reset(seed=cfg.get("seed", 0) + ep)
            done = False
            step = 0
            while not done:
                # 先用随机动作（后面接入第二章策略时再替换这里）
                action = step % n_actions
                next_obs, reward, done, info = client.step(action)

                row = {
                    "episode": ep,
                    "step": step,
                    "obs": obs,
                    "action": action,
                    "reward": float(reward),
                    "done": bool(done),
                    "info": info,
                    "next_obs": next_obs,
                }
                wf.write(json.dumps(row, ensure_ascii=False) + "\n")

                obs = next_obs
                step += 1

    print("[DONE] wrote:", args.out)


if __name__ == "__main__":
    main()
