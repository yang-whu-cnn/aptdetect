import mbrl.env.cartpole_continuous as cartpole_env
import mbrl.env.pendulum as pendulum_env
import mbrl.env.reward_fns as reward_fns
import mbrl.env.termination_fns as termination_fns
import mbrl.models as models
import mbrl.planning as planning
import mbrl.util.common as common_util
import numpy as np
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import argparse


def run(env, reward_fn, term_fn, num_seeds, num_trials, max_steps, betas, device):
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    cfg_dict = {
        "dynamics_model": {
            "model": {
                "_target_": "mbrl.models.GaussianMLP",
                "device": device,
                "num_layers": 3,
                "ensemble_size": 4,
                "hid_size": 200,
                "use_silu": True,
                "in_size": "???",
                "out_size": "???",
                "deterministic": False,
                "propagation_method": "fixed_model"
            }
        },
        "algorithm": {
            "learned_rewards": False,
            "target_is_delta": True,
            "normalize": True
        },
        "overrides": {
            "trial_length": max_steps,
            "num_steps": 10000,  # size of replay buffer
            "model_batch_size": 32,
            "validation_ratio": 0.05
        },
        "device": device
    }
    cfg = OmegaConf.create(cfg_dict)

    all_rewards = np.zeros((len(betas), num_seeds, num_trials))
    all_costs = np.zeros((len(betas), num_seeds, num_trials))
    for s, seed in enumerate(np.random.randint(0, 100, num_seeds)):
        seed = int(seed)
        env.seed(seed)
        rng = np.random.default_rng(seed=seed)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        for b, beta in enumerate(betas):

            dynamics_model = common_util.create_one_dim_tr_model(cfg, obs_shape, act_shape)
            model_env = models.ModelEnv(env, dynamics_model, term_fn, reward_fn)
            model_trainer = models.ModelTrainer(dynamics_model, optim_lr=1e-3, weight_decay=5e-5)

            replay_buffer = common_util.create_replay_buffer(cfg, obs_shape, act_shape, rng=rng)

            common_util.rollout_agent_trajectories(
                env,
                replay_buffer.capacity,
                planning.RandomAgent(env),
                {},
                trial_length=max_steps,
                replay_buffer=replay_buffer,
                condition=lambda obs: not env.unsafe_obs(obs),
            )

            dynamics_model.update_normalizer(replay_buffer.get_all())
            dataset_train, dataset_val = common_util.get_basic_buffer_iterators(
                replay_buffer,
                batch_size=cfg.overrides.model_batch_size,
                val_ratio=cfg.overrides.validation_ratio,
                ensemble_size=cfg.dynamics_model.model.ensemble_size,
                shuffle_each_epoch=True,
                bootstrap_permutes=False,
            )
            model_trainer.train(
                dataset_train,
                dataset_val=dataset_val,
                num_epochs=50,
                patience=50)

            cem_agent_cfg = OmegaConf.create({
                "_target_": "mbrl.planning.TrajectoryOptimizerAgent",
                "planning_horizon": 10,
                "replan_freq": 1,
                "verbose": False,
                "action_lb": "???",
                "action_ub": "???",
                "optimizer_cfg": {
                    "_target_": "mbrl.planning.CEMOptimizer",
                    "num_iterations": 5,
                    "elite_ratio": 0.3,
                    "population_size": 200,
                    "alpha": 0.1,
                    "device": "cpu",
                    "return_mean_elites": True,
                    "uncertainty_guided": True,
                    "beta": beta
                }
            })

            agent = planning.create_trajectory_optim_agent_for_model(model_env, cem_agent_cfg,
                                                                     num_particles=12)

            # to train obs and std normalisers
            agent.optimizer.keep_last_solution = False
            for _ in range(100):
                agent.act(env.reset())
            agent.optimizer.keep_last_solution = True

            for trial in range(num_trials):
                obs = env.reset()
                agent.reset()
                done = False
                total_reward = 0.0
                total_cost = 0.0
                steps_trial = 0
                while not done:
                    action = agent.act(obs)
                    next_obs, reward, done, info = env.step(action)
                    obs = next_obs
                    total_reward += reward
                    total_cost += info["cost"]
                    steps_trial += 1
                    if steps_trial == max_steps:
                        break
                all_rewards[b, s, trial] = total_reward
                all_costs[b, s, trial] = total_cost
            print(f"Beta = {beta}, Seed = {seed}, Ave reward = {all_rewards[b, s, :].mean()},"
                  f" Ave cost {all_costs[b, s, :].mean()}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    episodes = np.tile(np.linspace(1, num_trials, num_trials).reshape(-1, 1), (1, len(betas)))
    reward_mean = all_rewards.mean(1).T
    cost_mean = all_costs.mean(1).T
    reward_std = all_rewards.std(1).T
    cost_std = all_costs.std(1).T
    ax1.plot(episodes, reward_mean, label=betas)
    ax2.plot(episodes, cost_mean, label=betas)
    for b in range(len(betas)):
        ax1.fill_between(episodes[:, b], reward_mean[:, b] - reward_std[:, b],
                         reward_mean[:, b] + reward_std[:, b], alpha=0.2)
        ax2.fill_between(episodes[:, b], cost_mean[:, b] - cost_std[:, b],
                         cost_mean[:, b] + cost_std[:, b], alpha=0.2)
    ax1.set_title("Return"), ax2.set_title("Cost")
    ax1.set_xlabel("Episode"), ax2.set_xlabel("Episode")
    ax1.set_ylabel("Average return"), ax2.set_ylabel("Average cost")
    ax2.legend(title=r"$\beta$")
    ax1.set_xlim(1, num_trials), ax2.set_xlim(1, num_trials)
    ax1.set_xticks(np.arange(2, num_trials + 1, 2))
    ax2.set_xticks(np.arange(2, num_trials + 1, 2))
    #ax1.set_ylim(0, 200), ax2.set_ylim(0, 200)
    fig.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str)
    parser.add_argument("--num_seeds", type=int, default=3)
    parser.add_argument("--num_trials", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--betas", type=str, default="0, 0.2, 0.3")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    betas = [float(item) for item in args.betas.split(',')]

    if args.env == "cartpole":
        env = cartpole_env.CartPoleEnv()
        reward_fn = reward_fns.cartpole
        term_fn = termination_fns.cartpole

        unsafe_obs = lambda obs: obs[2] < -0.5 * env.theta_threshold_radians

    elif args.env == "pendulum":
        env = pendulum_env.PendulumEnv()
        reward_fn = reward_fns.inverted_pendulum
        term_fn = termination_fns.inverted_pendulum

        def unsafe_obs(obs):
            theta = np.arccos(obs[0])
            if obs[1] < 0: theta *= -1
            return -3/4 * np.pi < theta < -1/4 * np.pi

    else:
        raise Exception("Only cartpole and pendulum environments implemented.")

    env.unsafe_obs = unsafe_obs

    run(env, reward_fn, term_fn, args.num_seeds, args.num_trials, args.max_steps, betas,
        args.device)
