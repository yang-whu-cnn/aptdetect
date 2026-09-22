from scipy.spatial.distance import cdist
import mbrl.env.cartpole_continuous as cartpole_env
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


def run(plot_cartpole_uncertainty_heatmap, rollout_from_threshold):
    seed = 0
    env = cartpole_env.CartPoleEnv()
    theta_threshold = -0.5 * env.theta_threshold_radians
    reward_fn = reward_fns.cartpole
    term_fn = termination_fns.cartpole
    device = 'cpu'
    env.seed(seed)
    rng = np.random.default_rng(seed=seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
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
            "trial_length": 200,
            "num_steps": 10000,  # size of replay buffer
            "model_batch_size": 32,
            "validation_ratio": 0.05
        },
        "device": device
    }
    cfg = OmegaConf.create(cfg_dict)
    dynamics_model = common_util.create_one_dim_tr_model(cfg, obs_shape, act_shape)
    model_env = models.ModelEnv(env, dynamics_model, term_fn, reward_fn)
    model_trainer = models.ModelTrainer(dynamics_model, optim_lr=1e-3, weight_decay=5e-5)

    replay_buffer = common_util.create_replay_buffer(cfg, obs_shape, act_shape, rng=rng)

    def cartpole_safe(obs, action, next_obs, reward, done):
        return obs[2] > theta_threshold and next_obs[2] > theta_threshold

    common_util.rollout_agent_trajectories(
        env,
        replay_buffer.capacity,
        planning.RandomAgent(env),
        {},
        trial_length=200,
        replay_buffer=replay_buffer,
        condition=cartpole_safe,
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
            "beta": 0.1
        }
    })

    agent = planning.create_trajectory_optim_agent_for_model(model_env, cem_agent_cfg,
                                                             num_particles=12)

    # to train obs and std normalisers
    agent.optimizer.keep_last_solution = False
    for _ in range(100):
        agent.act(env.reset())
    agent.optimizer.keep_last_solution = True

    if plot_cartpole_uncertainty_heatmap:
        xs = np.linspace(-0.2, 0.2, 41)
        thetas = np.linspace(-0.2, 0.2, 41)
        next_obs_std = np.zeros((len(xs), len(thetas)))
        for i, x in enumerate(xs):
            for j, theta in enumerate(thetas):
                obs = np.array([x, theta])
                if theta < theta_threshold:
                    obs *= -1
                nearest_obs = replay_buffer.obs[cdist(replay_buffer.obs[:, (0, 2)], obs.reshape(1, -1))
                                                    .argmin(), :]
                obs = np.insert(obs, [1, 2], nearest_obs.reshape(1, -1)[:, (1, 3)].flatten())
                if theta < theta_threshold:
                    obs *= -1
                agent.act(obs)
                next_obs_std[i, j] = agent.optimizer.optimizer.next_obs_std.mean().item()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4),
                                       gridspec_kw={"width_ratios": [1, 1.25]})
        ax1.scatter(replay_buffer.obs[:, 0], replay_buffer.obs[:, 2], s=1, alpha=0.5)
        ax1.plot([-0.2, 0.2], [-env.theta_threshold_radians / 2, -env.theta_threshold_radians / 2],
                 c='k')
        ax1.set_ylim(-env.theta_threshold_radians, env.theta_threshold_radians)
        ax1.set_xlim(-0.20, 0.20)
        ax1.set_ylim(-0.20, 0.20)
        ax1.set_xlabel("Cart position")
        ax1.set_ylabel("Pole angle")
        contourf = ax2.contourf(xs, thetas, next_obs_std, cmap="Reds")
        fig.colorbar(contourf)
        ax2.plot([-0.2, 0.2], [-env.theta_threshold_radians / 2, -env.theta_threshold_radians / 2],
                 c='k')
        ax2.set_xlim(-0.2, 0.2)
        ax2.set_ylim(-0.2, 0.2)
        ax2.set_xlabel("Cart position")
        fig.show()

    # change line 177 on mbrl/planning/trajectory_opt
    if rollout_from_threshold:
        agent.optimizer.keep_last_solution = False
        obs = np.array([0, 0, -env.theta_threshold_radians/2, 0])
        agent.act(obs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--heatmap", type=bool, default=False)
    parser.add_argument("--rollout", type=bool, default=False)
    args = parser.parse_args()
    run(args.heatmap, args.rollout)
