# Uncertainty Guided CEM Planning

Accompanying repository for Risk Sensitive Model-Based Reinforcement Learning using Uncertainty
Guided Planning, accepted at the NeurIPS 2021 Safe and Robust Control of Uncertain Systems Workshop.

To generate the cartpole data distribution and heatmap plot:
```bash
python cartpole_experiments --heatmap True
```

To plot the cartpole planning distribution plot from the training data threshold:
```bash
python cartpole_experiments --rollout True
```

To run complete episodes and generate return and cost plots:
```bash
python offline_episodes.py --env "cartpole" --num_trials 10 --betas "0, 0.2, 0.3" --num_seeds 3
```