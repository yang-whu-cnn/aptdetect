# Step A4.4 — Bootstrap Probabilistic Ensemble World Model 审核记录

日期：2026-09-16  
状态：PASS

## Git 范围

实现提交：

`19810193d4155e5f8cfa2244b6ae204c417fee04`

相对 A4.4 contract 冻结点仅新增：

- `chapter2_region_detection/formal_experiments/training/bootstrap_world_model.py`
- `chapter2_region_detection/tests/test_gate_a_bootstrap_world_model.py`

未修改 FormalState、adapter、resolver、incident reward、CEM、UG uncertainty 或 PPO。

## 源码审核

正式 dynamics ensemble 已实现：

- formal state_dim=27；
- formal n_actions=4；
- ensemble_size=5；
- 两层 MLP，hidden=128；
- diagonal Gaussian mean/log-variance output；
- 每个 member 独立 model seed；
- 每个 member 独立 bootstrap seed；
- 每个 member 独立 optimizer；
- bootstrap 为有放回采样；
- train-only state/next-state normalizer；
- state std 下限 eps；
- dynamics dataset 学习 executed action，而非 fallback 前 requested action；
- terminal mid-action / action_completed=False 默认不进入标准 dynamics training；
- absolute target 为默认；
- delta target 作为 A4.5 validation 备选；
- 输出 epistemic / aleatoric / total variance；
- checkpoint 保存 config、normalizer、model/bootstrap seed 与 member weights；
- 提供 fixed-member vectorized tensor prediction 接口，供后续 H-step rollout 复用。

## 测试

本文件实际包含 16 个测试，不是最初说明中的 15 个；多出的第 16 个是 invalid target_mode validation，因此用户实际输出：

- A4.4：16 tests，OK；
- 完整 Gate A：126 tests，OK。

该数量与当前源码一致，无异常。

## 论文与任务书一致性

当前小论文要求：对每个候选多步计划，由 ensemble probabilistic dynamics models 递归预测下一状态，并据此估计 cumulative reward 与 predictive uncertainty。当前 A4.4 实现正是该 dynamics 基础。

Bootstrap 属于 ensemble 的训练实现细节，不改变论文方法定义；Ours / UG-CEM / CEM 将共享同一 world-model checkpoint bundle。

A4.5 仍需在 held-out validation 上比较 absolute vs delta，并验证 one-step / H=2 / H=4 error 与 uncertainty-error calibration。只有通过质量 Gate 后，WM 才可进入正式 planner comparison。

FINAL STATUS: PASS
