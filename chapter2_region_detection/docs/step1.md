# Step 1 — Baseline Scaffold 审核记录

> 项目：UG-CEM-APT  
> 分支：`ug-cem-apt`  
> 阶段：Step 1 — baseline scaffold  
> 审核日期：2026-09-16  
> 最终状态：**PASS / 历史脚手架已完成；旧配置不得直接用于 v2.1 正式实验**

---

## 1. 本阶段目标

Step 1 只建立 UG-CEM baseline 的代码目录和比较配置入口，为后续 Step 2/3 独立实现 Categorical CEM 与 UG uncertainty 提供隔离空间。

本阶段不要求配置已经满足后续 v2.1 四动作正式协议。

## 2. 对应提交

首次 baseline scaffold 提交：

`86ac5b6abae5276944ab7a42590a2e654bc37224`

提交说明：

`chore(ug-cem): add baseline scaffold and comparison config`

新增文件：

```text
chapter2_region_detection/baselines/__init__.py
chapter2_region_detection/baselines/ug_cem_apt/__init__.py
chapter2_region_detection/configs/compare_ug_cem_local_online.yaml
```

没有在该提交中修改稳定主方法实现。

## 3. Scaffold 结果

建立了：

```text
baselines/
└── ug_cem_apt/
```

后续 Step 2 / Step 3 的 `categorical_cem.py` 与 `uncertainty.py` 均沿用这一隔离目录。

同时新增统一比较配置入口：

`configs/compare_ug_cem_local_online.yaml`

## 4. v2.1 状态说明

Step 1 的历史 scaffold 本身仍然有效，但它最初创建的 YAML 属于旧协议，目前存在：

- `world_model.state_dim: 8`
- `world_model.action_dim: 5`
- `ug_cem.n_actions: 5`
- 旧 `costs / delays`
- 旧 `lambda_cost / lambda_delay`

这些字段与当前 v2.1 正式协议不一致。

> Step 1 可以保持 PASS，但 `compare_ug_cem_local_online.yaml` 只能视为历史/开发配置，不能作为正式 v2.1 Ours vs UG-CEM vs CEM 实验配置。

正式四动作配置冻结属于 Gate A 的 A4.6b，而不是回头修改 Step 1 的历史验收结论。

## 5. 验收

- baseline package scaffold：PASS；
- baseline 与主方法目录隔离：PASS；
- comparison config 入口建立：PASS；
- 未提前耦合 WM/PPO/LLM 实现：PASS；
- v2.1 正式配置：**不属于本阶段；待 A4.6b 冻结**。

## 6. 最终审核结论

```text
Step 1 — Baseline Scaffold

Directory scaffold : PASS
Baseline isolation : PASS
Config entry       : PASS
v2.1 formal config : DEFERRED TO A4.6b

FINAL STATUS: PASS
```

下一阶段为 Step 2 — Categorical CEM。