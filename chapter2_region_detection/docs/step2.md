# Step 2 — Categorical CEM 审核记录

> 项目：UG-CEM-APT  
> 分支：`ug-cem-apt`  
> 阶段：Step 2 — Categorical CEM  
> 审核日期：2026-09-15  
> 最终状态：**PASS / 已完成，可进入 Step 3**

---

## 1. 本阶段目标

Step 2 只实现一次离散 Categorical CEM 搜索，不接入 APT 环境语义。

本阶段固定的抽象量为：

```text
H = planning horizon
A = n_actions = 5
N = population size
```

本阶段明确不负责：

- world model；
- UG uncertainty；
- CC4 / CybORG；
- LLM；
- PPO；
- reward；
- 跨真实环境时间步的 MPC 概率左移。

因此，CEM 只操作动作类别 ID，不知道 `analyse / remove / restore / control_traffic / no_op` 的具体语义，也不包含旧动作名称、cost、delay 或 reward。

---

## 2. 实现文件

```text
chapter2_region_detection/baselines/ug_cem_apt/categorical_cem.py
chapter2_region_detection/tests/test_categorical_cem.py
```

核心接口：

```python
optimize(objective_fn, initial_probs=None)
```

输出：

```text
CEMResult
├── best_plan      [H], torch.long
├── best_score     scalar
└── final_probs    [H,A], torch.float
```

候选计划：

```text
plans  : [N,H], torch.long
scores : [N],   torch.float
probs  : [H,A], torch.float
```

---

## 3. 与 UG 官方 CEM 的一致性检查

已重新对照官方：

```text
mbrl-lib-uncertainty_guided_planning/
└── mbrl/planning/trajectory_opt.py
```

### 3.1 Elite 数量

官方使用：

```text
elite_num = ceil(population_size * elite_ratio)
```

当前实现一致。

### 3.2 Alpha 更新方向

官方连续 CEM：

```text
new_mu = mean(elite)
mu = alpha * old_mu + (1 - alpha) * new_mu
```

离散领域适配：

```text
p_new
=
alpha * p_old
+
(1 - alpha) * p_elite_frequency
```

当前实现方向正确，没有把 alpha 写反。

### 3.3 Best sampled plan

官方 CEM 支持返回全局最佳已评估样本；原实验还存在连续空间 elite mean 的配置。

由于本文动作 ID 是离散类别，动作编号不能做有意义的算术平均，因此 UG-CEM-APT 主实现采用：

> 跨全部 CEM iteration 的最高分、且真实被 objective 评估过的 sampled plan。

这是离散动作空间的有意适配。

### 3.4 Warm-start 职责分离

官方 `TrajectoryOptimizer` 负责跨真实时间步保存和左移 previous solution。

当前实现保持相同职责分离：

- `CategoricalCEMOptimizer` 只完成一次搜索；
- 允许显式传入 `initial_probs`；
- 不把上一次 `final_probs` 静默作为下一次默认起点；
- 后续由 Step 5 `planner.py` 实现概率矩阵左移。

---

## 4. Step 2 已确认的实现项

审核确认以下功能均已实现：

- Torch-first categorical sampling；
- `plans.shape == [N,H]`；
- plan dtype 为 `torch.long`；
- 动作 ID 范围为 `[0, A-1]`；
- `elite_num = ceil(N * elite_ratio)`；
- elite action frequency；
- 官方一致的 alpha 更新语义；
- optional `initial_probs`；
- uniform default initialization；
- 不隐式继承上一轮 `final_probs`；
- seed reproducibility；
- objective 显式接收 zero-based `iteration`；
- 跨所有 iteration 跟踪 best sampled plan；
- 返回 `final_probs`；
- probability / shape / finite-value 防御；
- 与 world model、PPO、LLM、CC4、reward 解耦。

---

## 5. Probability floor

当前实现使用幂等 lower-bound projection：

```text
p = normalize(p)
residual = max(p - floor, 0)

q =
floor
+
(1 - A*floor)
* residual / sum(residual)
```

已确认满足：

- 每个动作概率不低于 floor；
- 每行概率和为 1；
- 合法分布重复应用 projection 不会持续向 uniform 被“抹平”；
- 验证 `A * floor < 1`。

这为 Step 5 的 MPC warm-start 保留了正确概率语义。

---

## 6. 非有限 objective score 防御

初次审核发现一个边界问题：

当 finite candidate 数量少于 `elite_num` 时，仅把 NaN/Inf 映射为 `-inf` 仍可能使无效候选进入 `topk` elite。

修复后逻辑为：

```text
finite_count == 0
    -> raise

0 < finite_count < elite_num
    -> raise

finite_count >= elite_num
    -> 非有限值映射为 -inf
    -> 继续 elite selection
```

新增对应单元测试，确保无效候选不会被强行纳入 elite。

---

## 7. 单元测试覆盖

当前测试文件覆盖：

1. uniform probability shape / normalization；
2. sampled plan shape / dtype / range；
3. elite count 使用 ceil；
4. elite frequency；
5. probability floor、归一化与幂等性；
6. alpha=0 与 alpha=1 的更新方向；
7. seed reproducibility；
8. synthetic target plan 搜索；
9. custom `initial_probs` 控制首轮采样；
10. optimize 不隐式继承 previous final probabilities；
11. iteration 按 `0..I-1` 传递；
12. 非法 config；
13. 非法 initial probabilities；
14. objective shape 错误；
15. 全部 non-finite scores；
16. finite scores 数量不足以形成 elite set。

人工目标采用类别位置匹配：

```text
target = [4,3,2,1]
score = number_of_positions_equal_to_target
```

没有把动作 ID 当作连续数值距离。

---

## 8. 代码隔离审核

Step 2 审核确认：

```text
categorical_cem.py
    不 import src/world_model
    不 import CC4 / CybORG
    不 import PPO
    不 import LLM
    不绑定 reward
    不绑定动作语义
```

同时：

- 未修改 `chapter2_region_detection/src/`；
- 未修改 UG 官方源码；
- Step 2 实现保持在 `baselines/ug_cem_apt/`；
- 测试保持在 `tests/`。

---

## 9. 是否还需要修改 / 优化

结论：

> **没有阻塞 Step 3 的修改项。Step 2 当前实现可以冻结并进入下一阶段。**

可选但非阻塞的工程增强：

- 将整数型配置（H/A/N/num_iterations）的类型检查再做得更严格；
- 扩展更多非法配置组合的单元测试；
- 统一测试文件的少量缩进风格。

这些都不改变 CEM 算法语义，也不要求在进入 Step 3 前完成。

后续如果没有发现集成级接口问题，不应在 Step 3 中顺手修改 CEM 算法，以避免阶段边界漂移。

---

## 10. 最终审核结论

```text
Step 2 — Categorical CEM

Implementation : PASS
Unit-test design: PASS
Official-source alignment: PASS
Domain adaptation: PASS
Module isolation: PASS
Numerical guards: PASS

FINAL STATUS: PASS
```

Step 2 正式关闭。

下一阶段：

```text
Step 3 — UG Uncertainty
```

Step 3 输入统一采用：

```text
next_states.shape = [H,N,M,D]
```

其目标是单独实现 UG 官方源码风格的 state-trajectory disagreement 与 running normalization；不在 Step 3 中接入 reward、beta penalty、world-model rollout 或 MPC。
