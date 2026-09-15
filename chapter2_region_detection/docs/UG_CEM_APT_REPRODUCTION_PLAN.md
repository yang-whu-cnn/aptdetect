# UG-CEM-APT 复现与公平对比实施总方案

> 仓库：`yang-whu-cnn/aptdetect`  
> 稳定备份分支：`me`  
> 实验分支：`ug-cem-apt`  
> 当前任务：将论文 **Risk Sensitive Model-Based Reinforcement Learning using Uncertainty Guided Planning** 的核心规划机制适配到本项目离散 APT 防御动作空间，形成 **UG-CEM-APT** 对比基线，并与 LWM-RL 做公平对比。  
> 最后更新：2026-09-15

---

## 0. 本文件的用途

本文件是后续 UG-CEM-APT 复现工作的唯一“总路线图”和检查清单。

以后每次开始下一步之前，必须先做以下事情：

1. 读取本文件；
2. 检查 `me` 与 `ug-cem-apt` 的当前差异；
3. 确认上一阶段是否已经审核通过；
4. 只执行当前阶段，不提前混入后续功能；
5. 本地测试通过后再 push；
6. push 后由 ChatGPT 直接读取 GitHub 分支进行 code review；
7. 审核通过后才进入下一阶段。

本文件用于避免后续长对话中出现方案漂移、参数遗忘、接口不一致和“为了跑通临时改算法”的情况。


## 0.1 方案变更记录（2026-09-15，优先级高于后文旧描述）

从本次更新开始，后续实现与审核必须以以下三条为准：

### 变更 A：高层动作空间重新设计

原先的：
`monitor / light_evidence / heavy_evidence / local_mitigate / strong_mitigate`
不再作为最终论文动作语义。

新的 5 类高层动作语义确定为：

- `analyse`：intrusion investigation（入侵调查）
- `remove`：user-level compromise removal（用户级失陷清除）
- `restore`：host reimaging（主机重装/恢复）
- `control_traffic`：traffic blocking（流量阻断/控制）
- `no_op`：monitor（不采取处置，仅监测）

当前只冻结“动作语义集合”和动作数 `A=5`，**暂不在 Step 2/3 阶段锁死数值 ID 顺序、动作成本和延迟参数**。这些内容在进入世界模型/正式环境集成前通过 Gate A 统一确定。

因此 Step 2 的 Categorical CEM 只依赖 `n_actions=5`，不允许把旧动作名称、旧 cost 或旧强度语义硬编码进优化器。

### 变更 B：删除“提前预警步数”

`early-warning lead steps / 提前预警步数` 不再作为：

- PPO reward 项；
- 最终主实验评价指标；
- 模型选择指标；
- baseline 调参指标。

原因：该目标可能诱导 PPO 通过拖延/不处置获得不期望的策略行为。后续若源码中仍保留旧的 `compute_early_gain`、`lambda1_early` 等兼容代码，在最终训练和评估配置中必须保证它们不参与 reward 和主指标计算。

### 变更 C：PPO reward 暂不冻结

新的 PPO reward 设计目前为 **TBD**。

当前原则：

1. Step 2（Categorical CEM）和 Step 3（UG uncertainty）继续推进，因为它们与 PPO reward 无关；
2. 不为了赶进度临时拍脑袋确定新的 PPO reward；
3. rollout evaluator 和 planner 设计必须允许 reward / plan-return 逻辑可替换，避免将旧 reward 写死在 UG-CEM 内部；
4. 在最终重新训练 PPO、正式公平比较和论文主实验前，必须通过 Gate B 冻结 reward 定义、权重及训练 checkpoint；
5. 旧 PPO checkpoint / 旧 local-online reward 仅用于联调，不自动视为最终论文版本。


---

# 1. 最终目标

最终要得到一个领域适配基线：

```text
UG-CEM-APT
=
共享的 APT 状态表示
+ 共享的集成概率世界模型
+ 离散 Categorical CEM
+ UG 官方源码风格的不确定性惩罚
+ MPC / Receding-Horizon 执行
```

它不使用：

- LLM 候选计划；
- PPO 后验选择；
- 人工启发式动作融合；
- 测试集未来信息或攻击标签。

它与本文 LWM-RL 共享：

- 状态表示；
- 动作空间；
- 世界模型结构及 checkpoint；
- 预测计划回报函数；
- 环境；
- 数据/种子划分；
- 评价指标。

因此正式论文中应称为：

> **UG-CEM-APT（领域适配实现）**

而不是“原论文完全复现”。

---

# 2. 三套代码必须严格隔离

## 2.1 本文稳定代码

```text
chapter2_region_detection/src/
```

原则：**尽量不修改。**

特别是：

```text
src/action_space.py
src/state_summary.py
src/world_model.py
src/plan_eval.py
src/ppo_posterior.py
src/env_region.py
src/cc4_client.py
```

这些代表当前主方法和共享基础设施。

---

## 2.2 UG 官方源码

```text
mbrl-lib-uncertainty_guided_planning/
```

原则：**只读，不修改。**

重点参考：

```text
mbrl/planning/trajectory_opt.py
mbrl/models/model_env.py
mbrl/util/common.py
uncertainty_guided_planning/offline_episodes.py
uncertainty_guided_planning/cartpole_experiments.py
```

---

## 2.3 我们的领域适配代码

统一放在：

```text
chapter2_region_detection/baselines/ug_cem_apt/
```

预计最终结构：

```text
chapter2_region_detection/
├── baselines/
│   ├── __init__.py
│   └── ug_cem_apt/
│       ├── __init__.py
│       ├── categorical_cem.py
│       ├── uncertainty.py
│       ├── rollout_evaluator.py
│       └── planner.py
├── tests/
│   ├── test_categorical_cem.py
│   ├── test_ug_uncertainty.py
│   └── test_ug_rollout.py
├── experiments/
│   ├── run_ug_cem_apt.py
│   ├── run_compare_planners.py
│   └── run_compare_official_cyborg.py
└── configs/
    └── compare_ug_cem_local_online.yaml
```

---

# 3. 当前项目中已经确定的共享接口

## 3.1 状态空间

来自：

```text
chapter2_region_detection/src/state_summary.py
```

当前结构化状态为 8 维：

```python
FEATURE_KEYS = [
    "cnt_auth_fail",
    "cnt_port_scan",
    "cnt_proc_spawn",
    "cnt_outbound_conn",
    "avg_severity",
    "unique_src",
    "unique_dst",
    "risk_proxy",
]
```

因此：

```text
D = 8
state = summary.vec
shape = [8]
```

UG-CEM-APT 不重新设计状态。

---

## 3.2 动作空间

最终论文动作空间语义已经在 2026-09-15 修改为 5 类：

| 动作语义 | 英文说明 | 设计含义 |
|---|---|---|
| analyse | intrusion investigation | 对可疑活动进行调查与证据确认 |
| remove | user-level compromise removal | 清除用户级失陷/会话级威胁 |
| restore | host reimaging | 对受影响主机执行恢复/重装 |
| control_traffic | traffic blocking | 对恶意或可疑通信进行流量阻断 |
| no_op | monitor | 不执行处置，仅持续监测 |

因此：

```text
A = 5
```

**当前阶段只冻结动作数与语义，不冻结数值 ID 顺序。**

旧版 `src/action_space.py` 中的：

```text
monitor
light_evidence
heavy_evidence
local_mitigate
strong_mitigate
```

属于旧方案实现，在完成 Step 2/3 之前暂不修改，以避免把动作重构和 CEM/uncertainty 基础实现混在一个阶段。

在进入 world-model / official CC4 集成前，必须通过 Gate A 完成：

- 数值 ID 顺序；
- CybORG 底层动作映射；
- action cost / delay（若最终 reward 需要）；
- 旧 checkpoint 是否还能复用；
- 是否需要重新收集 replay / 重训 world model。

原 UG-CEM 是连续动作 CEM，领域适配仍必须使用 **Categorical CEM**。

---

## 3.3 世界模型

来自：

```text
src/world_model.py
```

核心模型：

```text
ProbDynamicsNet
DynamicsEnsemble
```

单成员输入：

```text
(s_t, a_t)
```

动作在模型内部 one-hot。

输出：

```text
mu(s_{t+1})
logvar(s_{t+1})
```

正式比较统一使用：

```text
ensemble_size M = 5
hidden_dim = 128
lr = 3e-4
```

UG-CEM-APT 与 Ours 共享同一批 world model checkpoint，不重新训练一套“更强”或“更弱”的模型作为主对比。

---

## 3.4 预测计划回报 / Reward 接口

最终 PPO reward 目前尚未确定，因此这里不再把旧的 `src.plan_eval._plan_return_from_risk()` 视为不可改变的最终论文 reward。

实现原则改为：

```text
UG-CEM 核心搜索器
    不知道 reward 细节
        ↓
RolloutEvaluator / Planner
    通过可替换的 plan_return_fn / reward adapter
    获得 predicted return
```

这样 Step 2/3 可以先独立完成，后续 reward 修改不需要重写 CEM 和 uncertainty。

现有 `_plan_return_from_risk()` 可以作为 **开发期兼容/联调实现**，但在 Gate B 之前不得视为最终 reward 定义。

公平比较最终要求：

- Ours 与 UG 使用一致的任务 reward 语义；
- 不重复计算 action cost；
- 不包含“提前预警步数”奖励；
- reward 权重在正式测试前冻结；
- test seeds 不参与 reward 调参。

---

# 4. UG 官方源码中必须保留的核心机制

以下不是“可选设计”，而是我们从官方源码中确认需要保留的核心思想。

## 4.1 CEM elite 更新

官方连续 CEM：

```text
elite_num = ceil(population_size * elite_ratio)

new_mu  = mean(elites)
new_var = var(elites)

mu  = alpha * old_mu  + (1-alpha) * new_mu
var = alpha * old_var + (1-alpha) * new_var
```

离散适配后：

```text
p_new
=
alpha * p_old
+
(1-alpha) * p_elite_frequency
```

注意：**alpha 方向不能写反。**

---

## 4.2 官方实验主要参数

官方 `offline_episodes.py`：

```text
planning_horizon = 10
num_iterations   = 5
elite_ratio      = 0.3
population_size  = 200
alpha            = 0.1
return_mean_elites = True
uncertainty_guided = True
num_particles    = 12
normalizer warmup = 100 planner calls
```

APT 适配不会机械照搬所有数值，因为本文动作空间和计划长度不同。

正式论文必须区分：

- **算法机制保留**
- **领域参数适配**

---

## 4.3 TS∞ / fixed-member rollout

官方模型配置使用：

```text
propagation_method = fixed_model
```

并在 `model_env.py` 中：

```text
sample=False
```

因此主要不确定性来源是：

> 不同 ensemble member 的预测分歧（epistemic uncertainty）

而不是每一步重新随机抽模型。

APT 适配要求：

> 对某条候选计划，某个 ensemble member 一旦选定，就负责整条 H 步 rollout。

绝对不能：

```text
第1步 model0
第2步 model3
第3步 model1
...
```

正确的是：

```text
member0: s0 -> s1 -> s2 -> s3 -> s4
member1: s0 -> s1 -> s2 -> s3 -> s4
...
member4: s0 -> s1 -> s2 -> s3 -> s4
```

---

## 4.4 官方 uncertainty 实现

官方核心代码等价于：

1. 更新全局状态均值/标准差；
2. 对预测状态做归一化；
3. 沿 particle/model 维计算 std；
4. 对状态维平均；
5. 维护每个 horizon 的 running std；
6. 再按 horizon 归一化；
7. 对 horizon 求平均；
8. 加入风险惩罚。

官方实现的核心形式：

```text
score
=
predicted_return
-
beta * uncertainty / (cem_iteration + 1)
```

这里的：

```text
/(cem_iteration + 1)
```

必须保留在 source-faithful 主版本中。

---

# 5. 经过再次审核后的关键优化

相比最初方案，后续实现采用以下优化。

## 5.1 Canonical rollout tensor 改为与官方源码一致

统一内部 shape：

```text
[H, N, M, D]
```

其中：

```text
H = horizon = 4
N = 当前 CEM population
M = ensemble members = 5
D = state dimension = 8
```

例如：

```text
[4, 64, 5, 8]
```

这样 `uncertainty.py` 可以几乎逐维对应官方：

```text
std(dim=model)
mean(dim=state)
normalize(dim=horizon)
mean(dim=horizon)
```

避免后续反复 transpose 导致维度错误。

---

## 5.2 rollout 必须批量向量化

不要对：

```text
64 plans × 5 models × 4 steps
```

逐条调用 `predict_next_by_member()`。

这样正式 100×500 实验会非常慢。

优化实现：

对每个 horizon 和每个 ensemble member：

```text
一次性输入 N 个 candidate states
shape [N,D]

一次性输入 N 个 actions
shape [N]
```

直接调用已有：

```python
model = world_model.models[m]
mu, logvar = model(state_batch, action_batch)
```

因此每轮 CEM 的模型 forward 数量约从：

```text
N × M × H
```

降到：

```text
M × H
```

这对最终大规模实验非常重要。

注意：这是计算优化，不改变算法语义。

---

## 5.3 CEM optimizer 与 MPC warm-start 分离

`categorical_cem.py` 只负责“一次 CEM 搜索”。

它接收：

```text
initial_probs (optional)
```

并返回：

```text
best sampled plan
final probability matrix
```

跨环境时间步的 warm-start 放在 `planner.py` 中维护。

官方 `TrajectoryOptimizer` 会把上一次 solution 左移一格，因此离散对应方式：

```text
上一次 final_probs:
P0
P1
P2
P3

下一步初始概率:
P1
P2
P3
Uniform
```

这样结构更清晰，也更接近官方职责划分。

---

## 5.4 主版本返回 best sampled plan

官方连续实验配置：

```text
return_mean_elites=True
```

连续空间可以返回 elite mean。

离散动作 ID 没有有意义的算术平均，因此领域适配主版本采用：

> **最高得分的已实际评估候选计划（best sampled plan）**

理由：

- 不会构造一个从未评估过的动作组合；
- 离散空间语义更合理；
- 保留 CEM 的 elite-search 核心机制。

可选敏感性实验可以增加：

```text
final_map = argmax(final_probs, axis=action)
```

但不作为第一版主结果。

论文中必须注明这是离散动作空间的必要适配。

---

## 5.5 不需要机械保留 12 particles

官方：

```text
ensemble=4
particles=12
```

当前共享世界模型：

```text
ensemble=5
```

而我们使用：

```text
sample=False
```

同一 member 重复三次不会产生新轨迹。

因此领域适配采用：

```text
M = 5 deterministic ensemble trajectories
```

即一个 member 对应一条 particle trajectory。

这是有意的领域适配，不是漏实现。

---

## 5.6 加入每个决策时刻的 raw trajectory cache

H=4、A=5 时理论计划总数：

```text
5^4 = 625
```

CEM 不同迭代中可能重复采到相同计划。

允许在**同一个真实环境决策时刻内部**缓存：

```text
plan tuple -> raw model trajectory / member returns
```

但是：

- uncertainty running stats 仍必须按当前 sampled population（包含重复样本）更新；
- 只能缓存 raw model prediction；
- 不能缓存已经经过当前 iteration normalizer 得到的最终 uncertainty score；
- 环境进入下一真实时间步后清空 cache。

这属于计算优化，不改变样本分布。

第一版如果复杂度太高，可以先不实现 cache，向量化完成后再加。

---

## 5.7 本地 local-online 与正式 CC4 必须分阶段

`OnlineCC4Client(base_url=local)` 是项目内部可交互仿真环境。

它适合：

- 单元测试；
- 联调；
- 检查 action 是否真正改变下一状态；
- 调试 CEM / uncertainty / MPC。

但正式论文最终结果不能只依赖本地仿真。

仓库已有官方 CybORG/CC4 接口：

```text
experiments/run_region_official_cyborg_eval.py
```

因此最终流程分成：

### 开发阶段

```text
local online
```

### 正式论文评估阶段

```text
CybORG / CAGE Challenge 4
```

两种方法都必须在同一官方环境适配器中比较。

---

## 5.8 官方 CC4 动作映射是最终实验前的重点审查项

当前官方 evaluator 中：

```python
ACTION_TYPE_MAP = {
    0: "Monitor",
    1: "Analyse",
    2: "Analyse",
    3: "Remove",
    4: "Restore",
}
```

即动作 1 与动作 2 在 CybORG 中都会映射成 `Analyse`。

这意味着：

> 当前五类高层动作并不是在官方 CC4 中一一对应五个不同底层动作。

因此在正式论文主实验之前，必须专门审核：

- Ours 与 UG 是否使用完全相同 mapping；
- 世界模型训练动作语义是否与官方执行语义一致；
- 是否需要重新定义/重新训练共享世界模型；
- 是否需要在论文中说明 1/2 两级补证对应同一 CybORG 动作但具有不同规划成本。

在这个问题解决前，不把 local-online 的结果当最终论文结论。

---

# 6. 统一开发参数

当前公平比较配置：

```text
configs/compare_ug_cem_local_online.yaml
```

开发阶段：

```text
D = 8
A = 5
M = 5
H = 4

population_size = 64
num_iterations  = 4
elite_ratio     = 0.30
alpha           = 0.10
beta            = 0.10
prob_floor      = 0.01

uncertainty_alpha = 0.01
normalizer_warmup = 100
```

正式实验候选：

```text
population_size = 200
num_iterations  = 5
elite_ratio     = 0.30
alpha           = 0.10
```

是否使用 N=200 / I=5 要在性能测试后决定。

H 保持 4，因为本文 LWM-RL 候选计划长度为 4，公平性优先于机械照搬原论文 H=10。

---

# 7. Beta 选择规则

开发值：

```text
beta = 0.10
```

它不是最终论文参数。

候选验证网格：

```text
beta ∈ {0, 0.1, 0.2, 0.3, 0.5, 1.0}
```

其中：

```text
beta = 0
```

自然构成：

> risk-neutral CEM 消融基线。

严禁：

> 看正式测试集结果后再选择 beta。

应使用：

```text
开发/验证 seeds -> 选择 beta
正式 test seeds -> 固定 beta
```

---

# 8. Uncertainty normalizer 的执行策略

## 8.1 官方行为

官方实验先执行：

```text
100 planner calls
keep_last_solution = False
```

用于训练：

```text
obs_mean
obs_std
horizon_std
```

然后：

```text
keep_last_solution = True
```

进入正式 episode。

---

## 8.2 APT 适配

开发阶段实现相同概念：

```text
normalizer_warmup = 100
```

warm-up 状态必须来自：

- 训练数据；
- 或专门 calibration seeds；

不能使用：

- 正式 test episode 的未来真实信息；
- 攻击标签；
- 测试结果反向调参数。

主 source-faithful 模式允许在在线决策时继续使用当前模型预测更新 running statistics，因为官方实现就是在线更新，且没有使用未来真实标签。

为做敏感性验证，可选增加：

```text
freeze_after_warmup = true/false
```

但第一版先实现官方风格在线更新。

---

# 9. 不确定性究竟是什么

本文当前 LWM-RL 的 Evidence.U：

```text
U_LWM = Var_m(plan_return_m)
```

即：

> 不同世界模型成员对整条计划“预测回报”的方差。

UG-CEM 的 uncertainty 不同：

```text
U_UG = normalized state-trajectory disagreement
```

即：

> 不同世界模型成员对未来状态轨迹的分歧。

因此：

**绝对不能直接拿 `Evidence.U` 当 UG uncertainty。**

UG 必须单独实现 `uncertainty.py`。

---

# 10. 计划评分

对第 i 个候选计划：

```text
G_i
=
所有 ensemble member 的预测计划回报平均
```

UG uncertainty：

```text
omega_i
=
基于 state trajectory disagreement 的归一化不确定性
```

第 k 轮 CEM：

```text
J_i^(k)
=
G_i
-
beta * omega_i / (k + 1)
```

CEM 按 J 排序选择 elite。

---

# 11. MPC / Receding Horizon

每个真实环境时间步：

1. 从当前状态生成 H=4 的候选动作序列；
2. CEM 优化；
3. 得到 best plan；
4. **只执行 best_plan[0]**；
5. 从真实环境获取新状态；
6. 上一轮最终概率左移作为下一轮 warm-start；
7. 再规划。

绝对不能一次执行整条 4 步计划。

---

# 11.5 两个强制设计 Gate

## Gate A — 动作语义与环境映射冻结（Step 3 之后、Step 4 正式集成前）

必须完成：

- 5 个新动作的数值 ID；
- `src/action_space.py` 与新方案同步；
- local-online 动作适配；
- CybORG/CC4 动作适配；
- `control_traffic` 的真实底层动作可用性验证；
- action cost / delay 是否保留以及如何定义；
- 旧 world-model checkpoint 是否因动作语义改变而失效。

若动作 ID 或语义与旧 replay 不一致，必须重新收集数据并重训 world model，不能直接把旧 checkpoint 当新动作模型使用。

## Gate B — PPO reward 冻结（正式 PPO 重训 / Step 8 公平比较前）

必须完成：

- PPO 单步 reward 数学定义；
- reward 每个分量的含义；
- 各权重；
- 是否需要 action cost；
- 与 CybORG official reward 的关系；
- 训练/验证 seed；
- 明确“不包含提前预警步数”。

Gate B 未通过前，可以做模块联调，但不得生成最终论文 PPO checkpoint 或主实验表格。

---

# 12. 完整实施阶段

## Step 0 — 建立实验分支

状态：**已完成**

```text
me         = 稳定完整备份
ug-cem-apt = 实验分支
```

审核结果：

```text
PASS
```

---

## Step 1 — 建立 baseline 骨架和公平比较配置

状态：**已完成**

新增：

```text
chapter2_region_detection/baselines/__init__.py
chapter2_region_detection/baselines/ug_cem_apt/__init__.py
chapter2_region_detection/configs/compare_ug_cem_local_online.yaml
```

审核结果：

```text
PASS
```

---

## Step 2 — Categorical CEM

状态：**已完成（审核通过）**

审核记录：

```text
chapter2_region_detection/docs/step2.md
```

最终结论：**PASS**

文件：

```text
baselines/ug_cem_apt/categorical_cem.py
tests/test_categorical_cem.py
```

### Step 2 的边界

本阶段只实现“单次离散 CEM 搜索器”，不接任何 APT 语义。

因此 CEM 只知道：

```text
H = planning horizon
A = n_actions = 5
N = population size
```

它不知道五个动作分别是 analyse / remove / restore / control_traffic / no_op，也不允许硬编码旧动作名称、cost、delay 或 reward。

暂时不实现：

- world model；
- uncertainty；
- CC4 / CybORG；
- LLM；
- PPO；
- reward；
- MPC warm-start 的“左移逻辑”（只保留 `initial_probs` 接口，真正 shift 在 `planner.py` 中）。

### Step 2 的实现约定

为减少后续接 PyTorch world model 时的 NumPy↔Torch 拷贝，本阶段采用 **Torch-first** 实现。

```text
plans       : torch.LongTensor [N,H]
probabilities: torch.FloatTensor [H,A]
scores      : torch.FloatTensor [N]
```

这样 Step 4 的批量 world-model evaluator 可以直接接收候选动作序列。

优化器应保持“单次调用无跨时间步规划状态”：

```text
optimize(objective_fn, initial_probs=None)
```

- `initial_probs=None`：从均匀分布开始；
- 传入 `initial_probs[H,A]`：从指定 categorical 分布开始；
- `categorical_cem.py` 不负责把上一时刻概率左移；
- 跨真实环境时间步的 warm-start 由 Step 5 的 `planner.py` 管理。

随机数生成器可以保留在 optimizer 内部，用 seed 保证可复现；但是 **不得把上一轮 `final_probs` 静默保存为下一次 optimize 的默认起点**。

### 必须实现的功能

- categorical sampling；
- elite selection；
- elite action frequency；
- `elite_num = ceil(N * elite_ratio)`；
- 与官方一致的 alpha 语义：
  `p_new = alpha*p_old + (1-alpha)*p_elite`；
- probability floor；
- 每行概率重新归一化；
- best sampled plan（跨所有 CEM iteration 的全局最佳已评估计划）；
- optional `initial_probs`；
- `final_probs` 返回；
- seed reproducibility；
- objective 接口显式接收当前 CEM `iteration`（0-based）；
- objective score shape / finite-value 防御；
- 配置与输入 shape 校验。

建议返回：

```text
CEMResult
├── best_plan      [H], long
├── best_score     scalar
└── final_probs    [H,A], float
```

可选的 debug history 可以后续增加，不作为 Step 2 必需项。

### Probability floor

不要简单 `clamp(min=floor)` 后忘记归一化。

推荐使用**幂等的 lower-bound projection**，避免 MPC warm-start 时对已经合法的概率分布重复“抹平”：

```text
p = normalize(p)
residual = max(p - floor, 0)
p_floor = floor + (1 - A*floor) * residual / sum(residual)
```

这样同时满足：

- 每个动作概率 >= floor；
- 每行概率和严格为 1；
- 如果输入本来已经满足 `p>=floor` 且行和为 1，则输出保持不变（幂等）；
- 下一时刻把上一轮 `final_probs` 作为 warm-start 时不会因为再次应用 floor 而被无意义地向 uniform 拉回；
- 需要验证 `A * floor < 1`。

### 非有限 score

正式模型以后可能出现 NaN/Inf。Step 2 中要求：

- 将非有限候选视为不可选（例如映射到 `-inf`）；
- 如果整批候选全部非有限，直接抛异常；
- 不能让 NaN 因排序行为意外进入 elite。

### Step 2 测试

至少验证：

1. 默认概率 shape = `[H,A]` 且每行和为 1；
2. sampled plans shape = `[N,H]`，dtype 为 long，动作范围 `[0,A-1]`；
3. elite 数使用 `ceil(N*elite_ratio)`；
4. elite frequency 计算正确；
5. prob_floor 生效且每行仍严格归一；
6. alpha 更新方向与官方一致，特别测试 `alpha=0` 和 `alpha=1`；
7. 相同 seed + 相同 initial_probs + 相同 objective 可复现；
8. 人工目标计划能被 CEM 找到；
9. 自定义 `initial_probs` 能真正影响首轮采样，且不被优化器静默覆盖；
10. 两次独立 optimize 默认都从 uniform/传入 initial_probs 开始，而不是自动继承上一次 `final_probs`；
11. 非法配置（H/A/N/ratio/alpha/floor）抛异常；
12. `initial_probs` shape、负概率、全零行、NaN/Inf 抛异常；
13. objective 能收到正确的 iteration 序列 `0..num_iterations-1`；
14. objective 返回 shape 错误抛异常；
15. objective 全部为非有限值时抛异常。

### Debug Oracle

H=4、A=5：

```text
5^4 = 625
```

可枚举所有计划作为测试 oracle。

推荐人工 objective 不使用动作 ID 的“距离平方”作为主要测试，因为动作 ID 只是类别标签，不代表连续强度。优先使用：

```text
score(plan) = number_of_positions_equal_to_target
```

目标例如：

```text
target = [4,3,2,1]
```

其唯一最优分数为 4。

exhaustive oracle：

> 只用于单元测试和 debug，不作为正式比较方法。

### Step 2 验收标准

只有满足以下条件才允许进入 Step 3：

```text
categorical_cem.py 不 import src/world_model / CC4 / PPO / LLM
tests 全部通过
未修改 src/
未修改 UG 官方源码
Git diff 只包含 Step 2 文件（外加本任务书修订）
```

---

## Step 3 — UG Uncertainty

状态：**下一步**。

文件：

```text
baselines/ug_cem_apt/uncertainty.py
tests/test_ug_uncertainty.py
```

输入统一：

```text
next_states.shape = [H,N,M,D]
```

维护：

```text
obs_mean[D]
obs_std[D]
horizon_std[H]
```

初始化对应官方：

```text
obs_mean = 0
obs_std = 0.1
horizon_std = 0.01
alpha_norm = 0.01
```

更新逻辑：

```text
obs_mean = (1-a)*obs_mean + a*mean(next_states)
obs_std  = (1-a)*obs_std  + a*std(next_states)
```

归一化后：

```text
member_std = std(dim=M)
state_mean = mean(dim=D)
horizon_normalize
omega = mean(dim=H)
```

添加很小 `eps` 防止除 0，这是数值稳定性适配。

### Step 3 测试

- 所有 member 预测相同 -> uncertainty 接近 0；
- member 分歧增大 -> uncertainty 增大；
- shape 正确；
- 无 NaN/Inf；
- running stats 更新正确；
- reset 正确；
- update_stats=False 时统计量不变化。

---

## Step 4 — World Model Rollout Evaluator

状态：待执行。

文件：

```text
baselines/ug_cem_apt/rollout_evaluator.py
tests/test_ug_rollout.py
```

目标：

```text
state [D]
plans [N,H]
↓
returns [N]
next_states [H,N,M,D]
```

要求：

- 固定 member 整段 rollout；
- 使用 deterministic member mean；
- 不使用随机 aleatoric sample；
- 批量向量化 N 条候选；
- 不修改 `src/world_model.py`；
- rollout evaluator 不硬编码最终 reward；通过可替换 `plan_return_fn / reward adapter` 计算预测回报。开发期可临时调用 `_plan_return_from_risk()`，但 Gate B 后必须替换/确认最终定义。

### Step 4 必须测试

- tensor shape；
- action ID 越界；
- horizon 不一致；
- fixed-member 语义；
- batch 与逐条计算结果一致；
- returns finite。

---

## Step 5 — UGCEM Planner

状态：待执行。

文件：

```text
baselines/ug_cem_apt/planner.py
```

整合：

```text
Categorical CEM
+
Rollout Evaluator
+
UG Uncertainty
```

每个 CEM iteration：

```text
plans
 -> rollout
 -> G
 -> omega
 -> score = G - beta*omega/(iteration+1)
 -> elites
 -> categorical update
```

Planner 负责：

- initial uniform probs；
- previous final_probs；
- MPC shift warm-start；
- reset；
- 返回 best_plan[0]；
- debug info。

建议返回：

```python
action_id, info
```

其中 info 至少包含：

```text
best_plan
best_score
predicted_return
uncertainty
final_probs
planning_time
```

---

## Step 6 — Normalizer Warm-up

状态：待执行。

实现官方的：

```text
100 planner calls
keep_last_solution=False
```

APT 版：

- 使用 calibration states；
- warm-up 时不携带上一个 CEM solution；
- 只更新 uncertainty normalizer；
- 不执行/不学习正式测试 episode；
- 完成后再打开 MPC probability warm-start。

必须记录：

```text
obs_mean
obs_std
horizon_std
```

是否 finite。

---

## Step 7 — Local-online Smoke Test

状态：待执行。

文件：

```text
experiments/run_ug_cem_apt.py
```

第一轮只跑：

```text
1 episode × 20 steps
```

确认完整链路：

```text
env.reset
 -> summary.vec
 -> UG-CEM
 -> action
 -> env.step
 -> new state
 -> replan
```

日志至少记录：

```text
t
risk
best_plan
pred_return
uncertainty
score
action
env_reward
planning_time
```

通过后：

```text
5 episodes × 100 steps
```

用于稳定性和速度检查。

---

## Step 8 — 公平比较 Harness

状态：待执行。

文件：

```text
experiments/run_compare_planners.py
```

支持：

```text
--method ours
--method ug_cem
```

### Ours 必须是“干净论文版本”

```text
LLM prior
 -> build_evidence
 -> PPO
 -> plans[idx][0]
```

不得加入：

```text
_heuristic_plan_score
_obs_heuristic_action
action_vote
risk/severity hand rules
```

因为现有 `run_deploy_region.py` 包含额外工程启发式，不适合作为公平主实验 Ours。

### UG-CEM

```text
summary.vec
 -> UGCEMPlanner
 -> best_plan[0]
```

### 必须共享

```text
environment
seed
summary
action mapping
world model checkpoint
predicted return
episode count
episode length
metrics
```

---

## Step 9 — Beta 验证和消融

状态：待执行。

比较：

```text
beta = 0
0.1
0.2
0.3
0.5
1.0
```

用途：

```text
beta=0        -> CEM
beta=best_val -> UG-CEM
```

这会直接得到：

> “不确定性引导是否真正有效”的消融实验。

---

## Step 10 — 正式 CybORG / CC4 对比

状态：待执行。

基于：

```text
experiments/run_region_official_cyborg_eval.py
```

新建统一 evaluator，而不是直接复制 hybrid boost。

正式 evaluator 必须：

- Ours 与 UG 共用相同 CybORG；
- 相同 red/green agent；
- 相同 episode seeds；
- 相同 target region；
- 相同 high-level -> CybORG action adapter；
- 相同 official reward；
- 不给任意一方额外 tactical heuristic。

先：

```text
2 episodes × 50 steps
```

再：

```text
20 episodes × 100 steps
```

最后论文设置：

```text
100 episodes × 500 steps
```

如果计算量过高，先报告 runtime，再决定是否调整 CEM population；不能悄悄只降低 UG 的资源而不记录。

---

## Step 11 — 指标、日志与统计

正式结果至少统一输出：

```text
mean episode return
std episode return
planning latency
action distribution
predicted return
uncertainty
```

如果论文主指标需要：

```text
average recovery time
recovery precision
```

`early-warning lead steps / 提前预警步数` 已从最终指标中删除。

必须让两个方法使用同一 metric evaluator。

建议使用完全相同的 episode seed 列表，因此结果天然形成 paired samples。

可以补充：

- 95% bootstrap CI；
- paired significance test；

但论文主表仍以统一指标为主。

---

## Step 12 — 最终复现实验与论文表格

最终至少形成：

```text
LWM-RL (Ours)
UG-CEM-APT
CEM-APT (beta=0)
```

以及后续其它论文 baseline：

```text
UA-MCTS-APT
CAICS-APT
```

最终报告中必须清楚写明：

> 所有对比方法均在保持其核心决策机制的基础上适配至本文状态空间、动作空间及 CC4 实验接口，因此属于领域适配实现，而非逐行完全复现原作者环境。

---

# 13. 正式比较时禁止的行为

以下行为一律视为“不公平实验”：

1. UG 使用不同 world model checkpoint；
2. Ours 使用额外 heuristic，UG 不使用；
3. 只有 UG 被额外收取动作成本；
4. 只有一个方法看到未来真实状态；
5. 使用 test seed 调 beta；
6. 使用 test 攻击标签做 planner 输入；
7. replay 固定轨迹环境用于证明在线控制性能；
8. Ours 和 UG 使用不同 episode seed；
9. 为了提高 UG 结果临时改变 reward；
10. 为了提高 Ours 结果保留 `run_deploy_region.py` 中额外启发式。

---

# 14. 代码审核标准

每个 Step push 后，ChatGPT 必须直接读取 GitHub 审核。

## Step 2 审核

检查：

- categorical sampling；
- elite 数；
- alpha 方向；
- prob floor；
- probability normalization；
- best plan；
- seed；
- 测试覆盖。

## Step 3 审核

检查：

- canonical shape `[H,N,M,D]`；
- std 是否沿 M；
- obs stats；
- horizon stats；
- `/(i+1)` 是否由 planner 正确使用；
- epsilon 是否仅用于数值稳定。

## Step 4 审核

检查：

- member 是否固定；
- batch rollout；
- 是否错误使用 aleatoric sampling；
- 是否共享原世界模型；
- predicted reward 是否一致。

## Step 5 审核

检查：

- CEM + uncertainty 组合；
- MPC warm-start；
- 只执行首动作；
- reset；
- logging。

## Step 8+ 审核

检查：

- 公平性；
- seed；
- checkpoint；
- action mapping；
- reward；
- label leakage；
- hidden heuristic。

---

# 15. Git 提交原则

建议一个阶段一个 commit。

示例：

```text
Step 1
chore(ug-cem): add baseline scaffold and comparison config

Step 2
feat(ug-cem): implement categorical CEM optimizer

Step 3
feat(ug-cem): add uncertainty normalization

Step 4
feat(ug-cem): add vectorized world-model rollout evaluator

Step 5
feat(ug-cem): integrate uncertainty-guided planner

Step 7
test(ug-cem): add local online smoke evaluation

Step 8
feat(eval): add fair planner comparison harness
```

不要：

- 一个 commit 同时改十几个无关模块；
- 把 `__pycache__`、`.pyc`、临时日志一起提交；
- 直接改 `me`；
- 改 UG 官方源码。

---

# 16. 当前进度

截至 2026-09-15：

```text
[x] Step 0  创建 ug-cem-apt 分支
[x] Step 1  baseline scaffold + comparison config
[ ] Step 2  Categorical CEM
[ ] Step 3  UG uncertainty
[ ] Gate A  冻结新动作 ID / 环境映射 / world-model 数据兼容性
[ ] Step 4  Vectorized rollout evaluator
[ ] Step 5  UGCEM planner
[ ] Step 6  Normalizer warm-up
[ ] Step 7  Local-online smoke test
[ ] Gate B  冻结新的 PPO reward 并重训正式 PPO
[ ] Step 8  Fair comparison harness
[ ] Step 9  Beta validation + ablation
[ ] Step 10 Official CybORG/CC4 comparison
[ ] Step 11 Metrics/statistics
[ ] Step 12 Final experiments / thesis tables
```

当前下一步：

> **Step 2：只实现 Categorical CEM，不接 world model。**

---

# 17. 每次继续时的固定口令

用户可以直接说：

```text
下一步
```

或：

```text
Step X 已 push，请审核
```

ChatGPT 收到后必须先读取：

```text
chapter2_region_detection/docs/UG_CEM_APT_REPRODUCTION_PLAN.md
```

再读取当前分支 diff 和本阶段相关源码，然后继续。

如果后续讨论中出现与本文件不同的新方案：

1. 先说明为什么要改；
2. 确认是否属于 bug fix、效率优化还是算法改变；
3. 算法改变必须先更新本文件；
4. 再修改实现。

---

# 18. 一句话记住整个方案

```text
共享状态 + 共享世界模型 + 在 Gate B 后冻结的统一 reward/预测回报，
只把 LLM+PPO 规划选择器替换成
Categorical CEM + 官方风格 trajectory uncertainty，
每步只执行最佳计划的第一个动作，
最后在相同 CybORG/CC4 条件下做 paired comparison。
```
