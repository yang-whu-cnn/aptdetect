# 第三版论文（0917）CC4 最终实验总任务书

> 状态：可执行总方案；不包含任何伪造或旧实验数字
> 目标分支：`ug-cem-apt`
> 正式协议：CC4 / D27 / A4 / 100 episodes × 500 ticks × 5 independent training repeats
> 第三版主方法：LLM prior → K=6 candidate plans → ensemble world model H=4 → PPO 选计划 → 只执行首动作 → 重规划

## 0. 结论先行

目前仓库已经完成第三版奖励回放、D27/A4 合约、绝对状态世界模型和响应奖励预测器，但**还没有任何可直接填入第三版 Table 1/2/3 的正式结果**。旧版五动作、旧奖励、短回合、旧 replay 或单次运行的数字均不得迁入第三版表格。

最快且可审计的路线是：先一次性封闭统一评测器和指标定义，再并行实现六条论文基线与主方法/消融；所有方法共享 CC4 场景、可见信息边界、A4 动作适配、动作持续时间、测试种子和最终评测器。训练目标按用户规则分流：论文有自己的奖励则保留其奖励；没有独立奖励则使用第三版奖励。无论训练奖励是什么，最终都用第三版表格的四项统一指标评测。

本任务书纠正两个容易导致错误比较的问题：

1. DCA 是攻击检测与攻击路径推断方法，不是 PPO 响应控制器；其进入 Table 1 时必须标注为 `DCA-CC4 (adapted)`，通过透明、固定的响应映射接入 A4。
2. CARL 是 CAICS/Dyna 风格的模型增强 PPO，而不是决策时树搜索；不得复用 MCTS/CEM 名称或实现冒充 CARL。

## 1. 冻结的研究问题和表格

### 1.1 Table 1：主对比

| 行名 | 正式实现名 | 训练奖励 | 是否有官方源码 | CC4 状态 |
|---|---|---|---|---|
| UAMCTS | `UAMCTS-CC4 (adapted)` | 原论文势函数塑形奖励；基础任务奖励使用第三版奖励 | 未在论文/官方页面发现 | 待实现 |
| RSMBRL | `RSMBRL-CC4` | 第三版奖励 | 有，固定上游 commit | 已有 UG-CEM 核心可复用，需改名封装和验收 |
| CARL | `CARL-CC4 (adapted)` | 原论文标准 CAICS 奖励的 CC4 显式映射 | 未发现 | 待实现 |
| DCA | `DCA-CC4 (adapted)` | 原论文攻击路径奖励只用于推断；响应映射无学习奖励 | 未发现 | 待实现 |
| PriorRL | `PriorRL-PPO-CC4` | 第三版奖励 | 有部分官方源码；公开仓库只有 value-based 路线 | 待实现 PPO-KL 适配 |
| TERLA | `TERLA-A4` | 原论文 cyber reward | 未发现 | 待实现 |
| Ours | `LWM-RL` | 第三版 Full-Reward | 仓库已有主体组件 | 正式 PPO/评测未完成 |

Table 1 列固定为：

1. `CC4 Official Reward ↑`
2. `Operation Failure Penalty ↓`
3. `Recovery Precision ↑`
4. `Recovery Time ↓`

### 1.2 Table 2：主方法组件消融

| 行名 | LLM prior | World model | PPO | 候选/动作语义 |
|---|---:|---:|---:|---|
| RL-Only | 否 | 否 | 是 | PPO 直接选择 A4 |
| LLM-RL | 是 | 否 | 是 | LLM 给 K=6 候选/先验，PPO 选候选 |
| WM-RL | 否 | 是 | 是 | 非 LLM 候选生成器 + WM rollout，PPO 选候选 |
| LWM-RL | 是 | 是 | 是 | 完整方法 |

四个变体必须分别初始化、分别训练；禁止训练完整模型后在测试期遮蔽模块来伪造消融。四者使用同一 Full-Reward、同一训练预算和同一测试种子。

### 1.3 Table 3：奖励消融

| 行名 | 真实训练奖励 | 候选 rollout 奖励预测器 |
|---|---|---|
| Delay-Only | `-lambda_time * delay` | 单独训练 Delay-Only predictor |
| Fail-Only | `-lambda_failure * HostWorkFail penalty` | 单独训练 Fail-Only predictor |
| Full-Reward | 两项之和 | 现有 Full-Reward predictor |

世界模型可以共享，但三个奖励预测器必须独立训练、独立固定。奖励语义必须同时作用于 PPO 的真实 transition reward 和 WM 候选 rollout 的 predicted reward，不能只改其中一处。

## 2. 不可变统一实验合约

### 2.1 环境与动作

- 环境：CybORG Challenge 4，`FiniteStateRedAgent`。
- 每回合严格 500 environment ticks；未满 500 或异常退出的回合不得静默进入正式结果。
- 五个 Blue agent：`blue_agent_0..4`。五个 agent 不是五次独立重复。
- A4：`0=no_op/Sleep`、`1=analyse/Analyse`、`2=remove/Remove`、`3=restore/Restore`。
- 持续时间：1/2/3/5 ticks；折扣使用 `gamma_tick ** duration`。
- 没有合法、可观测目标时统一回退 Sleep，并同时记录 `requested_action`、`executed_action`、原因和目标。
- 所有方法必须通过 `shared/action_contract.py`、统一 target resolver 和 CybORG adapter，禁止各基线自行改变动作有效性。

### 2.2 信息公平而非表示强制相同

所有策略只能使用当前及历史 Blue 可观测信息；禁止读取 Red session、真实 compromise flag、未来事件或评测标签做决策。默认输入是 `shared.formal_state.FormalStateEncoder` 的 D27。

为了保留论文核心结构，允许以下**同信息集、不同表示**：

- TERLA 将相同 Blue observation 构造成异构图后输入 HGT。
- DCA 将相同 Blue evidence 构造成告警序列和攻击路径状态。
- 其余方法使用 D27。

隐藏真值只允许在环境外部的训练奖励计算、离线监督标签和最终评测统计中出现，并且这些通道必须与 policy observation 隔离并通过泄漏测试。

### 2.3 数据、模型和种子

- train replay：32 个种子 `1000..1031`，当前 43,297 transitions。
- validation：8 个种子 `2000..2007`，当前 10,796 transitions。
- calibration：`3000..3007`，只做归一化器/阈值/超参冻结。
- final test：100 个共享种子 `4000..4099`。
- 五个独立训练重复：建议 policy seeds `51001, 51002, 51003, 51004, 51005`。
- 同一 repeat 内所有方法使用相同 test episode seeds，便于配对分析；测试 seed 不得参与训练、早停、超参选择或归一化更新。
- 每次正式运行保存代码 commit、配置 SHA256、论文 PDF SHA256、上游源码 commit、Python/依赖版本、硬件、训练 seed、episode seed 和模型文件 SHA256。

### 2.4 两条奖励通道

必须在代码中区分：

- `training_objective_reward`：用于训练/规划，按各论文规则选择。
- `evaluation_metrics`：只用于统一最终评价，不回流给策略。

奖励矩阵冻结如下：

| 方法 | 训练/规划奖励 | 选择理由 |
|---|---|---|
| LWM-RL、RSMBRL、PriorRL | 第三版 `-λ1*delay-λ2*fail` | 论文没有可直接迁移的独立 cyber reward |
| UAMCTS | 第三版基础奖励 + 原论文 potential shaping | 原方法明确提出 progress potential shaping |
| CARL | 原论文 standard CAICS reward 的固定 CC4 映射 | 原方法有明确奖励 |
| DCA | 原论文攻击方路径奖励仅用于其路径推断 | 原奖励不是 defender control reward |
| TERLA | 原论文 `-(red sessions + service unreliability × OT multiplier)` | 原方法有明确 cyber reward |

## 3. 四项最终指标的唯一计算定义

### 3.1 CC4 Official Reward ↑

每回合累加环境返回的**团队官方奖励**，每 tick 只计一次，禁止因为五个 agent 返回相同团队奖励而乘以 5。正式 repeat 值是 100 回合的算术平均。

### 3.2 Operation Failure Penalty ↓

统计全 Blue 管辖范围内 CC4 `GreenLocalWork/Host Work Fail` 的原始负惩罚绝对值：

`failure_penalty_episode = -sum(min(raw_lwf_penalty, 0))`

因此数值越小越好。另存 signed raw sum 和 failure count 做审计，但表格只放正的 penalty magnitude。不得只统计 incident host。

### 3.3 Recovery Precision ↑

恢复干预限定为实际执行并完成的 `remove` 或 `restore`：

`precision = TP / (TP + FP)`

- TP：动作真正开始执行前，目标 host 存在 active incident。
- FP：动作真正开始执行前，目标 host 不存在 active incident。
- 回退到 Sleep、不合法请求、未开始动作不进入分母；同时单独记录 fallback rate。
- 按 repeat 汇总 100 回合的 TP/FP 后做 micro-average，不能先给“无恢复动作”的回合随意赋 0 或 1。
- 若整个 repeat 的分母为 0，值为 `NA` 并触发正式结果审查；不得替换为 0。

动作后的实际根除成功率作为辅助诊断保存，但不与 targeting precision 混为一谈。

### 3.4 Recovery Time ↓

主表采用包含未恢复事件的**截尾恢复时间**：每个 incident 从 `t_compromise` 计到 `t_recovered`；若回合结束仍未恢复，则计到 episode end（500）。repeat 内对全部 incident 做 micro-average。这样不会因漏掉难以恢复的事件而获得虚假的低时间。

必须额外输出两个辅助字段：completed-only mean recovery time、unrecovered rate。论文正文/表注需要说明主表采用 episode-end restricted/censor-aware 定义。

### 3.5 `mean ± std` 聚合

每个训练 repeat 先在 100 个 episode 上得到一个 repeat-level 值；表格最终写五个 repeat-level 值的均值和样本标准差（`ddof=1`）。不得把 500 个 episode 当成 500 个独立训练重复计算标准差。保留 episode-level 原始数据以便 paired bootstrap/Wilcoxon 做补充统计，但显著性不是填表阻塞项。

## 4. 六篇基线的实现任务书

### 4.1 UAMCTS-CC4（无源码，适配复现）

**论文核心**：学习世界模型上的 MCTS；LLM action prior；progress potential shaping；把世界模型、进度估计和先验不确定性组合后进入 hybrid UCB；根节点按 visit count 选动作。

**实现边界**：

- 复用冻结的 5-member probabilistic WM，H=4；禁止在线更新 WM。
- 复用离线 LLM prior cache；正式运行 cache miss 必须 fail closed，禁止在线 LLM 调用。
- 节点保存 `N(s)`, `N(s,a)`, `Q(s,a)`, prior `P(a|s)` 和三类 uncertainty。
- hybrid score 实现论文项：`Q - rho(phi)*u + c_puct*P*sqrt(N)/(1+N_sa) + c_u(phi)*u`。
- uncertainty：WM ensemble disagreement、progress scorer uncertainty、LLM prior entropy，经 calibration split 归一化；测试期只读。
- progress potential `Phi(s)` 不使用图像/VLM，因为 CC4 无等价图像。训练一个仅接收 Blue-observable D27 的轻量 ensemble progress scorer，监督目标只来自 train replay；其任务是预测归一化 future response return/progress。此项必须在名称和论文中标为 CC4 modality adaptation。
- shaped reward：`r' = r_v3 + beta*(gamma^d*Phi(s')-Phi(s))`；leaf 可加论文终端 potential 项。`r_v3` 由冻结 Full-Reward predictor 给出。
- 每个决策 epoch 从根搜索，执行 visit 数最高分支的首动作，再清根重规划。

**待调参数**：simulation budget、`c_puct`、`rho`、`c_u`、potential `beta`。只允许 calibration/validation 选择。先用 `{64,128}` simulations 的小网格，预算不足时固定 64；其他参数只做小型正交网格，避免组合爆炸。

**文件建议**：`baselines/uamcts_cc4/{planner.py,node.py,progress.py,config.yaml,README.md}`。

**单测/验收**：手工树上 UCB 公式、uncertainty 单调性、root visit 选择、只执行首动作、同 seed 确定性、隐藏字段置乱不改变 action、cache miss 失败、测试期 scaler 不变。

### 4.2 RSMBRL-CC4（有源码，算法适配）

**上游固定**：`sradicwebster/mbrl-lib` uncertainty-guided branch，commit `9f97859594f2b0547e01193a8758936090b0b2ec`。

**论文核心**：PETS 风格概率 ensemble + uncertainty-guided CEM，目标为 `sum reward - beta*uncertainty`。原始默认：ensemble=4、3×200 MLP、H=10、CEM iterations=5、elite ratio=.3、population=200、particles=12、alpha=.1。

**CC4 实现**：

- 当前 `baselines/ug_cem_apt` 已实现最接近的 categorical CEM，不重写核心；增加 `baselines/rsmbrl_cc4` 薄封装和来源说明。
- 为统一数据/算力和隔离 planner 贡献，正式比较共享第三版冻结 WM 与 Full-Reward predictor；这是 CC4 reproduction，不宣称位级复现原连续控制模型。
- 连续 CEM 改成每个 horizon step 的 A4 categorical distribution；采样完整动作序列，按 elite 更新概率。
- H 固定 4、duration-aware discount、执行首动作；population/iterations/elite/particles 尽量保留论文值，在显存不够时只允许在配置中显式降级并重新跑全部 repeats。
- epistemic uncertainty 从固定 ensemble member rollout disagreement 得到；normalizer 只用 calibration，测试时不得 EMA 更新。
- `beta` 只在 validation 上选；`beta=0` 是诊断 CEM，不占 Table 1 行。

**验收**：与上游公式逐项对照；`beta` 增大时同一候选的 risk-adjusted score 不增；categorical 概率归一；同 seed 可重复；测试 scaler hash 前后相同；输出名称只能是 RSMBRL-CC4。

### 4.3 CARL-CC4（无源码，适配复现）

**论文核心**：CAICS/Dyna/SimPLe 风格模型增强 PPO。真实 trajectory 后训练 transition model 与 causal reward SCM，生成多条 model trajectories，真实与合成经验共同更新 PPO；不是在线规划器。

**CC4 实现**：

- Policy：D27 → MLP PPO → A4，保持 action duration-aware GAE。
- Transition：最快路线复用冻结第三版 WM 作为环境模型，并在结果中披露；若审稿需要更严格复现，再训练 CARL 自有 NN transition 作为补充。
- Causal reward model：显式 DAG 节点至少含 observable alert/evidence、action、target role、pre/post incident proxy、service failure/availability、reward；根节点经验采样，连续/离散子节点分别线性模型或 RandomForest。只用 train split 拟合。
- 每条真实 rollout 生成 8 条合成 rollout（论文设置）；rollout horizon 先保持 256，若 H=4 WM 的长期误差门禁失败，则使用截断 imagined rollout 并在方法名/附录披露，不得悄悄修改。
- 真实和合成样本分别打标签，日志给 synthetic ratio；PPO minibatch 中保持固定比例。

**原奖励的 CC4 映射**：选论文 standard CAICS reward，不使用 HVT 变体。论文的 compromised-node 状态项映射为当前 active incident 数，compromise change 映射 incident start/end；无 isolate 动作，因此 isolate 状态/变化项为 0；动作成本映射为 analyse=0、remove=1、restore=1、no-op=0。固定权重保留 `alpha_comp=.025, beta_comp=2`；论文中的 ±1000 terminal 仅在 CC4 可定义的攻击清除/严重失败终止事件触发，固定 500 tick 正常结束不加。此训练奖励可用环境真值计算，但 policy observation 不得包含真值。

**验收**：真实/合成 buffer 不串 seed；SCM intervention 改 action 时 reward 可响应；测试期间模型冻结；合成轨迹从训练状态起点采样；隐藏真值置乱不改变 policy action；原奖励映射有逐项单测。

### 4.4 DCA-CC4（无源码，检测器到响应器的适配）

**论文核心**：配电系统攻击告警的 DBSCAN 聚类、时序排序、abductive reasoning、tabular model/value iteration/Dyna，用于推断攻击者目标和路径。原论文 action 是攻击者可能采取的攻击，不是 defender response。

**实现原则**：不能声称直接复现防御控制器。Table 1 行、配置、图注统一写 `DCA-CC4 (adapted)`。

**CC4 适配**：

- 从 Blue observable process/network/file evidence 构造离散 alert token；按 host、时间窗和证据类型聚类，默认 DBSCAN `minPts=3`，epsilon 只在 validation 冻结。
- 构建有限攻击路径状态与候选目标，使用 Dyna/tabular value iteration 更新 `(state, attacker_action, next_state, reward)`。
- 保留论文攻击方奖励结构：power-loss 项改成归一化 mission/service impact proxy，`1/n_step` 保留，命中推断攻击目标时加 `r_obj`。该奖励只用于攻击路径/目标排序，不训练 defender reward。
- 固定、可审计的响应映射：无证据→no-op；单一/低置信证据→analyse；高置信 user-level compromise 且可移除→remove；持续、多通道或 remove 后仍持续的 compromise→restore。阈值在 validation 固定。
- 目标 host 只能来自观察到的 evidence，不得从真实 compromised host 列表读取。

**优势**：无需 PPO 训练，是最早可跑通的端到端基线。
**验收**：已知告警序列的聚类/路径单测；映射四分支覆盖；相同 observation 相同 action；隐藏真值置乱不变；空告警安全 no-op；所有阈值进入 manifest。

### 4.5 PriorRL-PPO-CC4（有部分源码，PPO-KL 适配）

**上游固定**：`yanxue7/RL-LLM-Prior`，commit `13b99c9bba5462b9c84c2a433b4a5ddec046177e`。当前公开仓库只包含 DQN/CQL value-based 脚本，论文的 GFlan-Prior/PPO 路线需要在本仓库实现。

**论文核心**：在环境回报之外用 `KL(pi || p_LLM)` 正则策略，LLM 生成候选动作/先验；不能加入 WM rollout，否则与 LLM-RL/LWM-RL 混淆。

**CC4 实现**：

- D27 → PPO actor/critic → A4。
- 从离线 cache 中 K=6 计划的首动作频率和置信度形成四动作 `p_LLM(a|s)`；加小 epsilon 后归一化。
- PPO loss 增加 `alpha_KL * KL(pi(.|s) || p_LLM(.|s))`，方向固定，不得误写反向 KL。
- cache miss fail closed；训练/测试均禁止在线调用 LLM。
- 使用第三版 Full-Reward，duration-aware GAE；不加载世界模型。
- `alpha_KL` 在 validation 小网格选择，例如 `{0.01,0.05,0.1,0.5}`；选定后冻结五次重复。

**验收**：KL 方向数值单测、alpha=0 退化成 RL-Only、prior 极尖时梯度方向正确、无 WM import/调用、cache miss 失败、五次独立初始化。

### 4.6 TERLA-A4（无源码，结构复现）

**论文核心**：CC4 网络的异构图表示、两层 HGT、global sum pooling、PPO，以及按 least/most compromised observable score 自动选目标的缩减动作空间。

**CC4 实现**：

- 从与 D27 同源的 Blue observation 构造 mission/subnet/host 异构图；不得加入 hidden Red truth。
- 两层 HGT + ReLU + global sum pooling。
- 原论文是 5 动作（含 deploy decoy）；第三版删除 decoy 后采用 A4。按论文维度公式 `(host_features + actions)*10`，2 host features + 4 actions → graph hidden 60，PPO MLP `[120,120]`。
- `analyse` 目标取 observable score 最低的可疑候选；`remove/restore` 目标取最高 compromised score；无合法目标统一 Sleep。
- 采用论文 single shared TERLA agent：五个 Blue agent 共享一套 policy 参数，但各自 recurrent/history/context 不串扰。
- PPO 起点：gamma=.97、lr=1e-4、entropy=.01、rollout=128；训练预算尽量对齐论文约 1M steps。若主方法正式训练预算低于此值，必须给所有 PPO 方法相同 environment-interaction budget，不能单独优待 TERLA。
- 训练使用原论文 reward：负的 Red sessions 加 service unreliability×OT multiplier。奖励包装器可读环境真值，但 policy 不能。若 CC4 当前版本取不到完全相同字段，必须停在适配审查，不得静默改成第三版奖励。

**验收**：图节点/边 schema、permutation/shape、五 agent 参数共享、目标选择、duration、reward wrapper 数值、hidden-truth leakage、固定 seed smoke。

## 5. 主方法与消融的完成路径

### 5.1 LWM-RL 正式 PPO

已有可复用组件：D27/A4、K=6 prior 接口、H=4 ensemble WM、Full-Reward predictor、duration-aware PPO core。剩余工作：

1. 生成/校验足量 offline prior cache，推荐至少 2,000 个代表性状态；按 agent/action/evidence 分层覆盖。
2. 正式 runtime 对 cache miss、损坏、schema/version 不匹配 fail closed。
3. 候选特征固定为 LLM prior、WM predicted return、WM uncertainty 及必要的 plan/action embedding；不得加入 hidden truth。
4. 五个 policy seed 独立训练；每次保存 best checkpoint 的选择规则只能看 validation。
5. 逐 checkpoint 跑 100×500 final test；所有结果进入统一 raw schema。

### 5.2 Table 2 实现细节

- RL-Only：actor 输出 A4；参数量报告出来。
- LLM-RL：候选来自 LLM，候选分数不包含 WM return/uncertainty；PPO 从 K 个候选中选。
- WM-RL：候选由固定非 LLM 生成器产生，建议使用均匀覆盖 + action-validity-aware 补齐，候选数量仍为 6；加入 WM return/uncertainty。
- LWM-RL：完整输入。
- 为避免重复候选改变有效 action mass，保存候选去重前后列表、mask 和选择概率；同一首动作的多个计划可保留，但聚合规则固定。

### 5.3 Table 3 实现细节

从现有 train/validation replay 分别派生 delay/fail 标签，训练两个新 predictor；模型架构、输入、标准化器、optimizer、学习率、batch size、固定 50 epochs、模型随机种子和 split 与 Full-Reward predictor 相同。Delay-Only 保持标准化奖励标签上的 MSE；Fail-Only 经正式批准仅将逐样本损失替换为标准化奖励标签上的未加权 SmoothL1/Huber（`beta=1.0`），禁止类别加权与正例重采样，原 MSE 仅保留为 diagnostic-only。两者均不做 validation checkpoint selection。先做 H1/H4 RMSE、rank correlation 和 constant/persistence baseline 门禁，再允许进入 PPO。

## 6. 代码落地结构与公共接口

在现有目录上增量实现，不另起冲突工程：

```text
chapter2_region_detection/
  baselines/
    uamcts_cc4/
    rsmbrl_cc4/
    carl_cc4/
    dca_cc4/
    priorrl_cc4/
    terla_cc4/
  configs/formal_v3/
    protocol.yaml
    methods/*.yaml
    ablations/*.yaml
  formal_experiments/
    common/
      agent_api.py
      reward_router.py
      run_manifest.py
    evaluation/
      metrics_v3.py
      run_formal_suite.py
      aggregate_tables.py
      validate_formal_run.py
  tests/
    test_metrics_v3.py
    test_reward_router.py
    test_formal_manifest.py
    test_no_information_leakage.py
```

所有 agent 实现统一协议：

```python
reset(*, episode_seed, agent_id) -> None
act(*, blue_observation, tick, action_mask) -> ActionDecision
observe(*, transition) -> None  # evaluation mode 必须只记录，不更新参数/归一化器
close() -> None
```

`ActionDecision` 至少含 method、requested A4、executed A4、target、fallback、decision latency、planner diagnostics。训练和评测模式必须显式分离。

## 7. 执行顺序：以最快得到可信结果为目标

### Phase 0：协议封闭（所有正式运行的前置阻塞）

1. 实现 `metrics_v3.py` 和 synthetic unit tests。
2. 实现统一 raw episode schema、manifest validator、500-tick completeness gate。
3. 用随机/固定策略跑 2 episodes × 20 ticks 的开发 smoke，只验证统计，不产生论文数字。
4. 扩展 final test seed 为 `4000..4099`；冻结测试期 normalizer。

### Phase 1：最快可运行的三条线

1. DCA-CC4：规则映射，无神经网络训练。
2. RSMBRL-CC4：复用现有 categorical UG-CEM。
3. PriorRL-PPO-CC4：复用现有 PPO 与 offline prior cache。

三者先跑 `pilot: 2 repeats × 5 episodes × 100 ticks`，只用于找 bug；通过后进入正式训练/评测。

### Phase 2：主方法和两张消融表

优先完成 LWM-RL、Table 2、Table 3，因为它们共享最多代码且直接决定论文核心主张。不同 variant 可以并行训练，但不能共享 policy checkpoint。

### Phase 3：高开发成本基线

1. TERLA-A4：HGT/PPO。
2. CARL-CC4：SCM + imagined PPO。
3. UAMCTS-CC4：MCTS + 三源不确定性 + progress potential。

### Phase 4：正式全量评测与填表

- 每条方法先完成五个训练 checkpoint，再运行共享 test suite。
- planner 型方法按 method/repeat/episode 并行，单个 episode 内保持环境顺序。
- 推理统一使用 `torch.inference_mode()`、批量 WM rollout、预加载 cache；不在线调用 LLM。
- 完成 7×5×100=3,500 个 Table 1 episodes；Table 2/3 与重复行可复用 LWM-RL Full-Reward 的同一正式结果文件，禁止重复跑后挑更好一次。

## 8. 可分派给子代理的工作包

### W0：统一评测器与协议（最高优先级，阻塞全体）

交付：四指标实现、raw schema、manifest、validator、聚合器、合成测试。
禁止：实现任何策略或修改指标以迁就已有输出。
验收：人工构造 TP/FP、未恢复 incident、五 agent 重复团队奖励、Host Work Fail 的 golden tests 全过。

### W1：主方法 + Table 2/3

交付：formal PPO runner、四个组件消融、三个奖励 predictor/variant、五 seed checkpoint。
前置：W0、offline cache gate。
验收：独立初始化证据、模块调用审计、reward propagation 测试、正式 manifest。

### W2：RSMBRL-CC4

交付：上游来源清单、现有 UG-CEM 的 RSMBRL 封装、validation beta、五次评测。
禁止：把自研方法写成 UAMCTS；测试期更新 normalizer。

### W3：PriorRL-PPO-CC4

交付：PPO-KL loss、LLM action prior adapter、alpha 选择报告、五次评测。
禁止：调用 WM；使用 test 选 alpha。

### W4：TERLA-A4

交付：graph builder、HGT encoder、A4 target mapping、原奖励 wrapper、五次训练/评测。
禁止：把 hidden compromised flag 放进图特征。

### W5：CARL-CC4

交付：reward DAG/SCM、real+synthetic buffer、model rollout、PPO、五次评测。
禁止：实现成决策时 planner；不披露截断 imagined horizon。

### W6：DCA-CC4

交付：alert tokenizer/cluster、abduction/path model、固定 A4 mapping、适配声明、正式评测。
禁止：称为原生 response RL；用真值选目标。

### W7：UAMCTS-CC4

交付：MCTS、hybrid UCB、三源 uncertainty、observable progress scorer、正式评测。
禁止：用未来回报/真值在线计算 Phi；在线 LLM。

### 总代理审查职责

总代理不只看最终 CSV，必须逐项审查：diff、单测、配置、manifest、随机种子、checkpoint hash、raw episode 数、500-tick completeness、测试期参数 hash、指标重算一致性。任何工作包只能写入自己的目录和已声明的公共接口；公共合约改动先由 W0/总代理审批，防止并行代理互相改变定义。

## 9. 正式门禁

| Gate | 通过条件 | 失败处理 |
|---|---|---|
| G0 论文映射 | README 逐条对应论文公式/组件/改动 | 标为 adapted，补齐差异 |
| G1 环境合约 | CC4、A4、500 ticks、五 Blue agent、统一 duration | 作废该 run |
| G2 奖励合约 | reward router 与矩阵一致，Table 3 双通道一致 | 作废并重训 |
| G3 信息泄漏 | hidden field mutation 不改变 action | 阻止正式运行 |
| G4 数据隔离 | test seed 未进训练/校准/早停 | 作废并重训 |
| G5 冻结性 | 测试前后 policy/model/scaler hash 一致 | 作废该评测 |
| G6 完整性 | 每 method 5 repeats，每 repeat 100×500，无缺 episode | 不聚合、不填表 |
| G7 可重算性 | raw log 重算与 summary 差异 < 1e-9 | 修 evaluator |
| G8 结果合理性 | NaN/Inf、precision 无分母、fallback 激增均有审查 | 不静默填 0 |

## 10. 结果文件规范

建议路径：`outputs/formal_v3/<table>/<method>/repeat_<01..05>/`。

每个 repeat 至少包含：

- `config.resolved.yaml`
- `manifest.json`
- `checkpoint.pt` 或不可训练方法的 `policy_spec.json`
- `episodes.jsonl`：100 行 episode summary
- `decisions.jsonl`：逐 decision 审计；允许以内容等价的 `decisions.jsonl.zst` 替代
- `metrics.json`
- `stdout.log`

聚合器输出：`table1.csv`、`table2.csv`、`table3.csv`、`tables.md`、`eligibility_report.json`。只有 `eligibility_report.passed=true` 的结果可以写入论文。

冻结的结果身份补充：Table 1 的 PriorRL 行使用 canonical slug `priorrl_ppo_cc4` 和显示名 `PriorRL-PPO-CC4`；`baselines/priorrl_cc4/` 只是实现/冻结原型目录名。Table 3 的 `full_reward` 不重复训练或复制 manifest，而是只读复用 Table 2 `lwm_rl`（`canonical_run_id=lwm_full`）完全相同的 5 个物理 repeats；若出现独立 `table3/full_reward` 结果，聚合与制图必须 fail closed，防止挑选结果。

Table 2 每个 repeat 的 `manifest.figure_artifacts.training_curve` 必须声明 `cc4_v3_training_curve_v1`、train split、横轴 `environment_steps`、纵轴 `training_objective_reward`、记录数、相对路径与 SHA256；同一 SHA 必须同时绑定在 `manifest.artifact_sha256` 和 validator 生成的 `eligibility_report.input_sha256` 中。曲线步数须为严格递增的非负整数，五个 repeats 与四种组件变体使用完全相同的 step grid。三张表未全部 PASS 或五类图中任一类失败时，不保留部分 CSV/Markdown/SVG/PNG，只输出 `PARTIAL`/`BLOCKED` 审计报告。

Table 3 `fail_only` 只接受已批准的 SmoothL1/Huber（`beta=1.0`）冻结模型；旧 MSE 仅为 `diagnostic_only`。每个正式 manifest 必须绑定冻结 checkpoint/manifest SHA、32 train 与 8 validation 无交集且 test leak 为 0 的 PASS 证据，并记录 OFOX 缓存审计：恰好 6 次调用、6 条生成/验证成功、0 失败、6 条精确 train-only D27、coverage PASS、测试期禁止在线调用，以及冻结 entry-set/logical-manifest SHA。任一字段漂移或出现额外调用即拒绝该正式行。

## 11. 超参、公平性与算力控制

- 环境交互预算对所有 PPO 方法相同；额外 model imagination 不计真实 CC4 interaction，但要报告 synthetic steps 和计算时长。
- 每种方法只做论文必要的小型超参集合；选择规则和 tie-breaker 在看 test 前固定。
- 共享冻结 WM 的方法使用同一 checkpoint；CARL 的 causal reward、UAMCTS progress、Table 3 rewards 是方法必要的独立模型。
- 记录 wall-clock、GPU-hours、峰值显存和决策 latency，作为补充结果，不替代四个主指标。
- 如果正式预算紧张，先缩小 validation grid，不能减少 5 repeats、100 episodes 或 500 ticks 后仍称为正式结果。

## 12. 论文与源码证据冻结

本次阅读的本地论文 SHA256：

| 文件 | SHA256 |
|---|---|
| `UAMCTS.pdf` | `5dffdc568bf6e7fa04a430c44dcac9d293b588c4a2b5211cc5f14324f10af145` |
| `RSMBRL.pdf` | `df594e5b3d2bd36e06d94d96e3cde0c549fa1894a9fe5d26ecd6c4552c91d4d7` |
| `CARL.pdf` | `42e8ef7d912007a2454002300852a54d1dfffad2d9b1099958f7c8d6a84cfae1` |
| `DCA.pdf` | `173fa04a1675e98ce636c2a8b2eda9eae5e3c9e2347f5f19cb4aafe6708e3cb1` |
| `PriorRL.pdf` | `40ab37985422db1a439a22757f30a368b04363b8a9a9b3d59dec3488085e1c71` |
| `TERLA.pdf` | `e2c53cc19c3647029870d9bd9438f9c20d50ab79e728374b8461852e729800ea` |
| `v3_LLM_WM_RL_APT_Response_Final_20260917.docx` | `f6c5dc8d38682434b2f694c83050a6a56e099dc08bec8ab91f48b7b259efefb2` |

官方源码冻结：

- RSMBRL：`https://github.com/sradicwebster/mbrl-lib/tree/uncertainty_guided_planning/uncertainty_guided_planning`，commit `9f97859594f2b0547e01193a8758936090b0b2ec`。
- PriorRL：`https://github.com/yanxue7/RL-LLM-Prior`，commit `13b99c9bba5462b9c84c2a433b4a5ddec046177e`；当前快照未包含论文 policy-based PPO/GFlan 实现。
- 其余四篇：截至本任务书审查，论文正文与官方落地页未给出可直接运行的官方源码；实现时再次检索一次，并将检索日期/链接写入各自 README。

## 13. 当前仓库完成度与立即下一步

已完成：D27/A4 合约、32/8 seed final replay、5-member absolute WM、Full-Reward predictor、动作一致性与 94 项回归基础、offline prior/cache 骨架、UG-CEM 主体。

尚未完成且不能省略：正式四指标 evaluator、完整 offline prior cache、formal PPO、六条基线封装/实现、Table 2/3 独立训练、5×100×500 正式运行、聚合器与 eligibility report。

立即执行顺序：

1. W0 在任何长训练前封闭指标与 manifest。
2. 同时完成 offline prior cache，消除 Ours/PriorRL/UAMCTS 的共同阻塞。
3. 先跑 DCA、RSMBRL、PriorRL pilot；随后主方法与消融。
4. TERLA、CARL、UAMCTS 进入独立实现线。
5. 所有 pilot 通过后才启动正式 5×100×500；正式 test 一次性锁定，不反复窥视调参。

最终原则：宁可表格暂时留空并标明 blocked，也不能使用旧结果、短回合结果、单 seed 结果、测试集调参结果或无法重算的手填数字。
