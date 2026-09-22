# 第三版论文真实 PPO 最终实现与加速计划

日期：2026-09-21
适用仓库：`D:\paper\github-me\aptdetect`
代码实施工作树：`D:\w\trueppo`
实施基线：`6d651d3f5c1359e5a1c9dfede467fdabb816f3a0`
实施分支：`codex/true-ppo-formal`

本文是当前第三版论文实验的实现依据，回答三个问题：最终必须交付什么、现有 AWR
资产哪些可以复用、以及如何以最快但不降低论文门禁的方式完成真实 PPO 和三表结果。
执行者不得以聊天记忆替代本文；发生冲突时，按
`USER_MANDATORY_REQUIREMENTS.md`、`CC4_V3_0917_FINAL_EXPERIMENT_TASKBOOK.md`、
`CHATGPT_HANDOFF_FINAL_EXPERIMENTS.md` 和两个 final manifest 的顺序 fail closed。

## 1. 最终必须达到的目标和交付

### 1.1 最终研究方法

自有方法必须真实实现以下闭环，而不是离线 AWR 或测试期遮蔽：

```text
Blue-observable D27 state
  -> GPT-5.6 Sol 离线 prior 检索，得到 K=6、H=4 候选计划和偏好
  -> 冻结 5-member world model 预测候选未来回报和不确定性
  -> on-policy clipped PPO 选择候选计划索引
  -> 只执行所选计划的第一个 A4 动作
  -> 动作完成后按真实 CC4 response reward 和 duration 记录 transition
  -> 下一决策时刻重新规划
```

固定合约：

- 环境：CybORG Challenge 4 / `FiniteStateRedAgent`；
- 状态：仅 Blue 可观测 D27，禁止 hidden Red truth；
- 动作：A4=`no-op/analyse/remove/restore`，duration=`1/2/3/5` ticks；
- 候选：LWM/LLM 变体 `K=6,H=4`，执行 `plan[0]` 后重规划；
- PPO：真实 reward、`gamma_t=0.99**decision_dt`、GAE lambda `0.95`、clipped PPO；
- 冻结起始超参：lr `3e-4`、rollout `128`、epochs `5`、minibatch `64`、
  clip `0.2`、entropy `0.01`、value coefficient `0.5`、grad clip `0.5`；
- 正式 PPO 必须 fresh 初始化；不得将 provisional 或 AWR 权重直接冒充正式 PPO 权重；
- 正式训练、validation、test 均不得在线调用 LLM/provider；cache/provenance/schema miss
  必须失败，不能 fallback。

### 1.2 三张正式表

必须交付以下 14 个显示行的 paper-eligible 结果：

1. Table 1：`UAMCTS-CC4 (adapted)`、`RSMBRL-CC4`、`CARL-CC4 (adapted)`、
   `DCA-CC4 (adapted)`、`PriorRL-PPO-CC4`、`TERLA-A4`、`LWM-RL`。
2. Table 2：`RL-Only`、`LLM-RL`、`WM-RL`、`LWM-RL`，四个策略独立初始化、
   独立训练、真实交互预算一致。
3. Table 3：`Delay-Only`、`Fail-Only`、`Full-Reward`。Delay/Fail 同时改变 PPO
   真实 transition reward 与 WM 候选 predicted-return reward；Full-Reward 只读复用
   Table 2 LWM-RL 的五个物理 repeats，不重复训练、不挑优。

每行统一报告：

- `CC4 Official Reward`（越大越好）；
- `Operation Failure Penalty`（越小越好）；
- `Recovery Precision`（越大越好）；
- `Recovery Time`（越小越好）。

正式口径固定为 5 个独立 policy repeats，policy seeds `51001..51005`；每个 repeat
最终 test seeds `4000..4099` 共 100 episodes，每 episode 严格 500 ticks。单元格按五个
repeat-level 值计算 mean ± sample std (`ddof=1`)。训练 seeds 为 `1000..1031`，validation
seeds 为 `2000..2007`，calibration seeds 为 `3000..3007`，只允许冻结 normalizer、阈值和
预先声明的超参；test 不得用于训练、归一化、超参、checkpoint 选择或调试。validation、
calibration、test 均不得更新 policy、WM、reward predictor、prior 或 scaler。

### 1.3 正式图和审计交付

完成任务书规定的五类正式图，至少包括同 step grid 的训练曲线和三表结果图。图只能读取
`eligibility_report.passed=true` 的正式结果；三表未全 PASS 时不得把 partial 图冒充论文图。

每个物理 repeat 至少交付：

- resolved config、完整 manifest 和 Git/code/config/artifact SHA；
- fresh policy checkpoint、optimizer/resume checkpoint 与 checkpoint 选择记录；
- `training_curve.jsonl`，横轴为真实 `environment_steps`；
- 100 行 `episodes.jsonl`、逐 decision audit、`metrics.json`；
- provider/cache 命中、真实 environment step、decision count、duration 分布；
- PPO loss/value/entropy/KL/clip fraction/advantage 诊断；
- `eligibility_report.json`，且最终进入论文时 `passed=true`；
- 不可覆盖的逐文件 SHA256 备份及复核报告。

训练曲线必须在 manifest 中声明
`figure_artifacts.training_curve.schema=cc4_v3_training_curve_v1`、`split=train`、
`x_field=environment_steps`、`y_field=training_objective_reward`、`record_count`、相对路径和
SHA256；同一 SHA 必须同时出现在 `manifest.artifact_sha256` 与
`eligibility_report.input_sha256`。四个 Table2 变体和五个 repeats 使用完全相同的 step grid。
三表未全部 PASS 或五类图任一门禁失败时，聚合/制图必须原子 fail closed，不保留可能被误用的
partial CSV、Markdown、SVG 或 PNG，只输出 PARTIAL/BLOCKED 审计报告。

当前严格 paper-table 完成度仍为 `1/14 = 7.14%`：只有 DCA 完整 PASS。工程依赖完成度
不能替代该比例。

## 2. 现有 AWR 与公共资产的可复用边界

### 2.1 已确认的 AWR 身份

当前 `formal_experiments/ours/run_table23_formal.py` 的训练 callback 调用
`offline_bc_awr_update`，读取冻结 replay 做 behavior-constrained offline actor-critic/AWR。
已有曲线明确记录：

```text
online_training_environment_steps = 0
offline_ppo_used = false
training_source = frozen_audited_train_replay
```

因此 AWR repeat 即使 32/32/32 checkpoint 完整、SHA 全匹配，也不是第三版论文要求的
on-policy PPO，不得直接 validation/test 后填入 LWM-RL、Table 2 或 Table 3。

### 2.2 可以直接复用于正式 PPO 的冻结资产

以下资产不需要重建，必须只读绑定并复核 SHA：

1. final train/validation replay：
   `outputs/formal_replay_final_20260917/`；train 32 seeds、43,297 decisions，validation
   8 seeds、10,796 decisions；manifest 为 `docs/FINAL_REWARD_REPLAY_MANIFEST.json`。
2. final 5-member absolute world model 和 Full-Reward predictor：
   `outputs/world_model_final_20260917/`；manifest 为
   `docs/FINAL_REWARD_MODEL_MANIFEST.json`，两者质量门均 PASS。
3. Delay-Only 和 Fail-Only 已冻结 predictor/provenance 资产；进入真实 PPO 前仍需按当前
   commit 做 identity preflight，但不重复训练模型。
4. GPT-5.6 Sol final offline prior cache：
   `outputs/lwm_rl_final_20260917/prior_cache_final/manifest.json`；
   `target_size=2000`、`verified_entries=2000`、`failed_entries=0`、
   `exact_model_id=openai/gpt-5.6-sol`、status PASS。train 下另有经过审计的 6 条
   supplement，provider 调用只发生在离线构建阶段。
5. 从上述 cache 冻结出的 train-only nearest-prototype prior、validation-only radius、coverage
   和 provenance。正式 runtime 可以复用 `FrozenPrototypePriorAdapter`，但必须把 prototype
   identity 追溯绑定到 final cache manifest、entry-set SHA、supplement manifest 和 exact model/
   prompt/schema；超过冻结 radius 必须 fail closed。
6. D27/A4/action-duration/target resolver、final reward bookkeeping、统一 CC4 adapter、正式
   evaluator、manifest/validator/backup 框架。
7. `ppo_core.py` 的 46D candidate actor-critic、duration-aware GAE、clipped PPO loss；
   `ppo_training.py` 的 async decision buffer、real reward transition 与 optimizer；
   `run_b4_tiny_pipeline_smoke.py` 中真实 CC4 rollout、非终止 next-state critic bootstrap、
   terminal `next_value=0` 的实现模式。
8. `posterior_features.py`、`table2_variants.py`、`lwm_runtime.py` 的 K6/H4、46D 特征、
   candidate plan、prior、WM return/uncertainty 组装逻辑。
9. AWR 的 stage journal、原子 checkpoint、SHA/provenance、resume、validation/test/hash/integrity
   编排代码。仅复用工程机制，不复用 AWR 训练语义。

### 2.3 AWR checkpoint/输出的允许用途

AWR repeat1–5 只能作为：

- verified dependency / non-paper development evidence；
- 检验特征、checkpoint、resume、manifest、备份链的输入；
- train/validation-only 的诊断参照，比较真实 PPO 是否学到不同策略；
- 故障恢复证据。

禁止用途：

- 不得直接填三表或生成正式图；
- 不得以 `500 ticks` 评估回合伪称 PPO 训练 environment steps；
- 不得把 AWR checkpoint 重命名成 PPO checkpoint；
- 不得在 test 上比较 AWR/fresh 后再选择；
- 正式 PPO 必须 fresh 初始化，因此本轮不把 AWR checkpoint 用作 primary 初始权重。

AWR 原始输出和备份均不得删除、覆盖或 reset。当前 repeat5 继续由固定 Luna-high 窗口每
15 分钟只读监控；完成后先验收、不可覆盖备份，再由主窗口决定非破坏性归档，绝不自动
进入 validation/test。

## 3. 最快合规路线

### 3.1 总体策略

最短关键路径不是重建 prior/cache/WM，也不是继续 AWR，而是：

```text
复用已通过的真实 PPO core + tiny live rollout
  -> 实现独立 true-PPO formal callback/runner
  -> 绑定现有 2000+6 prior/prototype trust chain
  -> 最小定向门禁
  -> 首先启动 LWM-RL repeat1 正式 train
  -> 资源实测后并行 repeats
  -> validation 冻结 checkpoint
  -> 一次性 100x500 test
  -> 复用同一 runner 并行完成 Table2/3
  -> 基线仅做正确公平运行，不额外调优
```

代码在独立 worktree `D:\w\trueppo` 上实施，避免修改正在运行 AWR 的
`C:\Users\25453\.codex\worktrees\3cf0\aptdetect`，也不触碰主仓库中用户拥有的 dirty docs。

### 3.2 真实 PPO runner 设计

优先新增独立模块，不把 AWR 语义静默改名：

- `formal_experiments/ours/formal_ppo_runtime.py`：variant-aware 候选/动作上下文、mask、
  offline prior/prototype trust chain、WM/reward routing；
- `formal_experiments/ours/formal_ppo_rollout.py`：真实 CC4 500-tick episode rollout、async
  action completion、real reward、next-context bootstrap、decision audit；
- `formal_experiments/ours/run_table23_true_ppo.py`：preflight/train/validation/prepare_test/
  test/hashes/integrity 阶段与 CLI；
- 对 `ppo_core.py`、`ppo_training.py` 只做必要的 K/mask 泛化，并保持现有 API 回归兼容；
- 对 AWR runner 只加明确 legacy/non-paper 标识或共享只读 helper，不删除其实现。

训练 callback 必须：

1. 每个 repeat fresh 构造 policy/optimizer，并绑定 policy seed `51001..51005`；
2. 依次运行 train seeds `1000..1031`，每 seed 严格 500 global ticks；一轮共固定
   `32*500=16,000` real environment steps，约等于现有 replay 的 43,297 decision transitions；
3. 所有 PPO 类方法以同一 16,000 global-tick budget 作为第一版冻结正式预算；不得因动作
   duration 不同而强行补齐不同数量 decisions；同时记录实际 decision count；
4. policy 在收集一个 on-policy rollout 期间保持冻结；只用该 policy 产生的 old log-prob；
   在安全 rollout/episode 边界执行 PPO update，严禁混合跨 policy-version transitions；
5. pending async action 完成后才写 transition；真实 response reward 累积到完成点；terminal
   next value=0，非 terminal 用完成后的 next observable context 计算 critic value；
6. checkpoint 在累计 seeds 8/16/24/32（环境 step 4k/8k/12k/16k）原子保存，曲线 step grid
   对所有 Table2/3 PPO 行一致；
7. checkpoint 必须保存 policy、optimizer、policy version、RNG、已完成 seeds、真实 env steps、
   decisions、update metrics、artifact/domain SHA；resume 只能从完全匹配身份继续；
8. `provider_calls=0`，正式 runtime 不得携带可用 live generator。

任务书“每方法五个训练 checkpoint”按五个独立 policy repeats 的最终 selected checkpoint
执行：每个方法最终必须有 policy seeds `51001..51005` 对应的五个正式 checkpoint。每个 repeat
内部的四个 4k/8k/12k/16k validation candidates 是 checkpoint 选择候选，不代替五个独立 repeats。

16,000 ticks 是最快且与冻结 32 train seeds 自然一致的合规预算，并低于历史 B4 的
100k-decision formal 上限。若 LWM repeat1 的 validation 明确显示未学习/不稳定，只能在看 test
之前由主窗口把所有 PPO 行统一提升到预先声明的第二档预算；不得只给 LWM 或某个基线增加
真实交互预算，也不得看 test 后改预算。

### 3.3 四个 Table 2 variant 的严格隔离

- `RL-Only`：直接 A4 categorical PPO。实现可用 K=4 一一对应的 action rows，但 manifest
  必须声明每个 index 唯一映射 A4、无 LLM prior、无 WM 调用，不能伪装 K=6 计划。四个
  action index 必须通过统一 target resolver 和 CybORG adapter，严格采用固定 duration
  `1/2/3/5`，逐 decision 记录 requested A4、executed A4、actual duration、target、fallback 和原因。
- `LLM-RL`：K=6 frozen teacher plans/prior + PPO；WM return/uncertainty 输入严格为禁用值，
  不加载或调用 WM evaluator。
- `WM-RL`：固定、确定性、非 LLM K=6 plan generator + frozen WM + PPO；不读取 teacher prior。
- `LWM-RL`：GPT-5.6 Sol frozen prior/prototype + frozen WM + PPO。

四者独立 fresh 初始化、相同 env-step/checkpoint grid、相同 Full-Reward 和 test seeds。必须保存
候选去重前/后列表、candidate mask、行为选择概率、selected index、plan[0]、requested/executed
A4 和 fallback。相同首动作的不同 H4 计划允许保留；不得偷偷聚合 action mass。

### 3.4 Table 3 双奖励通道

- Delay-Only 读取冻结 Delay predictor，PPO real reward 只含 delay；
- Fail-Only 读取批准的 SmoothL1/Huber beta=1.0 predictor，PPO real reward 只含 failure；
- 两者的 WM candidate predicted return 必须使用同语义 predictor；
- Full-Reward 不产生独立物理 run，只读引用 source=`table2/lwm_rl`、
  `canonical_run_id=lwm_full` 的五个 repeats；出现独立 `table3/full_reward` 物理结果时必须
  fail closed；
- runner/preflight 必须拒绝 reward-mode、predictor、normalizer、checkpoint 或 provenance 漂移。

Delay/Fail preflight 还必须绑定与 Full predictor 相同的模型架构、输入、normalizer、optimizer、
学习率、batch size、固定 50 epochs、模型随机种子、32 train/8 validation 且 overlap/test leak=0；
绑定 predictor checkpoint/manifest SHA 和质量门。Fail-Only 只接受未加权 SmoothL1/Huber
`beta=1.0`，旧 MSE 只能标为 diagnostic-only。OFOX 审计必须为恰好 6 calls、6 verified、
0 failed、6 条精确 train-only D27，并绑定 entry-set/logical-manifest SHA；任一漂移即拒绝。

### 3.5 prior cache 的正式绑定

prior adapter 在启动前一次性验证：

- final manifest status PASS、target/verified=2000、failed=0；
- exact model=`openai/gpt-5.6-sol`、prompt version、registry、state-bank/replay SHA；
- 2000-entry set SHA、prototype file/internal SHA、coverage/provenance SHA；
- 6-entry supplement 恰好 6 calls/6 verified/0 failed，全部 train-only D27；
- validation 只冻结检索 radius，不进入 prototype 内容；
- query agent-local，超 radius 或任何字段/hash/schema miss 立即失败；
- runtime counters 始终 provider calls=0，不能存在在线 fallback。

### 3.6 checkpoint 选择和正式测试

每个 repeat 的 4 个 checkpoint 只在 validation seeds `2000..2007` 上按预先冻结的主指标/
tie-breaker 选择。选择后写入只读 selection manifest 并绑定 checkpoint SHA。只有五个 repeat
全部通过 train/validation 门，才运行 test seeds `4000..4099`；test 只运行一次，策略、WM、
reward model、prior/prototype、normalizer 前后 hash 必须相同。

calibration seeds `3000..3007` 仅运行只读 calibration 阶段以核验/冻结 normalizer、检索 radius、
阈值和预先声明超参身份，不参与 policy gradient 或 checkpoint 排名；validation 只做 checkpoint
选择，不更新 calibration 资产。calibration manifest 必须记录输入 SHA、冻结输出 SHA 和零 test leak。

### 3.7 最小但充分的测试门

为节省时间和 token，不跑无关全套测试。正式启动前只要求：

1. 现有 PPO core/training 定向回归；
2. mask/K4/K6、old-log-prob、ratio clip、duration GAE、terminal/nonterminal bootstrap；
3. K6/H4、plan[0]、candidate before/after、variant isolation；
4. prior exact identity/prototype radius/cache miss/provider disabled；
5. reward real/predicted 双通道和 split leakage；
6. manifest/resume/checkpoint/hash/equal-budget fail-closed；
7. 一个 20-tick 离线 tiny CC4 smoke，要求真实 env steps>0、policy update、provider calls=0；
8. 一个独立 500-tick train-seed dry/pilot 仅验证完整 episode、资源和恢复点，不能填表。

所有测试和 pilot 通过后立即启动 LWM-RL repeat1，不等待非关键重构或全套回归。

### 3.8 资源与并发

- 优先把 GPU/显存给 LWM-RL/WM rollout/PPO；CC4 环境与日志使用 CPU；
- 当前 AWR repeat5 未结束时只做代码实现和轻量定向测试，不并发第二个重训练 writer；
- 首个真实 PPO 500-tick pilot 记录 RAM、VRAM、CPU、速度；若 RAM/VRAM 余量满足硬门，再把
  五个 repeat 以安全的 2-way 或更高并行执行，输出目录和 writer 完全隔离；
- Table2/3 变体在主方法 repeat1 启动后可并行，但 LWM 主方案优先；
- baseline 仅在不抢占主方案资源时并行，禁止为了提高 baseline 效果反复调参；
- 每个完整 checkpoint/repeat 立即做不可覆盖 SHA 备份，再开启下一阶段。

## 4. 实施工作包与验收

### WP1：真实 PPO 代码实现（Luna xhigh）

范围：只改 `D:\w\trueppo`，实现第 3.2–3.5 节；不运行正式 train/test，不修改 AWR
canonical 输出，不删除/移动/覆盖资产，不 reset/clean/restore，不推送。

验收：

- Git diff 仅包含声明的 code/tests/docs，worktree 无意外文件；
- 真实 runner 不导入/call `offline_bc_awr_update`；
- train 使用真实 CC4 environment，`online_training_environment_steps>0`；
- PPO batch 包含 old log-prob、real reward、duration、next value 和 policy-version provenance；
- prior/provider、split、variant、reward、budget、resume 全部 fail closed；
- 定向测试通过，输出测试清单和耗时；
- 提交到 `codex/true-ppo-formal`，报告 commit 与逐文件 SHA。

### WP2：主窗口独立复核

主窗口逐项审查 diff、算法语义、tests、manifest 字段、资源门和命令。若发现 AWR、offline
reward、`next_value=0` 非终止、在线 provider、hidden truth、test leakage、非 fresh init 或预算不等，
立即拒绝，不启动正式实验。

### WP3：Luna xhigh 运行最小门禁与正式实验

主窗口审核通过后另行给出冻结 commit、精确命令、输出目录和资源门。执行顺序：20-tick
offline smoke -> 500-tick train pilot -> LWM repeat1 train -> 主窗口审核/备份 -> 其余 repeats
安全并行 -> validation -> frozen selection -> final test。每个任务仍由固定 Luna-high 监控窗口
每 15 分钟只读检查，正常静默，完成/异常立即交主窗口。

## 5. 最终判定

最快合规路线会最大化复用 AWR 周边工程和所有已冻结 final 资产，但不会复用 AWR 的训练
身份或权重来伪造 PPO。只有真实 on-policy CC4 interaction、fresh PPO、validation-only selection、
100x500x5 test、完整 hash/provenance/backup 且 eligibility PASS 的结果，才算最终交付。
