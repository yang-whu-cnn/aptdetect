# UG-CEM-APT 复现、领域适配与公平对比总任务书（v2.0）

> 仓库：yang-whu-cnn/aptdetect  
> 稳定备份分支：me  
> 实验分支：ug-cem-apt  
> 当前阶段：Gate A3 之前的总方案复核  
> 最后更新：2026-09-15  
> 本文件是后续实现、审核、实验和论文撰写的唯一总路线图。若后续方案发生实质变化，必须先更新本文件，再改实现。

---

# 0. 本次 v2.0 复核为什么必须做

Step 2、Step 3、Gate A1、Gate A2 已经完成。进入真实 CybORG / CAGE Challenge 4 之前，再次对照：

1. Webster & Flach, Risk Sensitive Model-Based Reinforcement Learning using Uncertainty Guided Planning；
2. 仓库内 UG 官方源码；
3. 当前第二章已有实现；
4. CAGE Challenge 4 官方动作、观测、奖励和多智能体规则；

发现旧任务书虽然方向正确，但仍有几个会直接影响最终效果和论文可信度的缺口：

- 旧任务书把正式状态长期锁死为 D=8，但 CC4 的 mission phase、traffic policy、blocked state 会直接影响奖励和 control_traffic 的价值；
- 旧 world model 的 ensemble 成员使用同一批 minibatch，没有显式 bootstrap，弱化了 epistemic uncertainty 的可信度；
- 旧 replay / checkpoint 的动作 ID 已与新 5 动作语义不一致，正式实验必须废弃并重新采集；
- 旧 plan_eval 仍包含旧 cost / delay / early-warning 兼容逻辑，不能继续作为正式计划回报；
- 旧 llm_prior 仍依赖 monitor / light_evidence / heavy_evidence / local_mitigate / strong_mitigate，不能直接用于新方案；
- 旧 official evaluator 仍使用 0->Monitor, 1/2->Analyse, 3->Remove, 4->Restore 的旧映射；
- CC4 中 Analyse/Remove/Restore 具有多 tick duration，agent 在动作完成前不能重新发起新动作；
- BlockTraffic 是持久化交通控制的一部分，CC4 同时提供 AllowTraffic；若只会 block 而无法处理 stale block，可能产生明显的 green availability penalty；
- BlueFixedActionWrapper 的 action index 对不同 agent 不能假定同义，必须使用 action_labels + action_mask；
- 正式 CC4 有 5 个 Blue agent，blue_agent_4 还负责多个 subnet；只做 region0-3 会降低正式 team reward；
- 当前世界模型只预测状态，而正式 CC4 reward 依赖 mission phase、green availability、red impact 等，不能继续只用 risk_proxy 手工近似正式回报。

因此 v2.0 的核心目标是：

> 在不改变论文核心思想的前提下，把“能跑通”的实现升级为“状态信息充分、模型可校准、动作真实可执行、回报与官方目标一致、对比公平、可以稳定做最终论文实验”的实现。

---

# 1. 论文与官方环境中必须保持不变的事实

## 1.1 UG-CEM 原论文核心

UG-CEM 的核心不是重新设计 reward，也不是额外训练一个策略网络，而是：

- 用 bootstrap ensemble dynamics models 近似环境 epistemic uncertainty；
- 用 CEM 规划未来动作序列；
- 对 ensemble 未来状态预测分歧较大的 action sequence 加惩罚；
- 引导规划器倾向模型更有把握的未来区域；
- 风险敏感性体现为 risk-return trade-off，而不是保证 reward 一定更高。

领域适配必须保留：

- ensemble model disagreement；
- fixed-member / TS∞ 风格 rollout；
- deterministic model mean rollout；
- state-trajectory disagreement uncertainty；
- running state / horizon normalizer；
- score = predicted_return - beta * uncertainty / (cem_iteration + 1)。

## 1.2 CC4 正式环境关键事实

正式 CC4 必须按以下事实设计：

- Monitor 是自动发生的默认监测行为；
- Sleep 才是显式“本 tick 不主动执行动作”；
- Analyse duration=2；
- Remove duration=3；
- Restore duration=5；
- BlockTraffic duration=1；
- Sleep duration=1；
- BlockTraffic 需要 from_subnet / to_subnet；
- AllowTraffic 用于撤销已有 firewall block；
- BlockTraffic 可能导致 green communication failure penalty；
- reward 随 mission phase 变化；
- Blue agents 在动作完成前不能重新发起另一个动作；
- 正式环境有 5 个 Blue agent；
- BlueFixedActionWrapper 提供 action_labels 和 action_mask，不能假定一个整数 action index 在不同 agent 上语义相同。

---

# 2. 最终论文方法与对比目标

最终至少比较：

- LWM-RL（Ours）
- UG-CEM-APT
- CEM-APT（beta=0）

最终需要证明的不是“某一次跑分最高”，而是：

1. Ours 在相同 state / action / world model / reward objective / environment 下取得更高或更稳定的 official return；
2. UG uncertainty 的作用可以通过 beta=0 与 beta>0 消融解释；
3. Ours 中 LLM prior、world-model foresight、PPO posterior 的作用可以通过 ablation 解释；
4. 所有方法都没有未来标签、test-seed tuning 或隐藏 heuristic；
5. 结果能在 paired seeds 下复现。

重要说明：

> 本任务书只能通过更合理的状态表示、更可靠的世界模型、更正确的 CC4 adapter、更严格的数据划分和调参流程来提高成功概率，不能在实验前保证 Ours 一定胜出。任何为了“保证赢”而只给 Ours 增加额外信息、heuristic 或 test-set tuning 的做法都禁止。

---

# 3. 全局公平性与性能原则

所有正式方法必须共享：

- 当前时刻可见的原始 CC4 observation；
- FormalStateEncoder；
- 高层动作契约；
- target resolver；
- official action adapter；
- train / validation / calibration / test seed 列表；
- dynamics ensemble architecture；
- world-model checkpoint；
- state normalizer；
- shared reward model / plan-return adapter；
- episode 配置；
- evaluation metrics。

只允许方法之间不同的部分：

LWM-RL：
- LLM candidate prior；
- evidence construction；
- PPO posterior plan selection。

UG-CEM-APT：
- categorical CEM；
- trajectory uncertainty penalty；
- MPC probability warm-start。

CEM-APT：
- 与 UG-CEM 完全相同，但 beta=0。

正式测试前必须冻结：

- state schema；
- action mapping；
- target selection；
- world model；
- reward model；
- beta；
- PPO checkpoint；
- LLM prompt / model / generation config；
- evaluation seeds。

---

# 4. 代码分层与“不得直接用于正式实验”的旧模块

## 4.1 Legacy src 保留

以下旧代码继续保留，作为历史实现和回滚参考：

- src/action_space.py
- src/cc4_client.py
- src/state_summary.py
- src/world_model.py
- src/plan_eval.py
- src/llm_prior.py
- src/env_region.py
- src/ppo_posterior.py
- experiments/run_region_official_cyborg_eval.py

原则：

> 不为了新方案强行改旧实现；正式方案优先新增 shared / formal 模块。

## 4.2 当前不能直接用于最终实验的原因

src/action_space.py：
- 仍是旧五档动作。

src/llm_prior.py：
- 仍生成 monitor / light_evidence / heavy_evidence / local_mitigate / strong_mitigate。

src/plan_eval.py：
- 仍依赖旧 action_cost / action_intensity；
- 存在旧 early-warning 兼容函数；
- 当前 risk/cost/delay surrogate 不是 official reward。

src/env_region.py：
- 仍从旧 ACTION_SPACE 建 world model / prior / PPO。

旧 official evaluator：
- action mapping 仍是 1/2 都映射 Analyse；
- 只控制指定 target_agent，其余 agent 默认 Monitor；
- 不支持新的 control_traffic；
- 不能作为最终公平比较脚本。

现有 replay / WM / PPO checkpoint：
- 由旧动作语义生成；
- 不能作为新五动作正式模型。

---

# 5. 状态表示：开发 D=8 与正式状态必须分开

## 5.1 Local development state

Gate A2 local simulator 继续使用当前 D=8：

1. cnt_auth_fail
2. cnt_port_scan
3. cnt_proc_spawn
4. cnt_outbound_conn
5. avg_severity
6. unique_src
7. unique_dst
8. risk_proxy

用途：

- unit test；
- local smoke；
- 检查动作 transition；
- 不作为最终 CC4 状态充分性的证明。

## 5.2 正式 CC4 状态存在的信息缺口

仅 D=8 会遗漏：

- mission phase；
- 当前 blocked traffic 状态；
- 当前 communication policy；
- block 与 policy 是否冲突。

这些变量会直接影响 official reward 和 traffic-control 决策，因此正式状态不能未经验证就永久锁死 D=8。

## 5.3 v2.0 默认正式候选：D=13

正式候选状态：

基础 8 维：
- 原 D=8 全部保留。

增加 5 维：
- mission_phase_1
- mission_phase_2a
- mission_phase_2b
- blocked_ratio
- policy_mismatch_ratio

其中：

policy_mismatch_ratio =
当前 firewall block state 与当前 mission communication policy 不一致的比例。

设计理由：

- phase 是 official reward 非平稳性的直接上下文；
- blocked_ratio 表示当前交通控制状态；
- policy_mismatch_ratio 直接反映可能造成 green penalty 的错误阻断或缺失阻断；
- 不加入未来 red ground truth；
- 不加入 hidden simulator state。

## 5.4 状态冻结规则

Gate A4 用 validation seeds 比较：

- D8
- D13

比较指标：

- one-step state prediction；
- H=4 open-loop rollout error；
- reward prediction error；
- official validation return；
- planning stability。

默认优先 D13。

只有 D8 在 validation 上不劣且明显更稳定/更快时，才允许保留 D8。

正式 test seeds 不能参与这个决定。

---

# 6. 高层动作契约

Gate A1 已完成并冻结：

0 = no_op
1 = analyse
2 = control_traffic
3 = remove
4 = restore

语义：

- no_op：no operation / monitor，仅不主动处置；
- analyse：intrusion investigation；
- control_traffic：traffic containment / blocking；
- remove：user-level compromise removal；
- restore：host reimaging。

动作 ID 只是 categorical index，绝不表示 0<1<2<3<4 的连续强度。

当前 CC4 计划映射：

- no_op -> Sleep
- analyse -> Analyse
- control_traffic -> BlockTraffic / traffic-control lifecycle adapter
- remove -> Remove
- restore -> Restore

Monitor 继续由 CC4 自动发生，不把 Monitor 当显式 no-op action 发出。

---

# 7. 动作 duration 与正式 transition 定义

这是 v2.0 新增的关键约束。

## 7.1 不再把每个 global tick 简单当成一个 planner transition

CC4 中 Analyse/Remove/Restore 跨多个 tick。

如果世界模型仍训练：

s_t, a_t -> s_{t+1 global tick}

但状态里又没有 ongoing action / remaining duration，就会产生明显的 non-Markov 问题。

## 7.2 首选方案：decision-epoch transition

对每个 Blue agent：

1. 只有 agent 可以发起新动作时才调用 planner；
2. planner 选择一个高层动作；
3. official adapter 发出真实底层 action；
4. 环境继续 global ticks；
5. 等该 agent 再次可选择动作时，形成 next decision state；
6. 将这整个区间记为一个 high-level transition。

Replay 保存：

- s_t
- high_level_action
- accumulated_official_reward
- s_next_decision
- decision_dt_ticks
- done

这样：

- duration 被环境真实体现；
- 不需要人为 lambda_delay；
- world model 学的是“一个高层动作完成后的下一决策状态”；
- H=4 表示未来 4 个高层决策，而不是机械 4 个 global ticks。

## 7.3 备选方案

如果 A3 实测发现 wrapper 无法稳定获取 decision availability，则必须：

- 将 current_action / remaining_ticks 加入 formal state；
- 明确 tick-level world model。

A3 未验证前，不允许假设 duration 可以忽略。

---

# 8. control_traffic 生命周期是 A3 的强制决策项

CC4 同时存在：

- BlockTrafficZone
- AllowTrafficZone

而当前论文高层动作只有 control_traffic。

如果 block 持续存在而 planner 永远没有 unblock 能力，可能造成长期 green service penalty，严重影响效果。

A3 必须实测：

1. block 是否持续到 AllowTraffic；
2. mission phase 改变后 stale block 是否保留；
3. action label 如何编码 subnet pair；
4. blue_agent_0..4 各自可见哪些 subnet pair。

A3 必须在以下两种方案中冻结一个：

方案 A（优先）：
- 高层仍只有 control_traffic；
- 将其定义为“traffic-control management / containment”宏动作；
- shared adapter 根据当前可见 block/policy/threat evidence，在 BlockTrafficZone 与必要的 AllowTrafficZone cleanup 之间做确定性解析；
- 该逻辑 Ours / UG / CEM 完全共享；
- 论文明确说明这是 5-action high-level abstraction 对 CC4 firewall lifecycle 的领域适配。

方案 B：
- 高层 control_traffic 严格只 BlockTraffic；
- 必须证明 stale block 不会导致不可恢复的明显 penalty，或存在 CC4 自动恢复机制。

如果 B 的实测不成立，不允许为了保持“block-only”而牺牲整个正式控制系统。

---

# 9. Shared target resolver

高层 planner 只选 action type，不直接选 host/subnet，因此正式实验必须有共享 target resolver。

要求：

- Ours、UG、CEM 完全相同；
- 只能使用当前 observation；
- 不使用 true red location；
- 不使用未来 reward；
- 不使用测试标签；
- tie-breaking 必须确定性。

Host action：
- Analyse / Remove / Restore；
- 从当前 Blue observation 中按 malicious process / malicious network evidence 给 host 排序；
- 相同分数按固定 host order 选；
- 不允许 planner-specific host heuristic。

Traffic action：
- 从合法 subnet pair 中选；
- 结合当前 network evidence、communication policy、blocked state；
- 只在 action_mask 为 valid 的底层 action 中选择；
- 不直接用 action index 的数值含义。

若目标不存在：
- 记录 requested high-level action；
- 执行明确的 fallback（默认 Sleep）；
- 记录 fallback_reason；
- replay 中保留 requested 与 executed 两者。

---

# 10. 世界模型：v2.0 正式设计

## 10.1 当前问题

旧 DynamicsEnsemble：
- 成员初始化不同；
- 但训练时所有 member 使用同一个 minibatch index；
- 不是真正显式 bootstrap dataset；
- 原论文的 epistemic uncertainty 依据因此不够强。

## 10.2 正式 dynamics ensemble

正式模型仍共享给 Ours / UG / CEM。

默认：

- ensemble_size M=5；
- 两层 MLP；
- hidden=128；
- probabilistic diagonal Gaussian output；
- 每个 member 独立初始化；
- 每个 member 独立 bootstrap sampling；
- train-only state normalization；
- state std 下限 eps；
- 每个 member 独立 optimizer；
- checkpoint 保存 model seed 与 bootstrap seed。

禁止：

- 只给 UG 使用 bootstrap；
- Ours 与 UG 使用不同 world-model data；
- test data 更新模型。

## 10.3 输入 / 输出 normalization

正式状态各维尺度不同：

- event count 可大于 1；
- risk / ratio 在 0..1；
- phase 是 one-hot。

因此正式模型必须：

- 对连续 state feature 用 train-set mean/std 标准化；
- one-hot phase 可以保留原值或统一进入相同 normalizer；
- normalizer 只在 train split 拟合；
- checkpoint 与 normalizer 一起保存。

## 10.4 absolute vs delta prediction

首选仍保持当前“预测 next state”的结构，减少一次大范围算法改动。

但 A4 validation 必须增加：

- absolute next-state target；
- delta-state target；

二者的小规模对比。

若 delta 在 H=4 rollout error 上稳定更好，则所有方法一起切换；不能只给某一个 baseline 使用。

## 10.5 模型质量 Gate

世界模型不能“train 完就算通过”。

必须在 held-out validation episodes 上记录：

- one-step RMSE / MAE；
- Gaussian NLL；
- H=2/H=4 open-loop rollout RMSE；
- per-feature error；
- persistence baseline error；
- ensemble disagreement 与实际 prediction error 的 Spearman correlation；
- high-error sample detection AUROC 或分位数 calibration。

最低接受条件：

- one-step 与 H=4 error 均应优于 persistence / naive baseline；
- ensemble disagreement 对 error 至少应有稳定正相关；
- 若 uncertainty 与 error 完全无关，不允许进入正式 UG 主实验。

---

# 11. Shared reward model / predicted return

## 11.1 为什么旧 risk surrogate 不够

正式 CC4 reward 受到：

- mission phase；
- green local work failure；
- service access failure；
- red impact / compromise；
- restore availability loss；
- traffic block 导致的 green communication failure；

共同影响。

因此只使用：

-risk_proxy - action_cost - delay

作为正式 predicted return，会与 official objective 明显错位。

## 11.2 v2.0 设计

A4 新增 shared reward model，所有方法共用。

输入候选：

- s_t
- high-level action
- predicted s_{t+1}
- decision_dt

输出：

- 从当前 decision epoch 到下一 decision epoch 的 accumulated official reward。

训练 target：

- CybORG 官方环境实际返回 reward 的区间累积。

首选 loss：

- Huber loss；
- 同时记录 MAE / RMSE。

计划回报：

G(plan) =
对 ensemble member 的 predicted accumulated official reward 做折扣累计，再对 member 求均值。

## 11.3 Duration-aware discount

不再人为加 lambda_delay。

若 action duration 为 d ticks：

- 使用 tick-level gamma；
- 按 cumulative ticks 折扣；
- duration 的代价由真实环境 reward + 时间折扣自然体现。

这样可以避免：

- official environment 已经惩罚 Restore/BlockTraffic；
- planner 又手工重复扣 cost/delay；

造成 double counting。

---

# 12. 数据与 seed 协议

## 12.1 四类 seed 必须分开

必须在正式训练前生成并提交固定列表：

- train_seeds
- validation_seeds
- calibration_seeds
- test_seeds

用途：

train：
- replay collection；
- world model / reward model / PPO training。

validation：
- D8 vs D13；
- absolute vs delta；
- beta；
- LLM/PPO hyperparameters；
- candidate count；
- population size 等。

calibration：
- UG uncertainty normalizer warm-up；
- final sanity check。

test：
- 只做一次冻结后的正式结果。

## 12.2 Replay 必须重新采集

旧 replay 与新 action semantics 不兼容，正式全部标记 legacy。

正式 replay schema 至少包含：

- episode_seed
- agent_id / region_id
- decision_index
- global_tick_start
- global_tick_end
- decision_dt
- state
- requested_high_level_action
- executed_low_level_label
- executed_target
- action_success / fallback_reason
- accumulated_official_reward
- next_state
- done

## 12.3 行为策略与 action coverage

不能纯随机后发现 Remove / Restore / BlockTraffic 几乎没有有效样本。

正式 collection policy 使用：

- high-level stratified exploration；
- 保证五类 action 都有最低样本覆盖；
- target resolver 与正式 evaluator 完全相同；
- action_mask 过滤 invalid low-level target；
- 同时保留自然状态分布，不做 test-label directed sampling。

必须报告：

- 每类 requested action 数；
- 每类 actual executed action 数；
- success / fallback rate；
- state coverage；
- region/agent coverage。

---

# 13. LWM-RL（Ours）正式迁移要求

旧 src/llm_prior.py 与新动作空间不兼容，因此正式 Ours 必须新增新 prior module，而不是继续兼容旧 action names。

## 13.1 LLM candidate prior

固定：

- action vocabulary = A1 五类动作；
- H=4；
- K 候选计划；
- prompt version；
- LLM model/version；
- generation temperature；
- max tokens；
- parser；
- fallback policy。

输出必须经过 validator：

- 只允许 0..4；
- 长度必须 H；
- 去重；
- 无效计划拒绝；
- 候选不足时使用固定模板补齐；
- 记录 raw response 与 parsed result。

## 13.2 Candidate diversity

为了避免 LLM prior 全部生成近似计划，记录：

- unique plan ratio；
- per-position action entropy；
- candidate duplicate rate。

如果 diversity 长期过低，优先优化 prompt / deterministic mutation，而不是在 test seeds 上调。

## 13.3 Evidence

正式 evidence 不再直接使用旧 plan_eval。

候选计划至少包含：

- G：shared predicted official return；
- U_LWM：ensemble member return disagreement；
- C：LLM checkpoint consistency；
- O：operational burden feature。

O 首选定义：

- 由 action duration 与 traffic-policy mismatch 等可观测 operational burden 构成；
- 只作为 PPO 输入 feature；
- 不自动等于额外 reward penalty。

## 13.4 Evidence normalization

G/U/C/O 的数值尺度可能差异很大。

正式 PPO 前增加 train-only running normalizer 或固定 feature scaling。

必须避免：

- 用 test episodes 拟合；
- candidate 数变化导致尺度漂移。

---

# 14. Gate A 详细实施

## A1 — Shared Action Contract

状态：PASS。

审核记录：
- docs/step3-A1.md

完成：
- 五动作语义；
- ID；
- CC4 action type placeholder；
- duration metadata；
- legacy isolation。

## A2 — Local-online Action Adapter

状态：PASS。

审核记录：
- docs/step3-A2.md

完成：
- LocalThreatState；
- 五类不同 transition semantics；
- partial observability；
- risk_proxy from observable events；
- reward=0 Gate B placeholder；
- 31 个 Gate A tests。

## A3 — Official CybORG / CC4 Action Adapter

状态：CURRENT。

### A3 目标

把五个高层 action 映射到真实 CC4 可执行 action，冻结 target selection、duration/availability 处理和 traffic-control lifecycle。

### A3 子任务

A3.1 Wrapper probe
- 初始化正式 EnterpriseScenario + BlueFixedActionWrapper；
- 枚举 blue_agent_0..4；
- 保存 action_labels；
- 保存 action_mask；
- 保存 hosts；
- 保存 subnets；
- 明确 padding 行为。

A3.2 no_op
- high-level no_op -> Sleep；
- 验证 Monitor 自动执行；
- 不把 Monitor 当显式 no-op。

A3.3 Host actions
- Analyse / Remove / Restore；
- 从 valid labels 中按 shared host resolver 选 target；
- 若 host 不存在或被 padding，不能选 invalid index。

A3.4 control_traffic
- 解析 BlockTrafficZone label；
- 解析 from_subnet / to_subnet；
- 验证 blue_agent_0..4；
- 验证 block persistence；
- 验证 AllowTraffic cleanup；
- 冻结第 8 节 lifecycle 方案。

A3.5 Duration / decision availability
- 实测 Analyse=2 / Remove=3 / Restore=5 的实际 wrapper 行为；
- 确认何时 agent 再次可选择新动作；
- 冻结 decision-epoch replay 定义。

A3.6 Multi-agent scope
- 正式主实验默认支持 blue_agent_0..4；
- blue_agent_4 多 subnet 必须通过；
- 如果因论文范围必须只研究部分 region，其余 agent 的背景 policy 必须对所有方法完全相同，并在论文中说明；
- 为保证 official team reward 和方案效果，优先支持全部 5 agent。

### A3 新增文件建议

- shared/cyborg_action_adapter.py
- shared/cyborg_target_resolver.py
- tests/test_gate_a_cyborg_adapter.py
- experiments/probe_cc4_action_contract.py

### A3 验收

必须满足：

- 5 个 high-level action 均可解析；
- 所有 agent 不使用固定 action index；
- 只用 label + mask；
- 相同 observation 得到相同 target；
- 无 latent / future leakage；
- control_traffic lifecycle 已冻结；
- decision-epoch timing 已冻结；
- 形成 docs/step3-A3.md。

## A4 — Formal State / Replay / World-model Compatibility

### A4 目标

在进入 Step 4 前冻结真正用于论文实验的 state/data/model contract。

### A4 子任务

A4.1 FormalStateEncoder
- D8 vs D13 validation；
- phase / block / policy feature 提取；
- text summary 与 vector 同源。

A4.2 Formal replay collector
- train seeds；
- all 5 agents；
- decision-epoch transition；
- stratified action coverage；
- official reward accumulation。

A4.3 旧数据兼容性
- 正式声明旧 replay / WM / PPO legacy-only；
- 不允许隐式加载旧 checkpoint。

A4.4 Bootstrap dynamics ensemble
- independent bootstrap；
- state normalization；
- model validation；
- uncertainty-error calibration。

A4.5 Reward model
- official reward target；
- Huber；
- held-out MAE/RMSE。

A4.6 Checkpoint bundle
一个正式 checkpoint bundle 必须同时记录：
- state schema version；
- action contract version；
- target resolver version；
- train seed list；
- normalizer；
- ensemble members；
- reward model；
- config hash。

A4.7 Config cleanup
- compare_ug_cem_local_online.yaml 中旧 costs/delays/old D=8 formal assumption 改为 development-only；
- 正式新增 formal comparison config；
- 不再从 src/action_space.py 读取正式动作定义。

### A4 验收

- formal state frozen；
- replay frozen；
- WM quality Gate 通过；
- reward model quality通过；
- Ours / UG 可加载同一 model bundle；
- 形成 docs/step3-A4.md。

## A5 — Gate A Final Review

检查：

- action contract；
- action execution；
- control traffic lifecycle；
- duration；
- formal state；
- replay；
- bootstrap ensemble；
- reward model；
- legacy isolation；
- no leakage。

只有 A5 PASS 才进入 Step 4。

---

# 15. Step 0–3 状态

Step 0：PASS  
Step 1：PASS  
Step 2 Categorical CEM：PASS，详见 docs/step2.md  
Step 3 UG uncertainty：PASS，详见 docs/step3.md

Step 2/3 不因 D8->D13 候选变化而重写，因为：

- CEM 只依赖 A/H；
- uncertainty 对 D 维度通用。

---

# 16. Step 4 — Vectorized Shared Rollout Evaluator

## 目标

给一批 plans[N,H] 和当前 state[D]，生成：

- next_states[H,N,M,D]
- member_returns[N,M]
- expected_return[N]

并作为 Ours / UG 的共享前瞻模型接口。

## 设计

- fixed member through whole horizon；
- deterministic mu rollout；
- 不做 aleatoric resampling；
- 每个 horizon × member 一次 N-batch forward；
- reward 由 shared reward model 预测；
- tick-aware discount；
- evaluator 不知道 CEM / PPO / LLM。

## 性能

forward 数量从：
N*M*H
降为：
M*H

可选 raw trajectory cache：
- 仅当前真实 decision epoch；
- 只缓存 raw model outputs；
- uncertainty normalization 仍对 sampled population 正常计算。

## 测试

至少：

- shape；
- member fixed；
- vectorized 与 slow reference 数值一致；
- no stochastic resampling；
- reward accumulation；
- duration-aware discount；
- CPU/GPU device；
- nonfinite guard；
- batch N=1/N=64；
- D8/D13 兼容。

---

# 17. Step 5 — UGCEM Planner

## 目标

组合：

Categorical CEM
+ SharedRolloutEvaluator
+ UGUncertainty
+ MPC warm-start

## 正式 score

第 k 个 CEM iteration：

J_i =
G_i - beta * omega_i / (k+1)

必须保留 /(k+1)。

## MPC

- 每个 decision epoch 重新规划；
- best sampled plan；
- 只执行 plan[0]；
- final_probs 左移；
- 最后一行补 uniform；
- episode reset 清空 warm-start。

## Debug 输出

至少：

- best_plan
- best_score
- expected_return
- uncertainty
- final_probs entropy
- CEM iteration history
- planning latency

---

# 18. Step 6 — Uncertainty Normalizer Warm-up

## 目标

对 obs_mean / obs_std / horizon_std 做 source-faithful calibration。

## 数据

只能来自：
- train；
- calibration seeds。

禁止：
- test future state；
- test labels；
- 正式 test 结果。

## 流程

- 100 planner calls 为默认；
- warm-up 不使用 previous_solution；
- 每 agent/planner 维护匹配其模型分布的 normalizer；
- 记录 finite；
- warm-up 后默认继续在线 EMA，作为 source-faithful 主版本。

敏感性：
- freeze_after_warmup true/false。

---

# 19. Step 7 — Integration Smoke Tests

## 7.1 Local smoke

用途仅为软件联调：

- 1 episode x 20 decisions；
- 5 episodes x 100 decisions。

允许 development-only return adapter。

结果不能进入论文主表。

## 7.2 Official train-seed smoke

在正式 CC4 train seed 上：

- 2 episodes x 50 ticks；
- 检查 action adapter；
- decision epochs；
- reward model；
- planner；
- multi-agent。

在 Gate B 前不生成最终 PPO 结果。

---

# 20. Gate B — Ours 的正式 reward / prior / PPO 冻结

Gate B 在正式 PPO 重训前执行。

## B1 Reward objective

首选主目标：

- official CC4 accumulated reward；
- action duration 由环境真实体现；
- 不再手工重复加旧 action cost / lambda_delay；
- 不包含 early-warning lead steps。

如果 PPO 收敛明显不稳定，可以研究 potential-based shaping：

F(s,s') = gamma*Phi(s') - Phi(s)

但必须：

- train/validation 决定；
- 不用 test；
- Ours / shared plan-return 语义一致；
- 正式报告 shaping 公式。

## B2 LLM prior v2

冻结：

- model；
- prompt；
- temperature；
- candidate K；
- H=4；
- parser；
- fallback；
- duplicate handling。

## B3 Evidence

冻结 G/U/C/O 定义和 normalization。

## B4 PPO

旧 PPO checkpoint 全部 legacy。

正式 PPO：
- 新 action semantics；
- 新 evidence；
- new reward；
- train seeds；
- validation model selection；
- 固定 random seeds；
- 保存 optimizer/config metadata。

## B5 PPO 稳定性

至少检查：

- entropy；
- KL / clip fraction；
- value loss；
- policy loss；
- action distribution；
- candidate-rank distribution；
- validation official return。

避免：
- action collapse；
- 永久 no_op；
- 永久 restore；
- 只靠 prior score 不看 evidence。

---

# 21. Step 8 — Fair Comparison Harness

新增正式 evaluator，不复制旧 hybrid boost。

统一 method API：

method.observe(current_shared_state, current_raw_obs)
method.plan()
-> high_level_action_id

之后统一：

high_level_action
-> shared target resolver
-> shared CybORG adapter
-> official environment

## Primary planner-level comparison

Ours：
LLM prior -> shared evaluator/evidence -> PPO -> plan[0]

UG：
Categorical CEM -> shared evaluator -> UG penalty -> plan[0]

CEM：
与 UG 相同但 beta=0。

不得加入：

- _heuristic_plan_score
- _obs_heuristic_action
- action_vote
- method-specific host resolver
- risk threshold hard override
- test-time bandit boost。

## Coordinator 问题

若后续 full-system Ours 使用 region coordinator：

- 若 coordinator 与 planner 核心无关，UG/CEM 也必须共享；
- 若 coordinator 是 Ours 的明确论文贡献，则必须另外报告“full-system comparison”，不能与“planner-level mechanism comparison”混成一张表。

---

# 22. Step 9 — Validation、超参数与消融

## 22.1 Equal tuning budget

Ours / UG 的主要超参数必须使用相同 validation seed pool，并限制相似的 tuning budget。

禁止：
- 只给 Ours 大量调参；
- UG 用默认值然后直接比较。

## 22.2 UG beta

候选：

0
0.05
0.1
0.2
0.3
0.5
1.0

beta=0 = CEM-APT。

## 22.3 CEM compute

validation 比较：

- N=64/I=4
- N=128/I=5
- N=200/I=5

同时记录：
- return；
- latency；
- plan diversity。

## 22.4 Ours ablation

至少：

- Full LWM-RL
- w/o LLM prior（uniform prior）
- w/o U evidence
- w/o PPO（greedy predicted G 或固定规则，必须提前冻结）
- 可选 w/o world-model foresight

## 22.5 Model ablation

可选论文附录：

- non-bootstrap ensemble
- bootstrap ensemble

用于证明 uncertainty quality 变化。

---

# 23. Step 10 — 正式 CybORG / CC4 主实验

## 23.1 主实验范围

优先：

- blue_agent_0..4 全部受控；
- 100 episodes；
- 500 global ticks；
- FiniteStateRedAgent；
- EnterpriseGreenAgent；
- 完全相同 environment seeds。

如果计算资源不足：

先：
- 20 x 100 validation smoke；

最终至少：
- 按计算能力给出 100x500 或清楚解释缩减原因。

## 23.2 Paired evaluation

同一 episode seed：

- Ours
- UG
- CEM

使用完全相同环境初始化。

## 23.3 Training-seed robustness

理想：
- 3 个 learned-model / PPO training seeds；
- 每个 seed 小规模 paired evaluation。

主表可使用冻结后的正式 checkpoint + 100 paired environment seeds；
附录报告 training-seed sensitivity。

---

# 24. Step 11 — Metrics 与统计

## Primary

- official team episode return；
- mean；
- std；
- median；
- 95% bootstrap CI。

## Secondary

- per-agent / per-region return；
- green availability penalty；
- red impact/access penalty；
- action success rate；
- fallback-to-Sleep rate；
- Analyse/Control/Remove/Restore distribution；
- block-induced service penalty；
- Restore frequency；
- planning latency；
- decision count；
- average decision_dt。

## Model diagnostics

- one-step WM error；
- H=4 rollout error；
- reward model MAE；
- uncertainty-error correlation；
- uncertainty quantiles。

## Statistical comparison

由于相同 episode seed 配对：

- paired bootstrap CI；
- 配对显著性检验；
- 同时报告 effect size。

不再使用：

- early-warning lead steps。

---

# 25. Step 12 — 最终论文表格

至少形成：

主表：
- LWM-RL
- UG-CEM-APT
- CEM-APT

消融表：
- LWM-RL full
- no LLM
- no U
- no PPO

模型质量表：
- bootstrap ensemble quality
- reward model quality
- uncertainty calibration

效率表：
- planning latency
- runtime
- model forward count

论文必须清楚写：

> 对比方法保留其核心决策机制，但共享本文统一状态、动作、世界模型、reward objective 与 CC4 adapter，因此属于 domain-adapted implementation，而非逐行复现原作者环境。

---

# 26. 影响最终效果的优先优化顺序

如果时间有限，优化优先级固定为：

P0：
1. 正确的 CC4 action adapter；
2. action duration / decision epoch；
3. D13 mission/policy context；
4. 重新采集正式 replay；
5. bootstrap ensemble；
6. reward model 对齐 official reward；
7. 新 LLM action vocabulary；
8. PPO 重训。

P1：
9. evidence normalization；
10. candidate diversity；
11. beta validation；
12. vectorized planning。

P2：
13. raw trajectory cache；
14. delta-model ablation；
15. extra statistical analyses。

禁止为了追求表面效果优先做：

- test-time heuristics；
- Ours-only action override；
- test seed 调参；
- hidden red-state leakage。

---

# 27. 正式比较时的禁止项

1. Ours / UG 使用不同 state；
2. Ours / UG 使用不同 WM checkpoint；
3. Ours / UG 使用不同 reward model；
4. 只给 Ours 看 mission phase / policy；
5. 只给 Ours 更强 target resolver；
6. 只给 UG 收 action cost；
7. 使用 test seed 调 beta；
8. 使用 test seed 调 PPO；
9. 使用 true red host/location；
10. 把 action ID 当连续强度；
11. 忽略 action_mask；
12. 假设不同 Blue agent 的同 index action 相同；
13. 使用旧 replay / checkpoint 冒充新语义；
14. 使用 local simulator 结果替代正式 CC4 主结果；
15. 把 early-warning lead steps 重新塞回 reward；
16. 保留 run_deploy / hybrid 脚本中的隐藏 heuristic 做主实验。

---

# 28. 每阶段审核模板

每次 push 后必须审核：

1. Git diff 是否只包含本阶段；
2. 是否修改 legacy src；
3. 是否改变已冻结 contract；
4. 是否有未来信息；
5. 是否有 method-specific heuristic；
6. unit tests；
7. numerical finite；
8. seed reproducibility；
9. config consistency；
10. 文档状态。

每个 Gate A 子阶段完成后新增：

- docs/step3-A1.md
- docs/step3-A2.md
- docs/step3-A3.md
- docs/step3-A4.md
- docs/step3-A5.md

---

# 29. 当前进度

[x] Step 0  experimental branch  
[x] Step 1  baseline scaffold  
[x] Step 2  Categorical CEM  
[x] Step 3  UG uncertainty  

Gate A：
[x] A1  Shared Action Contract  
[x] A2  Local-online Action Adapter  
[ ] A3  Official CybORG / CC4 Action Adapter  <- CURRENT  
[ ] A4  Formal State / Replay / Bootstrap WM / Reward Model  
[ ] A5  Gate A Final Review  

之后：
[ ] Step 4  Vectorized Shared Rollout Evaluator  
[ ] Step 5  UGCEM Planner  
[ ] Step 6  Normalizer Warm-up  
[ ] Step 7  Integration Smoke Tests  
[ ] Gate B  Formal Reward / LLM Prior / Evidence / PPO Freeze  
[ ] Step 8  Fair Comparison Harness  
[ ] Step 9  Validation + Ablation  
[ ] Step 10 Official CC4 Main Experiment  
[ ] Step 11 Metrics + Statistics  
[ ] Step 12 Final Thesis Tables  

---

# 30. 下一步固定要求

当前下一步不是直接写 A3 代码。

进入 A3 实现前先完成：

1. 阅读本任务书；
2. 运行真实 CC4 action-space probe；
3. 确认 5 Blue agents 的 labels / masks / hosts / subnets；
4. 确认 BlockTraffic / AllowTraffic lifecycle；
5. 确认 action duration 与 decision availability；
6. 再冻结 A3 adapter API；
7. 最后开始编码。

---

# 31. 一句话记住 v2.0

共享 current observation
+ phase/policy-aware formal state
+ shared bootstrap probabilistic dynamics ensemble
+ shared official-reward model
+ shared target/action adapter
+ 严格 train/val/calibration/test 分离，

然后只把：

LLM prior + PPO posterior

与：

Categorical CEM + source-faithful uncertainty penalty

作为核心方法差异，

最终在相同 5-agent CybORG/CC4 条件下做 paired comparison。
