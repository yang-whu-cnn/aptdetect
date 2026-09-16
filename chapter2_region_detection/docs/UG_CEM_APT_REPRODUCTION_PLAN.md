
A4.5b 已完成并正式选择 absolute Bootstrap Probabilistic WM：one-step RMSE=0.132581 < persistence 0.180653；H=4 RMSE=0.191241 < persistence 0.218748；H=4 epistemic-error Spearman=0.695511，8/8 validation episodes 为正；delta H=4 RMSE=0.193774，仅赢 1/8 episodes，因此不满足冻结切换规则。selected target_mode=absolute。MAE 并非全面优于 persistence，论文只按实际结果报告，不事后改 Gate。
# UG-CEM-APT 复现、领域适配与公平对比总任务书（v2.1）

> 仓库：yang-whu-cnn/aptdetect  
> 稳定备份分支：me  
> 实验分支：ug-cem-apt  
> 当前阶段：Gate A4.6 integration review；A4.6a 已 CLOSED；A4.6b formal v2.1 config/legacy isolation source 已实现，待本地 9-test 关闭  
> 最后更新：2026-09-16  
> 本文件是后续实现、审核、实验和论文撰写的唯一总路线图。若后续方案发生实质变化，必须先更新本文件，再改实现。

---

# 0. v2.1 方案冻结：以新小论文为准

本版任务书以以下论文设计为当前 LWM-RL 正式方案来源：

- 原硕士论文《大模型+预测强化学习-电力系统APT检测》提供总体 LLM + WM + PPO 框架；
- 新小论文《融合LLM+WM+RL的APT动态响应小论文-20260915》作为当前方法细节的最新版本；
- 实验环境继续使用 CAGE Challenge 4（CC4），不切换到 DARPA；
- UG-CEM 作为当前核心对比 baseline。

v2.1 相对 v2.0 的冻结变化：

1. 高层动作空间从 5 类改为 **4 类**：
   - no_op
   - analyse
   - remove
   - restore
2. 删除 control_traffic，不再把 BlockTraffic / AllowTraffic 纳入当前 LWM-RL 与 UG-CEM 主对比动作空间。
3. LLM 仍生成多步候选计划及先验偏好。
4. World Model 仍对候选计划做多步 rollout，但正式前瞻评估只保留：
   - value / cumulative reward
   - predictive uncertainty
   不再把独立 action cost 作为第三个 PPO evidence 分量。
5. PPO 的 observation / candidate posterior representation 对齐新小论文：
   - current incident/state vector
   - candidate plan
   - LLM prior preference
   - predicted value
   - predictive uncertainty
6. 正式 response reward 改为围绕：
   - attack eradication time
   - normal operation failure
7. attack eradication time 与原硕士论文式(3.18)平均恢复时间中的“单次恢复事件时长”采用同一语义：
   T_erad = t_normal - t_compromise。
8. normal operation failure 定义为：
   **当前响应/攻击事件所在主机的 CC4 Host Work Fail**，
   不是全网所有主机 Host Work Fail 的总和。
9. early-warning lead steps 不进入当前正式 PPO reward。
10. 旧固定 action cost、lambda_delay 不进入当前正式主 reward。
11. 由于动作契约从 A=5 改为 A=4，已经通过的 A1/A2 五动作实现属于历史版本，必须先修订后才能进入 A3。

因此当前执行顺序改为：

A1R 四动作契约修订
-> A2R local adapter 修订
-> A3 official CC4 adapter
-> A4 formal state/replay/bootstrap WM/reward adapter
-> 后续 UG-CEM / LWM-RL 公平比较。

---

# 1. 论文与 CC4 环境中必须保持不变的事实

## 1.1 UG-CEM 原论文核心

UG-CEM 的核心保持：

- bootstrap ensemble dynamics models；
- CEM 多步 action-sequence optimisation；
- ensemble disagreement 估计 epistemic uncertainty；
- fixed-member / TS∞ 风格 rollout；
- deterministic mean rollout；
- score = predicted_return - beta * uncertainty / (cem_iteration + 1)。

领域适配允许改变原始任务的 state/action/reward，但 LWM-RL、UG-CEM、CEM-APT 必须共享同一套领域接口。

## 1.2 当前 LWM-RL 新小论文核心

正式方法链保持：

current incident information
-> LLM candidate plans + prior preferences
-> ensemble world-model rollout
-> value + uncertainty
-> PPO posterior plan selection
-> execute first response action
-> replan at next decision epoch。

高层动作固定为：

- no_op
- analyse
- remove
- restore

## 1.3 CC4 正式环境关键事实

正式 CC4 实现必须验证并遵守：

- Monitor 为自动监测行为；
- no_op 的显式执行使用 Sleep；
- Analyse / Remove / Restore 需要 host target；
- Analyse / Remove / Restore 具有 multi-tick duration；
- BlueFixedActionWrapper 的 action index 不能假设跨 agent 同义；
- 必须基于 action_labels + action_mask 解析真实底层动作；
- 正式环境包含 5 个 Blue agent；
- policy observation 不能使用 hidden red ground truth。

CC4 虽然存在 BlockTraffic / AllowTraffic，但 v2.1 的四动作主实验 **不把它们作为可选高层动作**。

---

# 2. 最终论文方法与对比目标

最终核心比较：

- LWM-RL（Ours）
- UG-CEM-APT
- CEM-APT（UG-CEM beta=0）

核心公平比较问题：

> 在相同 CC4 observation、四动作空间、world model、response reward、target resolver 和 paired seeds 下，
> “LLM prior + PPO posterior candidate selection”
> 是否优于
> “Categorical CEM + uncertainty-guided planning”。

正式主目标优先体现新小论文：

1. 更短 attack eradication time；
2. 更少/更低的当前 incident host Host Work Fail；
3. 更高、更稳定的 cumulative response return。

同时保留 CC4 official episode return 作为重要外部评价指标，用于检查自定义 response objective 是否没有牺牲整体系统表现。

论文结论不能依赖一次最高分，而必须使用 paired seeds、均值/方差/置信区间和消融实验。

---

# 3. 全局公平性与性能原则

所有正式方法共享：

- current visible CC4 observation；
- FormalStateEncoder；
- 四动作 high-level contract；
- host target resolver；
- official CC4 action adapter；
- train / validation / calibration / test seeds；
- bootstrap dynamics ensemble architecture；
- 同一个 world-model checkpoint bundle；
- state normalizer；
- 同一个 response reward definition / reward adapter；
- 相同 action duration / decision-epoch convention；
- evaluation metrics。

方法间只允许不同：

LWM-RL：
- LLM candidate prior；
- PPO posterior candidate-plan selection。

UG-CEM-APT：
- Categorical CEM；
- source-faithful uncertainty penalty；
- MPC warm-start。

CEM-APT：
- 与 UG-CEM 相同，但 beta=0。

正式 test 前冻结：

- state schema；
- A=4 action mapping；
- target resolver；
- world model；
- response reward；
- beta；
- PPO；
- LLM prompt/model/config；
- seeds。

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

# 5. 状态表示：开发 D=8 与正式 incident state

## 5.1 Local development state

A2 local simulator 当前 D=8 可以继续作为开发接口：

- cnt_auth_fail
- cnt_port_scan
- cnt_proc_spawn
- cnt_outbound_conn
- avg_severity
- unique_src
- unique_dst
- risk_proxy

但它不自动等于正式 CC4 state。

## 5.2 新小论文对正式 state 的要求

新小论文中的 incident information 同时包含：

- textual incident context；
- vector representation；
- alarms；
- logs；
- associated hosts。

因此正式 LWM-RL / UG-CEM comparison 的 numeric state 必须能够支持：

- world-model state transition；
- host target resolution；
- attack eradication tracking；
- current incident host Host Work Fail reward association；
- action duration / next decision availability。

## 5.3 正式 state 不预先锁死 D=13

v2.0 中为 control_traffic 设计的：

- blocked_ratio
- policy_mismatch_ratio

在 A=4 后不再是主状态必备项。

A4 必须从真实 CC4 observation 中冻结 FormalStateEncoder。

候选信息应优先包括：

- 原有行为统计/风险证据；
- mission phase（若当前 Blue observation 可合法获得且 validation 证明有用）；
- current incident / suspicious host 的可观测证据摘要；
- 当前 agent 是否可决策、必要时的 action-duration context。

禁止把 true red location、真实 compromise ground truth 直接作为 policy / planner state。

## 5.4 状态冻结规则

只用 train/validation seeds 比较候选 state schema。

比较：

- one-step prediction error；
- H=4 rollout error；
- response-reward prediction error；
- validation response return；
- official CC4 validation return；
- planning stability。

test seeds 不参与 state schema 选择。

---

# 6. 四类高层动作契约（v2.1）

新小论文正式动作空间固定：

A = 4

0 = no_op
1 = analyse
2 = remove
3 = restore

语义：

- no_op：no operation；CC4 中使用 Sleep 表达不主动响应，Monitor 仍由环境自动进行；
- analyse：intrusion investigation；
- remove：user-level compromise removal；
- restore：host reimaging。

计划底层映射：

- no_op -> Sleep
- analyse -> Analyse(host)
- remove -> Remove(host)
- restore -> Restore(host)

动作 ID 仅为 categorical index，不表示强度。

旧 A=5 contract：

0 no_op
1 analyse
2 control_traffic
3 remove
4 restore

自 v2.1 起 **superseded**，不得用于正式 replay、WM、PPO、LLM prompt 或 UG-CEM config。

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

# 8. 正式 response reward：Attack Eradication Time + Incident-Host Work Fail

## 8.1 Attack Eradication Time

对第 j 个攻击/恢复事件：

T_erad,j = t_normal,j - t_compromise,j

其定义与原硕士论文式(3.18)平均恢复时间中的单事件恢复时长一致。

最终平均：

MTR / Mean Attack Eradication Time
= mean_j(T_erad,j)

## 8.2 Normal Operation Failure

对当前响应事件 j，对应事件主机 h_j。

只统计：

HostWorkFail(h_j, t)

即 **当前事件所在主机** 的正常运行失败。

不把同一 tick 其他无关主机的 Host Work Fail 直接并入该事件 reward。

若 CC4 对不同 host work / mission work 提供不同 penalty，则保留 CC4 对应 penalty magnitude，而不是全部二值化为 1。

## 8.3 训练时的逐步实现

完整 T_erad 只有事件恢复后才完全知道。

为了便于 PPO / planning，可以用数学等价的逐 tick 延迟惩罚：

r_t^resp
=
- lambda_T * I(current incident still active)
- lambda_F * HostWorkFailPenalty(h_j,t)

在一个事件从 compromise 到 recovery 的整个区间求和时：

sum_t I(active) = T_erad

因此该实现不改变“最小化攻击消除时间”的目标，同时避免必须等恢复后才一次性回传全部时间代价。

## 8.4 重要边界

- early-warning lead steps 不进入正式 response reward；
- 不再使用旧固定 action cost；
- 不再额外手工加 lambda_delay；
- duration 的时间代价自然通过 active-duration penalty 进入；
- Host Work Fail 只取 current incident host；
- response reward 可以用于训练/规划；
- CC4 official reward 作为独立评价指标继续记录。

A4 必须通过真实 CC4 probe 确认：
- compromise/recovery event 的可记录时间点；
- current incident host 的稳定 event/host identity；
- Host Work Fail 的真实可访问字段或 reward breakdown；
- policy observation 与 reward bookkeeping 的信息隔离。

---

# 9. Shared host target resolver

planner 只选四类 high-level action，不直接输出具体 host。

正式实验必须共享同一个 host target resolver。

Host-target actions：

- Analyse
- Remove
- Restore

要求：

- Ours / UG / CEM 完全相同；
- 只能使用当前可见 observation；
- 不使用 true red location；
- 不使用未来 reward；
- tie-breaking deterministic；
- 只选择当前 action_mask 中 valid 的真实底层 action label。

no_op：

- 解析为当前 agent 可用的 Sleep。

若 requested action 当前没有合法 target：

- 保留 requested_high_level_action；
- 执行明确 fallback（默认 Sleep）；
- 记录 fallback_reason；
- replay 同时保存 requested 与 executed action。

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

# 11. Shared response-reward adapter / predicted value

## 11.1 正式目标

不再把 CC4 完整 official reward 直接定义为 LWM-RL 的论文 reward。

当前论文 response objective 固定为：

- attack eradication time；
- current incident host Host Work Fail。

UG-CEM / CEM 与 LWM-RL 共享相同 objective。

## 11.2 环境真实 reward adapter

真实交互时，shared response reward adapter 接收：

- incident_event_id
- incident_host_id
- active / recovered status
- elapsed ticks / decision_dt
- incident-host Host Work Fail penalty

输出当前 decision interval 的 accumulated response reward。

## 11.3 World-model planning value

新小论文要求 World Model 对每个 plan 估计 cumulative reward。

优先实现：

G(plan) =
discounted cumulative predicted response reward

如果 FormalStateEncoder 能使 reward 从 predicted state / action 直接计算，则使用 deterministic reward function。

如果真实 Host Work Fail 无法仅从 predicted state 稳定计算，则允许增加一个 **所有方法共享的 auxiliary response-reward predictor**，但它只是工程实现，不改变论文 reward 定义。

其训练 target 必须仍是第 8 节定义的 response reward，而不是另造 surrogate。

## 11.4 验收

需要在 held-out validation 上报告：

- response reward MAE / RMSE（若使用 predictor）；
- predicted plan value 与真实 rollout return 的相关性；
- H=4 cumulative value error。

如果 predicted value 与真实 response outcome 无明显关系，不能直接进入最终 planner 对比。

---

# 12. 数据与 seed 协议

## 12.1 Seeds

固定：

- train_seeds
- validation_seeds
- calibration_seeds
- test_seeds

train：
- replay；
- world model；
- reward predictor（如需）；
- PPO。

validation：
- state schema；
- beta；
- WM/reward design；
- LLM/PPO hyperparameters。

calibration：
- UG uncertainty normalizer。

test：
- 冻结后正式 paired evaluation。

## 12.2 Replay 必须重新采集

旧 replay 的动作语义与 A=4 不兼容，全部 legacy-only。

正式 replay 至少保存：

- episode_seed
- agent_id / region_id
- incident_event_id
- incident_host_id（仅 reward bookkeeping；不得自动作为 policy hidden input）
- decision_index
- global_tick_start / end
- decision_dt
- state
- requested_high_level_action
- executed_low_level_label
- executed_target_host
- action_success / fallback_reason
- interval_attack_active_ticks
- interval_incident_host_work_fail_penalty
- accumulated_response_reward
- accumulated_official_reward
- next_state
- done

## 12.3 Action coverage

正式 collection policy 对四动作做 stratified exploration：

- no_op
- analyse
- remove
- restore

必须报告：

- requested count；
- executed count；
- valid-target rate；
- fallback rate；
- agent/region coverage；
- incident-host coverage。

不得利用 test ground truth 指导 exploration。

---

# 13. LWM-RL（Ours）正式迁移要求

旧 src/llm_prior.py 仍使用旧五档动作名，因此不能作为正式 LWM-RL prior。

## 13.1 LLM candidate prior v2.1

固定：

- action vocabulary = {no_op, analyse, remove, restore}；
- A=4；
- H=4（除非 validation 后统一调整）；
- K candidate plans；
- prompt/model/temperature/parser/fallback 全部版本化。

validator：

- action ID 必须 0..3；
- plan length=H；
- duplicate handling；
- invalid plan rejection；
- candidate shortage 时固定 fallback plan generation。

## 13.2 Candidate diversity

记录：

- unique plan ratio；
- per-position action entropy；
- duplicate rate。

## 13.3 PPO posterior observation 对齐新小论文

对候选计划 i，posterior input 只围绕：

- current vector state s_t；
- candidate plan pi_i；
- LLM prior p_i；
- predicted value g_i；
- uncertainty u_i。

不再把旧的独立 Cost / operational burden 当作必需 evidence 维度。

## 13.4 Feature normalization

state / prior / value / uncertainty 使用 train-only normalizer 或稳定固定 scaling。

禁止使用 test episodes 拟合。

---

# 14. Gate A 详细实施（v2.1 修订后）

## A1 历史状态

原 A=5 Shared Action Contract 曾 PASS，并记录于：

docs/step3-A1.md

但由于 2026-09-15 新小论文方案冻结为 A=4，该历史 PASS 仅代表旧五动作版本。

## A1R — Four-Action Contract Revision

状态：CURRENT。

目标：

- 从 shared/action_contract.py 删除 control_traffic；
- 冻结：
  0 no_op
  1 analyse
  2 remove
  3 restore
- duration：
  no_op=1
  analyse=2
  remove=3
  restore=5
- N_ACTIONS=4；
- 更新所有 A1 tests；
- 明确旧 ID 2/3/4 重排会使旧 replay/checkpoint 彻底失效。

验收：

- 无 control_traffic；
- name/ID mapping；
- duration；
- invalid ID；
- legacy old names rejected；
- docs/step3-A1R.md。

## A2 历史状态

原 A2 五动作 local adapter 曾 PASS，记录于：

docs/step3-A2.md

其中 no_op / analyse / remove / restore 的 semantic 设计仍可复用；
control_traffic 相关 transition 和 tests 必须删除。

## A2R — Local-online Adapter Revision

目标：

- local adapter 仅四动作；
- local client action sequence 仅 0..3；
- 删除 network-specific control_traffic action effect；
- 保留 partial observability / visibility / no hidden leakage；
- reward 仍可保持 development placeholder，正式 reward 由 A4 接入。

验收：

- 四动作 semantic tests；
- reproducibility；
- StateSummary compatibility；
- no ordinal action assumption；
- docs/step3-A2R.md。

## A3 — Official CybORG / CC4 Action Adapter

目标：

冻结真实四动作底层执行。

A3.1 Wrapper probe：
- blue_agent_0..4；
- action_labels；
- action_mask；
- hosts；
- padding；
- Sleep / Analyse / Remove / Restore 的真实 label。

A3.2 no_op：
- no_op -> Sleep；
- Monitor 自动发生。

A3.3 Host actions：
- Analyse / Remove / Restore；
- shared host resolver；
- valid labels only。

A3.4 Duration / decision availability：
- 验证 Analyse/Remove/Restore multi-tick；
- 冻结 decision-epoch transition。

A3.5 Reward-source probe：
- 仅探测并记录 compromise/recovery timestamp 的可获得方式；
- current incident host identity；
- Host Work Fail breakdown；
- 不把 ground truth 泄漏给 planner state。

A3.6 Multi-agent：
- blue_agent_0..4；
- 正式比较默认共享相同背景策略/控制范围。

验收：
- 4 high-level actions 全可解析；
- 无固定跨-agent action index；
- deterministic resolver；
- duration frozen；
- reward bookkeeping source frozen；
- docs/step3-A3.md。

## A4 — Formal State / Replay / Bootstrap WM / Response Reward

A4.1 FormalStateEncoder：
- 从真实 CC4 observation 冻结；
- incident host evidence；
- mission phase 是否保留由 validation 决定；
- 不再为 traffic-control 强加 blocked/policy 特征。

A4.2 Formal replay：
- A=4；
- decision-epoch；
- new schema；
- response reward bookkeeping；
- all required agents。

A4.3 Legacy isolation：
- old replay / WM / PPO legacy-only。

A4.4 Bootstrap dynamics ensemble：
- independent bootstrap；
- normalization；
- probabilistic next-state prediction；
- absolute / delta target support。

A4.5 Validation + planning-value readiness：
- A4.5a 重新采集正式 A=4 decision-epoch replay，并冻结 train / validation split；
- A4.5b held-out one-step / H=2 / H=4 validation + uncertainty-error calibration；
- A4.5c 验证 response reward 是否可由 predicted state/action 直接计算；若不能，则训练所有方法共享的 auxiliary response-reward predictor，并验证 predicted-value quality。

A4.6 Config cleanup / integration review：
- n_actions=4；
- 删除正式 control_traffic/cost/delay 配置；
- 新 formal comparison config。

## A5 — Gate A Final Review

只有以下全部冻结后才进入 Step4：

- A=4 action contract；
- CC4 host-target adapter；
- duration / decision epoch；
- formal state；
- replay；
- bootstrap WM；
- response reward；
- legacy isolation；
- no leakage。

---

# 15. Step 0–3 状态

Step 0：PASS，详见 docs/step0.md  
Step 1：PASS，详见 docs/step1.md  
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

# 20. Gate B — Ours 的正式 prior / posterior PPO 冻结

Gate B 在正式 PPO 重训前执行。

## B1 Response reward contract

再次确认：

r_resp 只围绕：

- attack eradication duration；
- current incident host Host Work Fail。

不包含：

- early-warning lead steps；
- 旧固定 action cost；
- 手工 lambda_delay。

若最终采用逐 tick active penalty，必须证明其与事件级 T_erad 累积等价。

## B2 LLM prior v2.1

冻结：

- 4-action vocabulary；
- prompt；
- LLM model；
- temperature；
- K；
- H；
- parser；
- fallback；
- duplicate handling。

## B3 Posterior observation

冻结：

state + plan + prior + value + uncertainty

不增加未经新小论文定义且只服务于 Ours 的额外 evidence。

## B4 PPO

旧 PPO checkpoint 全部 legacy。

正式 PPO：

- A=4；
- new posterior observation；
- new response reward；
- train seeds；
- validation model selection。

## B5 PPO stability

检查：

- entropy；
- KL / clip fraction；
- value loss；
- policy loss；
- selected candidate rank；
- validation eradication time；
- validation incident-host work fail；
- official CC4 return。

避免策略 collapse 到永久 no_op 或永久 restore。

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

## Primary metrics

与新小论文 response objective 对齐：

- Mean Attack Eradication Time / 原式(3.18) MTR；
- incident-host Normal Operation Failure：
  - failure count；
  - CC4 weighted Host Work Fail penalty；
- cumulative response return。

## External system metric

- CC4 official team episode return；
- mean / std / median / 95% bootstrap CI。

## Secondary response metrics

- Restore precision（若实验需要沿用原论文恢复精确率）；
- action distribution；
- valid target rate；
- fallback-to-Sleep rate；
- per-agent/per-region response performance；
- planning latency；
- decision_dt。

## Model diagnostics

- one-step WM error；
- H=4 rollout error；
- predicted response-value error；
- uncertainty-error correlation；
- uncertainty quantiles。

## Statistical comparison

相同 episode seeds 做 paired evaluation：

- paired bootstrap CI；
- paired significance test；
- effect size。

early-warning lead steps 不作为本轮 LWM-RL vs UG-CEM 主 reward 或主指标。

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

P0：

1. 四动作契约修订正确；
2. Sleep / Analyse / Remove / Restore CC4 adapter；
3. 正确 host target resolver；
4. action duration / decision epoch；
5. current incident host 与 Host Work Fail bookkeeping；
6. formal replay 重采集；
7. bootstrap ensemble；
8. response reward / predicted value 与真实 eradication + host-work-fail 对齐；
9. 新 LLM four-action prior；
10. PPO 重训。

P1：

11. state schema validation；
12. evidence normalization；
13. candidate diversity；
14. beta tuning；
15. vectorized rollout。

P2：

16. cache / speed optimisation；
17. delta-state ablation；
18. extra statistical analysis。

禁止用以下方式“保证效果”：

- test-time heuristic override；
- Ours-only hidden information；
- test-seed tuning；
- true red-state leakage。

---

# 27. 正式比较时的禁止项

1. Ours / UG 使用不同 state；
2. Ours / UG 使用不同 WM checkpoint；
3. Ours / UG 使用不同 response reward；
4. Ours 使用 hidden current-compromise truth 作为 policy input；
5. 不同方法使用不同 host resolver；
6. action ID 当连续强度；
7. test seeds 调 beta / PPO / prompt；
8. old A=5 replay/checkpoint 用于 A=4 正式实验；
9. 把全网 Host Work Fail 错算成“当前事件主机 Host Work Fail”；
10. 把 early-warning lead steps 重新加入正式 PPO reward；
11. 使用 local simulator 代替 CC4 正式主实验；
12. 保留 method-specific hidden heuristic；
13. 固定跨-agent action index 而忽略 labels/mask；
14. 把 control_traffic 继续混入当前四动作 formal config。

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

[x] A1R Four-Action Contract Revision
[x] A2R Local-online Adapter Revision
[x] A3  Official CybORG / CC4 Four-Action Adapter（汇总见 docs/step3-A3.md）
[~] A4  Formal State / Replay / Bootstrap WM / Response Reward  <- CURRENT
[ ] A5  Gate A Final Review

历史记录：

- A1 A=5：PASS，但已被 v2.1 A=4 方案 supersede；
- A2 A=5：PASS，但必须按 A=4 修订。

之后：

[ ] Step 4  Vectorized Shared Rollout Evaluator
[ ] Step 5  UGCEM Planner
[ ] Step 6  Normalizer Warm-up
[ ] Step 7  Integration Smoke Tests
[ ] Gate B  Four-action LLM Prior + PPO Freeze
[ ] Step 8  Fair Comparison Harness
[ ] Step 9  Validation + Ablation
[ ] Step 10 Official CC4 Main Experiment
[ ] Step 11 Metrics + Statistics
[ ] Step 12 Final Thesis Tables

---

# 30. 下一步固定要求

A1R / A2R / A3.1 / A3.2 / A3.3 / A3.4 / A3.5 / A3.6 已完成。

当前继续 A3：

- [x] A3.1 CC4 wrapper contract probe
- [x] A3.2 no_op -> Sleep
- [x] A3.3 Analyse / Remove / Restore deterministic shared host resolver
- [x] A3.4 multi-tick duration / next-decision availability
- [x] A3.5 compromise/recovery + current incident host work-failure bookkeeping
- [x] A3.6 multi-agent integration — synchronous + async readiness PASS

A3.4 已冻结 decision epoch：

- no_op / Sleep: decision_dt = 1
- analyse / Analyse: decision_dt = 2
- remove / Remove: decision_dt = 3
- restore / Restore: decision_dt = 5

A3.5 已冻结 bookkeeping：

- hidden red-presence truth 仅用于 reward / evaluation bookkeeping；
- t_compromise = 当前 incident host 首次 False -> True 的 global tick；
- t_normal = 当前 incident host 首次 True -> False 的 global tick；
- attack eradication time = t_normal - t_compromise；
- 论文 Host Work Fail 对应 CC4 LWF (Local Work Fails)；
- 只累计当前 incident host 的 GreenLocalWork failure；
- 其他 host LWF、ASF、RIA、aggregate team reward 不计入 incident-specific Host Work Fail；
- official CC4 team return 单独作为外部评价指标。

A3.6 必须确认五个 Blue agent 在真实 wrapper 中都可共用同一 action adapter / resolver，覆盖不同 action-space size、不同 host 数量与 blue_agent_4 多子网场景。

A3.6 async extension 正式要求：五个 agent 必须使用不同 duration 的 mixed action queues；scheduler readiness 只根据本地 executed duration 推导，controller internals 仅作 oracle；busy agent 不提交 filler action；每次 ready launch 重新解析当前 labels/mask/adapter；seed 42/43/44 × pad false/true 全覆盖。

A3.6 async extension 已通过：mixed-duration per-agent readiness 使用 scheduler-local executed duration；busy agent 省略；controller internals 仅作 oracle。A3 正式关闭。

当前进入 A4：

- [x] A4.1 Formal State contract + observable host evidence
  - [x] A4.1a raw Blue observation contract probe
  - [x] A4.1b ObservableHostEvidenceTracker + FormalStateEncoder
  - [x] A4.1c valid-target availability correction
- [x] A4.2 Decision-epoch replay schema / collector
- [x] A4.3 Incident bookkeeping + response reward implementation
- [x] A4.4 Bootstrap probabilistic ensemble world model
- [x] A4.5 Validation + planning-value readiness
  - [x] A4.5a formal CC4 replay recollection + train/validation split
  - [x] A4.5b WM held-out validation / rollout / uncertainty calibration
  - [x] A4.5c response-reward prediction path decision / validation
- [~] A4.6 A4 integration review  <- CURRENT
  - [x] A4.6a model-space requested→executed action consistency audit — PASS / CLOSED
  - [~] A4.6b formal v2.1 config freeze + legacy isolation — source ready；local 9-test pending  <- CURRENT
  - [ ] A4.6c A4 final integration review

A4.1 必须基于真实 CC4 Blue observation 构造 planner-visible state / host evidence；不得读取 hidden red sessions、true compromise labels、future information 或 A3 probe-only synthetic scores。A4.1a 已确认 reset Processes 属于 baseline，不得直接当 threat evidence；正式 tracker 仅允许 post-reset Monitor / Analyse evidence 提升 host threat state。

A4.2 已冻结 decision-epoch replay：每个 agent 独立异步维护 open decision；requested / executed action 同时记录；fallback 使用 executed duration；terminal mid-action 保存 actual dt。incident event / host identity 与并发 incident 规则延迟至 A4.3 与 response reward 一起冻结。

A4.3 已冻结 incident/event-level response objective：每 host False->True 新建 event、True->False 关闭 event；并发 incidents 分别累计 active ticks；只有 active incident host 的 GreenLocalWork failure 计入 LWF；response reward = -lambda_time * active incident-ticks + lambda_failure * raw LWF penalty；hidden truth 仅用于 reward/evaluation。

A4.4 正式 dynamics contract 在实现前冻结：M=5、两层 MLP hidden=128、diagonal Gaussian、每成员独立初始化与独立 bootstrap sampling、train-only state normalizer。Dynamics 学习真实 executed action，而不是 fallback 前 requested action；未完成的 terminal mid-action transition 不进入 dynamics training，因此正式 WM 保持 p(s_next | s, executed_action)，不额外把 decision_dt 作为模型输入。默认先实现 absolute next-state target，同时保留 delta target 开关，A4.5 在相同 validation episodes 上比较 H=4 rollout error 后统一选择。

A4.5a 正式 replay collection contract：必须使用真实 EnterpriseGreenAgent + FiniteStateRedAgent + BlueFixedActionWrapper，不允许 probe-only fp/reliability/attack 注入；planner/collection action selection 只能使用 observable tracker state。每个 agent 使用 scheduler-local executed duration 独立异步进入 decision epoch；requested action 采用 deterministic stratified round-robin exploration，target scores 只来自 ObservableHostEvidenceTracker；hidden controller state 仅用于 IncidentResponseBookkeeper / LWF reward bookkeeping。split 按完整 episode seed 划分，严禁 random transition split。冻结 seeds：train=1000..1031（32），validation=2000..2007（8），calibration=3000..3007（8），test=4000..4019（20）；A4.5 阶段只采集 train + validation，calibration/test 保持未触碰。默认 episode steps=100（CC4 EnterpriseScenarioGenerator 官方默认）。

A4.5a 已完成正式 replay：train seeds 1000..1031 共 9041 transitions（8952 completed），validation seeds 2000..2007 共 2336 transitions（2315 completed）；四类 requested action 近似均衡；train executed Sleep/Analyse/Remove/Restore=6114/925/984/1018，validation=1641/214/233/248；train/validation fallback rate=0.428824/0.455479，valid-target rate=0.430188/0.395111；五个 Blue agent 与 incident hosts 均有充分覆盖。无需 collection-policy 优化；action imbalance 作为真实 partial-observability/fallback 现象保留，并在 A4.5b 增加 per-executed-action WM error 诊断。正式 collection 统一 pad_spaces=False；calibration/test seeds 未触碰。

---

# 31. 一句话记住 v2.1

CC4 不变，
LWM-RL 核心仍是：

LLM prior
+ bootstrap ensemble WM foresight
+ PPO posterior plan selection。

但正式领域契约现在冻结为：

4 actions
(no_op / analyse / remove / restore)
+ attack eradication time
+ current-incident-host Host Work Fail。

UG-CEM 与 CEM 使用完全相同的 state / action / WM / response reward，
只比较 planning / posterior-selection 机制差异。


A4.5b selection rule（在看到 WM 结果前冻结）：absolute 与 delta 使用完全相同 train replay、architecture、bootstrap/model seeds、epochs 与 validation episodes。Primary selector 为 held-out H=4 final-state RMSE。只有当 delta 的 aggregate H=4 RMSE 至少比 absolute 低 2%，且 8 个 validation episodes 中至少 6 个 episode 的 H=4 RMSE 更低时，才切换到 delta；否则保留 absolute。Selected model quality Gate：one-step RMSE 与 H=4 final-state RMSE 都必须优于 persistence baseline；与论文多步 plan uncertainty 对齐，Gate 使用 H=4 fixed-member mean rollout 的 ensemble disagreement 与 H=4 per-window final-state error：aggregate Spearman 必须 >0，且至少 5/8 validation episodes 为正。one-step uncertainty correlation 仍报告但不作为主 Gate。报告 mixture Gaussian NLL、H=2/H=4 RMSE、27维 per-feature、per-executed-action、per-agent diagnostics、H=4 high-error AUROC 与 uncertainty quantile calibration。不得使用 calibration/test seeds。

A4.5c reward-model contract（实现前冻结）：
- 27D FormalState 不包含 hidden incident_active_ticks、incident_event_id/host_id 或 GreenLocalWork LWF penalty，因此不能从 predicted 27D state + action 精确 deterministic 重建论文 response reward；A4.5c 采用所有方法共享的 auxiliary response-reward predictor。
- predictor 只能输入 planner-visible model-space variables：current 27D state、executed/canonical 4-action one-hot、next 27D state；hidden incident truth / incident IDs / host IDs / LWF breakdown 只作为 replay label bookkeeping，绝不能作为 predictor input。
- 监督 target 仍是 A4.3 冻结的 accumulated response_reward，不另造 surrogate；只使用 completed train transitions，terminal incomplete 与 dynamics 一致地排除。
- architecture：2-layer MLP, hidden=128, ReLU；input=27 + 4 + 27；train-only state normalization + train-only scalar reward normalization；MSE on normalized reward；Adam lr=3e-4；batch=256；epochs=50；固定 seed。
- selected A4.5b absolute WM 与 reward predictor 组合验证 H=4 planning value；reward predictor 使用 r_hat(s,a,s_next)，每个 fixed WM member 沿 horizon 独立累积 member return。
- paper 只说明 gamma 为 discount factor、未给数值。实现统一冻结 gamma_tick=0.99，所有方法共享；variable-duration decision epoch 使用 gamma_tick^(cumulative elapsed global ticks) 做 duration-aware discount。该数值属于实现超参数，不宣称来自论文。
- one-step reward Gate：held-out RMSE 必须优于 train-mean constant baseline，并报告 MAE / Pearson / Spearman。
- H=4 planning-value Gate：WM+reward predicted expected return 的 RMSE 必须优于 train-derived constant-return baseline；aggregate Spearman 必须 >0.3；至少 5/8 validation episodes 的 Spearman >0。另报告 oracle-state reward-model H=4 value error，用于区分 reward-model error 与 dynamics compounding error。
- train seeds 仅用于 predictor fitting；validation seeds 仅用于 design validation；calibration/test seeds 继续保持未触碰。


A4.5c 已完成：auxiliary response-reward predictor 输入仅为 planner-visible/model-space state + executed/canonical action + next_state，target 为冻结的两项 penalty 合成 response_reward；one-step RMSE=2.249156 < train-mean baseline 4.854820；oracle H4 value Spearman=0.729760；selected absolute-WM + reward predictor H4 RMSE=7.614861 < constant baseline 15.522030，Spearman=0.675280，8/8 validation episodes 为正。A4.5c PASS。

A4.6 integration review 必须在进入 Step 4 前解决 model-space requested→executed action consistency。A4.4/A4.5 dynamics 与 reward predictor 都学习 executed/canonical action，但未来 LLM/CEM/PPO 输出 requested high-level action；真实 CC4 targeted action 可能因无合法 observable target fallback Sleep。因此必须先在冻结 replay 上审计：FormalState feature any_observable_target 与真实 fallback 的关系、是否存在 feature=1 但某 targeted family 仍 fallback、以及能否仅从 planner-visible state + requested action 无泄漏地重建 canonical action。若不能近乎确定重建，不得直接把 requested action ID 送入 executed-action WM；需要在 A4.6 冻结新的 model-space action-availability representation 或其它所有方法共享、无 hidden truth 的一致接口。

A4.6a frozen audit rule（运行前冻结）：FormalState feature any_observable_target 是当前 model-space 唯一 action-availability signal。先用真实 replay 检验 canonicalize(requested,state)：no_op 始终 Sleep；targeted action 在 any_observable_target<0.5 时映射 Sleep，否则保持 requested。硬条件：train 与 validation 的 true-state requested→executed family 重建必须 100% 一致；若出现 feature=1 但真实 targeted fallback，或 feature=0 但真实 targeted non-fallback，则 27D state 不足以无泄漏重建 executed action，必须在进入 Step4 前扩展 action-availability representation，不能用近似 heuristic 掩盖。若 true-state mapping PASS，再用 selected absolute WM + reward predictor 做 H=4 requested-plan integrated rollout：每个 member 每一步根据其 predicted current state 独立 canonicalize requested action，再预测 next_state/reward。集成 Gate：H=4 final-state RMSE 仍须优于 persistence；H=4 predicted response return RMSE 仍须优于 train-derived constant baseline；aggregate value Spearman >0.3，至少 5/8 validation episodes 为正。另报告 targeted-action canonicalization match rate 作为诊断，不用 validation 结果事后改阈值。

A4.6a runtime failure confirms feature 17 representation insufficiency：train/validation mapping accuracy=0.992368/0.990582；feature=1 but real fallback=69/22；feature=0 but real non-fallback=0。Current `any_observable_target` only checks non-empty observable evidence, while production resolver requires intersection with current wrapper-valid host actions. Corrective A4.1c keeps D=27 but replaces feature 17 semantics/name with `any_valid_observable_target`, computed only from planner-visible labels/mask + observable_host_scores. Exact Gate is NOT relaxed. Existing formal replay/WM/reward checkpoints are superseded for final use and must be regenerated/revalidated after A4.1c.

A4.1c 已通过源码审核与 seed=1000 smoke：D 保持 27，第17维正式语义为 `any_valid_observable_target`；它只使用 current wrapper labels/mask 与 observable host scores。seed=1000 replay mapping overall/targeted accuracy 均为 1.0，feature=1 but fallback=0，feature=0 but non-fallback=0；collection trajectory statistics 与旧 seed=1000 一致。旧 formal replay 与其训练出的 WM/reward checkpoints 正式 superseded。

A4.5a corrective recollection Gate：使用同一 frozen seeds train=1000..1031、validation=2000..2007、steps=100、pad_spaces=False，写入新的输出目录，禁止覆盖/混用旧 replay。完整新 replay 在任何 WM/reward training 前必须再次通过 train/validation requested→executed true-state exact mapping accuracy=1.0、feature=1 fallback=0、feature=0 nonfallback=0。若旧 replay 尚在，本轮还应核对 transition/requested/executed/fallback/incident trajectory statistics 与旧 collection 一致；feature17 state semantic 差异是预期变化。

A4.5b corrective rerun 仍按原冻结 absolute-vs-delta selection rule重新选择，不预设 absolute 必然再次获选。A4.5c 只能在新的 A4.5b selected target mode 确定后 rerun；若仍选 absolute，现有 A4.5c implementation 可直接复用；若新结果选择 delta，则先把 A4.5c 的 world-model loading/validation 泛化为 selected target mode，不能强行使用旧 absolute 假设。

A4.5a v2 recollection 已完成：train/validation transition counts、requested/executed action counts、fallback/valid-target rates、incident-host counts 与旧正式 collection 完全一致；full exact mapping train/validation overall+targeted accuracy 均为 1.0，feature=1 fallback=0，feature=0 nonfallback=0。Old-vs-v2 integrity audit 进一步确认除 feature17 外其余 26 维 state/next_state 与全部 non-state replay fields 完全相同；train state17/next17 分别变化 103/109 条，validation 分别变化 30/30 条。A4.5a v2 PASS。

A4.5b v2 rerun 已完成：absolute one-step RMSE=0.132232 < persistence 0.180830；absolute H4 RMSE=0.191052 < persistence 0.219133；H4 epistemic-error Spearman=0.687621，8/8 validation episodes 为正。delta H4 RMSE=0.192143，relative improvement=-0.005709，且仅 2/8 episode wins，因此按原冻结 selection rule 继续选择 absolute。Selected absolute quality Gate 全部 PASS。MAE 并未全面优于 persistence，但 MAE 不是冻结 Gate，故不事后调参。A4.5b v2 PASS。

A4.5c v2 rerun 已完成：one-step RMSE=2.258064 < train-mean baseline 4.854820；oracle-state H4 RMSE=6.944553、Spearman=0.746864；selected absolute-WM + reward predictor H4 RMSE=7.626011 < constant baseline 15.522030，Spearman=0.670119，8/8 validation episodes 为正；return-uncertainty / true-error Spearman=0.630400。Frozen A4.5c Gate 全部 PASS。A4.5c v2 PASS。下一步重新执行 A4.6a，必须使用 v2 replay + v2 absolute WM + v2 reward checkpoint，原 100% requested→executed mapping hard Gate 与 H4 integrated rollout Gate 均不放宽。

A4.6a v2 runtime rerun 已通过：train/validation true-state requested→executed mapping overall/targeted accuracy 均为 1.0，feature=1 fallback=0，feature=0 nonfallback=0；integrated requested-plan H4 state RMSE=0.200015 < persistence 0.219133，value RMSE=7.608135 < constant baseline 15.522030，value Spearman=0.681405，8/8 validation episodes 为正。targeted member-step match rate=0.917540 仅为 predicted-state rollout diagnostic，不是 hard Gate。A4.6a evaluator/source 已 push；随后已修复 formal artifact binding：默认 replay/checkpoint/output 统一指向 outputs/formal_replay_v2 与 outputs/world_model_v2，并新增两项 regression tests，测试文件现共 9 个 test_*。本地 `.venv_cc4` 已完成 9/9 unit tests 与无参数 numerical rerun：所有 mapping/fallback hard Gate 保持精确通过，H4 state/value 指标逐项复现，quality_gate.pass=True，report 写入 outputs/world_model_v2/a4_6a/action_consistency_report.json。A4.6a 正式 CLOSED；详见 docs/step3-A4.6a-v2-source-audit.md。

A4.6b formal v2.1 config source 已实现：新增 `configs/compare_ug_cem_formal_v2_1.yaml`，冻结 D=27、A=4、H=4、gamma_tick=0.99、M=5、hidden=128、selected target_mode=absolute、v2 replay/WM/reward artifacts 与 exact seed protocol；`compare_ug_cem_local_online.yaml` 已明确标记 LEGACY / DEVELOPMENT ONLY。新增 `tests/test_gate_a_formal_comparison_config.py` 共 9 个 regression tests。当前待 `.venv_cc4` 本地执行通过后关闭 A4.6b；详见 `docs/step3-A4.6b.md`。