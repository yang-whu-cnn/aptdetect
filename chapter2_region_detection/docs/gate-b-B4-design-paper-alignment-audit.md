# Gate B4 — 任务书 / 论文方法一致性审计与 B0.2 多 LLM 修订

日期：2026-09-17  
状态：**PASS / B0.2 AMENDMENT FROZEN**

---

## 1. 审计目的

在进入真实 CC4 tiny pipeline smoke 与 provisional PPO 前，重新核对当前实现是否同时满足：

1. `docs/UG_CEM_APT_REPRODUCTION_PLAN.md` v2.3 Final Execution Contract；
2. 参考硕士论文第 4 章的 LWM-RL 核心方法职责分工；
3. 当前 CC4 dynamic-response 研究问题与公平比较要求。

本审计不重新调参，不使用 calibration/test，不根据性能结果修改算法。

---

## 2. 结论

**没有阻塞 tiny pipeline smoke 的结构性问题。**

当前实现与 v2.3 任务书一致，并保留参考论文最核心的方法链：

```text
当前 planner-visible state
-> LLM 生成 K=6/H=4 候选计划 + prior
-> ensemble probabilistic WM 多步前瞻
-> candidate posterior representation
-> PPO 在 candidate set 上做后验选择
-> 只执行 selected plan[0]
-> 新 decision state 重新生成 / 评估 / 选择
```

PPO 使用真实环境反馈更新，而不是把 LLM prior 或 WM predicted value 作为真实 reward。

---

## 3. 与 v2.3 任务书逐项核对

### 3.1 State / action / decision epoch

一致：

```text
D27
A4
feature17 = any_valid_observable_target
ready-only decision epoch
busy agent no filler transition
real decision_dt
```

Policy / LLM / planner 不使用 hidden Red truth、true compromise、future reward 或 test information。

### 3.2 LLM prior

一致：

```text
K=6
H=4
temperature=0.2
provider-neutral registry
strict structured output + semantic parser
persistent cache
host target not selected by LLM
```

Primary LLM 仍未选择。

### 3.3 World Model evidence

一致：

```text
frozen shared D27/A4 M=5 WM
same reward predictor
value = mean(member cumulative response returns)
uncertainty = population std(member cumulative response returns)
```

当前 WM 不因 LLM variant 改变而重训。

### 3.4 Posterior representation

一致：

```text
27 state
16 plan one-hot
1 prior preference
1 predicted value
1 uncertainty
= 46D / candidate
```

无 provider/model ID、latency/cost、rationale、hidden evidence。

### 3.5 PPO

一致：

```text
candidate-wise shared encoder 46->128->128
shared scalar actor head
mean-pool critic
PPO action = candidate index
one shared PPO policy / LLM variant across 5 Blue agents
per-agent GAE boundaries
```

Duration-aware rule：

```text
gamma_t = 0.99 ** decision_dt
GAE continuation = gamma_t * lambda
lambda = 0.95 per decision transition
```

### 3.6 Reward

一致：

```text
real_response_reward
= - incident_active_ticks
  + raw incident-host LWF penalty
```

禁止 predicted WM value / LLM prior 成为 PPO reward。

### 3.7 Split / fairness

一致：

```text
PPO collection = train 1000..1031 only
validation = checkpoint/model selection only
calibration = UG normalizer only
test = final frozen evaluation only
```

不同 LLM 正式训练独立 PPO，但共享 architecture、budget、seeds、WM/reward/env contract。

---

## 4. 与参考硕士论文第 4 章的关系

当前研究是 **domain-adapted LWM-RL**，不是原论文逐参数、逐奖励项复刻。

### 4.1 保留的论文核心机制

保留：

- LLM 提供候选计划与先验偏好，而不是直接给最终动作；
- Ensemble World Model 对候选计划做多步前瞻；
- 前瞻评估同时提供 future-return evidence 与 predictive uncertainty；
- PPO 基于候选计划后验表示做最终选择；
- PPO 使用真实交互反馈纠正 LLM / WM 偏差；
- 在线重规划：只执行候选计划首动作，下一决策点重新规划；
- K=6、H=4、temperature=0.2、ensemble M=5、hidden=128；
- PPO lr=3e-4、rollout=128、epochs=5、clip=0.2、GAE lambda=0.95、entropy=0.01。

### 4.2 有意的 CC4 domain adaptation

参考论文原方案还使用：

- 结构化状态 + 文本上下文的状态摘要；
- 前瞻评估中的累计动作代价；
- reward 中的提前预警、误报/风险、动作开销；
- 原文 forward discount / risk / cost / delay 参数；
- 原文指定的 LLM 与 episode 数设置。

当前 v2.3 为保证 CC4 dynamic-response 主问题、共享 baseline 契约与可解释公平比较，明确改为：

- planner-visible D27 为统一机器状态；LLM prompt 只由该可见状态构造；
- posterior 不含 action cost / delay / early-warning evidence；
- PPO reward 只使用 frozen response objective；
- `gamma_tick=0.99` 并按真实 action duration 使用 `gamma_tick**dt`；
- 比较 3 个当前 LLM，而非预先固定原论文单一模型；
- 使用 transition budget + frozen train/validation/test split。

因此最终论文必须使用类似表述：

> “基于参考 LWM-RL 的核心职责分解，在 CC4 动态响应任务上进行领域适配。”

不得写成“完全复现原论文全部奖励、状态摘要和参数”。

---

## 5. 实验覆盖审计

当前最终实验矩阵能够回答：

- LLM prior quality / repeatability；
- 多 LLM end-to-end PPO 差异；
- LWM-RL vs UG-CEM-APT vs CEM-APT；
- prior、LLM generator、uncertainty、PPO 的机制消融；
- K/H sensitivity；
- API cost / latency；
- PPO stability；
- WM OOD / uncertainty reliability；
- paired test statistics。

### 5.1 Step 9 保留一个预冻结复核点

参考论文具有 `RL-Only / LLM-RL / WM-RL / Full` 的模块级消融，而 v2.3 当前使用更细粒度的：

```text
Full
w/o prior preference
w/o LLM generator
w/o uncertainty
w/o PPO
optional w/o WM foresight
```

这不是当前实现阻塞项。

但在 Step 9 正式冻结消融前，应再次判断论文是否需要把 `w/o WM foresight` 从 optional 提升为 mandatory，以便直接支撑“World Model 前瞻模块具有独立贡献”的论文论断。若提升，必须先修订任务书再训练该 ablation，不能事后根据结果决定。

---

## 6. B0.2 多 LLM OOD 审计修订（本次冻结）

发现原 v2.3 的 B4.6/B0.2 写成单个 provisional PPO probe，但 formal training 将产生 3 个不同 LLM candidate distribution + 3 套独立 PPO policy distribution。

单个 LLM probe 不能充分证明 frozen WM 对另外两个 variant 的 policy-induced state distribution 仍可靠。

因此从本审计起，B4.6/B0.2 执行语义修订为：

```text
for each formal LLM variant:
    one provisional PPO development seed
    train environment seeds only
    <= 20,000 real decision transitions / variant
    provisional weights never enter formal result

B0.2:
    report per-model diagnostics
    + union diagnostics
    PASS requires every formal LLM variant satisfy frozen B0.2 hard conditions
```

三 variant 必须共享：

- PPO architecture/hyperparameters；
- provisional PPO seed；
- allowed train environment seed schedule；
- frozen WM/reward；
- B0.2 threshold/reference distribution；
- transition cap。

禁止根据某模型 OOD 结果为它单独调 PPO 或 WM threshold。

若某 variant coverage incomplete，可在其预先允许的 20k 上限内继续 train-only collection；不能改变超参来凑 Gate。

若任何 variant 发生真实 model shift，按原 v2.3 FAIL 分支重新开放 shared WM/reward，且所有正式方法最终仍共享同一新 frozen checkpoint。

---

## 7. Tiny pipeline smoke 冻结设计

下一阶段只验证工程链路，**不做模型优劣判断**。

默认使用：

```text
model alias = llm_l_gemini35_flash_lite
```

理由仅为：已经通过 B2 capability Gate 且 API 成本/延迟最低，适合工程 smoke。

这不是 primary-model selection；registry 中 `primary_model_selected` 必须仍为 false。

Tiny smoke：

```text
train seed = 1000
short complete CC4 episode
all 5 Blue agents
train cache only
same frozen WM/reward
fresh PPO initialization
one finite PPO update
no checkpoint kept as formal result
```

验收：

- requested A4 == selected candidate `plan[0]`；
- shared resolver/adapter 决定实际 target/fallback；
- PPO reward/dt/done 与 formal replay transition 精确一致；
- next critic value 使用下一 decision-state posterior；terminal 使用 0；
- next-state candidate context 只构造一次并复用到下一 decision，避免重复 API/WM 计算；
- cache split/model namespace 正确；
- optimizer epochs 不增加 API/cache misses；
- five-agent async boundaries 正确；
- all outputs finite；
- no validation/calibration/test；
- smoke policy/checkpoint 不进入 formal result。

---

## 8. 最终审计判定

```text
Taskbook alignment: PASS
Paper core-mechanism alignment: PASS
Domain-adaptation disclosure required: YES
Reward leakage: NONE
Hidden/test leakage: NONE
Multi-LLM fairness: PASS
PPO/WM responsibility separation: PASS
B0.2 single-model coverage gap: FIXED BY THIS AMENDMENT
Tiny pipeline smoke: CLEARED TO IMPLEMENT
```
