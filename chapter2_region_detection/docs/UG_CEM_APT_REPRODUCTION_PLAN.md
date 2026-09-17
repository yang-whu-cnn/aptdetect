# UG-CEM-APT / LWM-RL 复现、领域适配与公平对比总任务书（v2.2）

> 仓库：`yang-whu-cnn/aptdetect`  
> 稳定备份分支：`me`  
> 实验分支：`ug-cem-apt`  
> 最后更新：2026-09-17  
> 当前阶段：Gate A、Step 4–7 已 PASS / CLOSED；Gate B 重新组织为 **WM Final Audit → Multi-LLM Prior Study / Cache → PPO → OOD Audit → Full Training**。  
> 本文件是后续实现、审核、实验与论文撰写的唯一 source of truth。任何实质方案变化必须先修订本文件，再改实现。

---

# 0. v2.2 修订目的与 supersede 规则

v2.2 保留 v2.1 已冻结的领域契约与已通过结果，但修订 Gate B 及后续实验计划，原因是进入 PPO 前需要解决三个问题：

1. 当前 World Model 已达到原 Gate A planning-readiness，但正式 PPO 可能访问训练 replay 未覆盖的状态，因此必须增加 **per-action quality audit** 与 **policy-induced OOD / model-exploitation audit**；
2. 正式研究不再把单一 Gemini 或单一 OFOX/GPT 模型预先冻结为唯一 LLM，而是增加 **多 LLM prior-quality + end-to-end 比较**；
3. PPO 需要 LLM prior，但不应在每次 gradient update 都重复调用 API，因此必须先冻结 **model-specific persistent prior cache protocol**。

以下历史记录继续保留在各阶段文档与 Git 历史中：

- `docs/step3-A*.md`：Gate A；
- `docs/step4.md`：SharedRolloutEvaluator；
- `docs/step5.md`：UG-CEM planner；
- `docs/step6.md`：UG normalizer；
- `docs/step7.md`：integration smoke；
- `docs/gate-b-B1-B3.md`：B1/B3 与旧 B2 contract 历史；
- `docs/gate-b-B2-live.md`、`docs/gate-b-B2-ofox-amendment.md`：Gemini/OFOX 接入历史。

从 v2.2 起：

- Gemini-3.1 与 OFOX `openai/gpt-5.6-sol` 都只能视为 **LLM candidate/provider history or candidate**，不得在多模型比较完成前宣称为最终唯一主模型；
- `A=4 / D=27 / H=4 / K=6 / reward / posterior evidence` 仍保持冻结；
- test seeds 在所有设计、调参、模型选择结束前保持不可见。

---

# 1. 最终研究问题

## 1.1 主问题

在相同 CC4 observation、FormalState、四动作空间、host resolver、World Model、response reward 和 paired seeds 下，比较：

```text
LWM-RL:
LLM candidate prior
+ bootstrap ensemble WM foresight
+ PPO posterior plan selection

vs

UG-CEM-APT:
Categorical CEM
+ source-faithful uncertainty penalty

vs

CEM-APT:
UG-CEM mechanism with beta=0
```

核心问题：

> LLM prior + model-based foresight + learned posterior selection，是否能在 APT 动态响应中减少攻击消除时间、降低 incident-host normal-operation failure，并提高 cumulative response return？

## 1.2 新增多 LLM 子问题

必须另外回答：

1. 不同 LLM 的 candidate-plan quality / diversity / stability 是否不同？
2. 更强 LLM prior 是否一定带来更好的 end-to-end PPO performance？
3. WM + PPO posterior 是否能纠正较弱或不完美的 LLM prior？
4. 性能提升相对于 API latency / token / monetary cost 是否值得？

---

# 2. 已完成且继续冻结的正式领域契约

## 2.1 Environment / agents

- CAGE Challenge 4 / CybORG CC4；
- `FiniteStateRedAgent`；
- `EnterpriseGreenAgent`；
- Blue agents：`blue_agent_0..4`；
- `pad_spaces=false`；
- planner 仅在 agent ready 的 decision epoch 被调用。

## 2.2 Formal state

```text
D = 27
FormalStateEncoder
feature 17 = any_valid_observable_target
```

只允许 planner-visible observation / tracker evidence。

禁止：

- hidden Red sessions；
- true compromise label；
- future reward；
- attack ground-truth label；
- test information。

## 2.3 High-level action contract

```text
A = 4
0 no_op   -> Sleep            duration 1
1 analyse -> Analyse(host)    duration 2
2 remove  -> Remove(host)     duration 3
3 restore -> Restore(host)    duration 5
```

动作 ID 是 categorical ID，不代表强度。

Analyse / Remove / Restore 共用同一个 planner-visible deterministic host target resolver。

若 targeted action 无合法 observable target：

- requested action 保留；
- runtime fallback 到 Sleep；
- model-space 通过 shared canonicalizer 映射为 executed/canonical action；
- fallback 必须记录。

## 2.4 Response objective

正式 response reward：

```text
r_resp
= -1.0 * incident_active_ticks
  + 1.0 * raw incident-host LWF penalty
```

事件级：

```text
T_erad = t_normal - t_compromise
sum(active tick penalty) = T_erad
```

不包含：

- early-warning lead reward；
- fixed action cost；
- `lambda_delay`；
- checkpoint coverage。

CC4 official team return 始终作为独立 external metric。

---

# 3. Seed / data protocol

冻结：

```text
train       = 1000..1031  (32)
validation  = 2000..2007  (8)
calibration = 3000..3007  (8)
test        = 4000..4019  (20)
```

用途：

- train：replay / WM / reward predictor / PPO；
- validation：WM audit、LLM comparison、PPO checkpoint selection、beta、K/H sensitivity、消融设计；
- calibration：UG uncertainty normalizer only；
- test：所有配置完全冻结后的 paired evaluation only。

禁止：

- transition-level random split 代替 episode-seed split；
- calibration/test 更新 WM、reward predictor、PPO normalizer、prompt、模型选择；
- test-seed tuning。

---

# 4. 已完成阶段状态

```text
[x] Gate A — A=4 / D=27 / replay / WM / reward / integration
[x] Step 4 — Vectorized SharedRolloutEvaluator
[x] Step 5 — UGCEM Planner
[x] Step 6 — UG uncertainty normalizer calibration
[x] Step 7 — integration smoke
```

Gate A 最终本地 regression：191/191 PASS。

Step 7 最终：

```text
local short = 20 planner calls
local long  = 500 planner calls
official seeds = [1000,1001]
scenario ticks = [50,50]
post-reset env steps = [49,49]
official decisions = [205,182]
pass = True
```

---

# 5. 当前 frozen World Model / reward predictor

## 5.1 Selected WM

正式 checkpoint：

```text
outputs/world_model_v2/a4_5b/world_model_absolute.pt
```

结构：

```text
state_dim = 27
n_actions = 4
ensemble M = 5
hidden = 128 x 2
probabilistic diagonal Gaussian
independent init / bootstrap / optimizer
absolute next-state target
train-only state normalization
fixed-member deterministic mean rollout
```

训练 config：

```text
lr = 3e-4
batch = 256
epochs = 50
```

正式 replay：

```text
train transitions      = 9041
validation transitions = 2336
```

validation executed counts：

```text
Sleep    = 1641
Analyse  = 214
Remove   = 233
Restore  = 248
```

## 5.2 Existing WM quality

```text
one-step RMSE = 0.132231702
persistence   = 0.180830250

H4 RMSE       = 0.191052066
persistence   = 0.219133339

H4 uncertainty-error Spearman = 0.687620634
positive validation episodes  = 8/8
```

因此当前结论是：

> WM 已达到短 horizon planning-ready 条件，但不能解释为“长期预测高度准确”。

## 5.3 Shared reward predictor

正式 checkpoint：

```text
outputs/world_model_v2/a4_5c/response_reward_predictor.pt
```

已有结果：

```text
one-step reward RMSE = 2.258064
train-mean baseline  = 4.854820

WM + reward H4 value RMSE = 7.626011
constant baseline          = 15.522030
H4 value Spearman          = 0.670119
positive episodes          = 8/8
```

A4.6 requested-plan integration rerun：

```text
H4 state RMSE = 0.200014838 < 0.219133339
H4 value RMSE = 7.608134616 < 15.522029957
H4 value Spearman = 0.681404826
8/8 episodes positive
```

---

# 6. Gate B0 — World Model Final Audit（PPO 前新增）

目标：不是重新训练 WM，而是确认当前 frozen WM 足以进入 PPO；只有 audit FAIL 才重新打开 WM training。

## B0.1 Per-action / coverage audit — CURRENT FIRST IMPLEMENTATION

### 任务

在 frozen v2 train/validation replay 上重新加载 selected WM，不训练参数，只做分动作与序列诊断。

### 设计

对 `Sleep / Analyse / Remove / Restore` 分别报告：

- train / validation completed sample count；
- one-step WM RMSE / MAE；
- action-specific persistence RMSE / MAE；
- ratio `WM_RMSE / persistence_RMSE`；
- sample-level ensemble uncertainty vs prediction-error Spearman；
- uncertainty quartile calibration；
- first-action-conditioned H=4 window count、H4 RMSE、H4 persistence RMSE；
- per-agent × action count / RMSE（diagnostic）。

不得：

- 重新 fit normalizer；
- 重新训练 WM；
- 使用 calibration/test；
- 根据结果修改 threshold。

### B0.1 验收

硬条件：

1. exact frozen train/validation seeds；
2. selected checkpoint target mode=`absolute`；
3. 所有数值 finite；
4. 四个 action validation completed count 均 >= 150；
5. 四个 action one-step `WM_RMSE / action_persistence_RMSE <= 1.25`；
6. 至少 3/4 action 的 one-step WM RMSE <= action-specific persistence；
7. aggregate one-step/H4/uncertainty 指标必须在数值容差内复现 A4.5b frozen report；
8. 不允许某 action 出现无法解释的空 bucket / action-ID mismatch。

若第 5/6 条失败：WM Gate 重新打开，必须先做 train-only targeted data augmentation，再全方法统一重训 WM。

H4 first-action bucket 与 per-action Spearman 先作为 diagnostic，不因单一小 bucket 轻微波动自动重训；若出现明显 catastrophic bucket，则在文档审核后再决定是否 reopen。

### 输出

```text
outputs/world_model_v2/b0_1/per_action_audit.json
```

记录文档：

```text
docs/gate-b-B0-WM-audit.md
```

## B0.2 Policy-induced OOD / model-exploitation audit — 在 provisional PPO 后执行

### 任务

正式 PPO 全量训练前，先用一个 provisional PPO seed 在 **train seeds only** 上做短 rollout，收集 PPO 实际访问的真实 decision transitions，检查其是否显著偏离 WM replay distribution。

### 设计

provisional probe：

```text
PPO seed: one development seed only
train environment seeds only
max decisions: 20,000
results: integration/audit only, not paper table
```

对真实 visited transitions 计算：

- one-step WM RMSE；
- model-space action coverage；
- normalized state z-RMS distance；
- normalized train-state kNN distance；
- ensemble uncertainty；
- uncertainty-error Spearman；
- top-quartile uncertainty vs bottom-quartile error；
- OOD fraction：visited state 的 normalized kNN distance 超过 frozen validation 99th percentile 的比例。

能构造连续窗口时额外报告 H=4 real rollout error。

### B0.2 验收

硬条件：

1. probe 只使用 train seeds；
2. no hidden truth into policy；
3. probe one-step RMSE <= `1.25 * frozen validation one-step RMSE`；
4. aggregate uncertainty-error Spearman > 0；
5. uncertainty top quartile 的 mean true error > bottom quartile；
6. OOD fraction <= 10%；
7. 无 action bucket 完全消失；
8. 不允许 PPO reward 由 WM predicted return 代替真实 CC4 response reward。

若 FAIL：

```text
train-only targeted recollection
-> append new replay version
-> retrain WM + reward predictor if target distribution changed
-> rerun A4.5/A4.6 + B0.1
-> invalidate provisional PPO
-> restart PPO from scratch
```

若 PASS：当前 WM 正式继续冻结到所有 LLM / UG / CEM 比较结束。

---

# 7. Gate B1 — Response Reward Re-audit

状态：**PASS / CLOSED**。

已通过 executable test 证明：

```text
False -> True @1
True  -> False @4
active ticks = 3
T_erad = 4 - 1 = 3
```

后续 PPO 使用真实 environment decision interval 的 accumulated `response_reward`。

禁止：

- 用 WM predicted value 直接当 PPO training reward；
- 用 LLM prior score 当 reward；
- 恢复旧 action cost / delay reward。

---

# 8. Gate B2 — Multi-LLM Prior Study（v2.2 重构）

B2 不再预先冻结一个唯一 provider/model。

## B2.1 Structural prior contract — FROZEN

主 contract：

```text
A = 4
K = 6
H = 4
prompt versioned
planner-visible D27 only
strict structured JSON if provider supports it
local semantic parser always required
exact duplicate -> keep highest score
invalid plan -> reject
candidate shortage -> deterministic neutral fallback
prior preference -> nonnegative linear normalize to sum 1
```

动作名：

```text
no_op / analyse / remove / restore
```

host target 不由 LLM 选择。

## B2.2 Candidate model tiers

正式比较至少 3 个 LLM，按能力/成本层级选择，而不是只比较品牌：

```text
Tier-H: strong / high-capability model
Tier-M: medium-cost general model
Tier-L: small / low-cost or open-weight-access model
```

当前 `openai/gpt-5.6-sol` via OFOX 只作为 Tier-H candidate；历史 Gemini-3.1 接入结果作为 engineering history，不自动进入最终模型集。

exact provider/model IDs 必须在 prior study 开始前写入 model registry；一旦开始 formal validation state bank，不得中途替换同一 model alias。

## B2.3 Generation-parameter fairness

默认请求：

```text
temperature = 0.2
K = 6
H = 4
```

若某 provider/model 明确不支持某 generation parameter：

- 不允许伪造支持；
- registry 记录 unsupported / omitted；
- paper 明确报告实际 generation config；
- 不能为了某个模型单独调 prompt 或 K/H 获得优势。

## B2.4 Validation state bank

只使用 validation seeds `2000..2007`。

构建固定 planner-visible state bank，默认目标 240 states：

- five agents 全覆盖；
- 8 validation seeds 全覆盖；
- 按 agent×seed deterministic selection；
- decision order 均匀取样；
- 若同一 agent/seed 同时存在 feature17=0/1，则两种 target-availability 都必须覆盖；
- 不看 hidden attack labels / future reward 选样本。

state bank 一旦冻结，所有 LLM 完全相同。

## B2.5 Prior quality metrics

每个 LLM 报告：

### Format / reliability

- API success rate；
- schema-valid rate before local fallback；
- semantic-valid plan rate；
- fallback candidate rate；
- duplicate rate。

### Diversity

- unique plan ratio；
- per-position action entropy；
- candidate-set action coverage。

### WM-based candidate quality

使用同一个 frozen WM/reward evaluator：

- top-prior plan predicted value；
- best-of-K predicted value；
- candidate mean predicted value；
- candidate-set value spread；
- candidate uncertainty mean / max；
- prior-score vs predicted-value Spearman；
- top-prior plan rank among K by predicted value。

这些只用于 candidate-quality analysis，不能单独证明真实 environment superiority。

### Efficiency

- API latency p50/p95；
- token usage（若 provider 提供）；
- monetary cost（若可获得）；
- cache hit/miss；
- failed/retried calls。

## B2.6 Repeatability / stochasticity

从 frozen validation bank 预先选 30 states，所有 LLM 相同；每个模型重复 3 次。

报告：

- candidate-set overlap / Jaccard；
- top-1 plan agreement；
- prior score variance；
- best-of-K predicted-value variance。

repeatability subset 不用于 prompt tuning。

## B2.7 Uniform baseline

Prior-quality study 必须加入非 LLM baseline：

```text
state-independent deterministic uniform sample from 4^4 plan space
uniform prior = 1/K
```

用于判断 LLM 是否真的改善 candidate set，而不只是由 WM/PPO 完成全部工作。

---

# 9. Gate B2.5 — Persistent LLM Prior Cache Protocol

这是 PPO 前必须完成的工程/实验协议，不再视为 P2 speed optimisation。

## 9.1 Cache key

默认禁止 state quantization，避免不同状态错误碰撞。

canonical key 至少包含：

```text
split
provider
exact model ID
prompt version
A/K/H
generation config
D27 float32 state bytes/hash
```

最终使用 SHA256 key。

## 9.2 Cache namespace

严格隔离：

```text
train / validation / test
model-A / model-B / model-C
prompt-version
```

不同模型不得共享 candidate output。

## 9.3 Cache value

保存：

- state hash；
- model/provider；
- prompt version；
- generation config；
- candidate plans；
- raw prior scores；
- normalized priors；
- source=`llm` or fallback；
- fallback count；
- prompt hash / response hash；
- latency；
- usage/cost metadata if available；
- creation timestamp / format version。

禁止保存：

- API key；
- hidden CC4 truth；
- future reward；
- test labels。

raw response 默认不进入正式 cache；如 debug 临时保存必须在非正式目录且不得提交 Git。

## 9.4 Runtime semantics

```text
state -> key
  cache hit  -> reuse frozen PriorBatch
  cache miss -> call provider -> validate -> write atomically -> use
```

API 调用发生在 cache miss，不发生在 PPO 每次 gradient epoch。

## 9.5 验收

- same key deterministic retrieval；
- different model/split/prompt cannot collide；
- atomic write；
- corrupt cache hard fail or quarantined，不 silent reuse；
- no API key serialization；
- cache hit does not call provider；
- cache miss exactly one provider call in unit mock；
- train/validation/test namespaces enforced。

---

# 10. Gate B3 — PPO Posterior Observation

状态：结构 contract 已通过，继续冻结：

每 candidate：

```text
current state             27
candidate plan one-hot    16
LLM prior                  1
predicted value             1
predictive uncertainty      1
--------------------------------
candidate feature          46
```

其中：

```text
value_i = mean(member cumulative returns)
uncertainty_i = population std(member cumulative returns)
```

不得加入：

- action cost；
- checkpoint coverage；
- delay；
- early-warning evidence；
- hidden incident identity。

---

# 11. Gate B4 — Formal PPO

旧 PPO checkpoint 全部 legacy。

## B4.1 Network architecture

为了支持候选 permutation consistency 与未来 K sensitivity，正式网络采用 candidate-wise shared encoder，而不是简单 flatten K×46：

```text
candidate feature [K,46]
      |
shared encoder per candidate
46 -> 128 ReLU -> 128 ReLU
      |
      +--> actor scalar head per candidate -> K logits -> Categorical
      |
      +--> mean pool K embeddings -> critic MLP 128 -> 128 ReLU -> 1
```

特点：

- actor 对 candidate permutation equivariant；
- critic 对 candidate order invariant；
- K 在实现上可变，但正式主实验 K=6；
- PPO action 是 candidate index，不是 high-level response action。

初始化与优化固定：

- orthogonal linear init；
- actor output gain=0.01；
- critic output gain=1.0；
- Adam。

## B4.2 PPO hyperparameters

主版本：

```text
learning_rate = 3e-4
rollout_length = 128 decision transitions
update_epochs = 5
minibatch_size = 64
clip_epsilon = 0.2
GAE lambda = 0.95
entropy_coef = 0.01
value_coef = 0.5
max_grad_norm = 0.5
advantage_normalization = true
gamma_tick = 0.99
```

这些属于实现超参数，不宣称全部来自原论文。

variable-duration GAE：

```text
gamma_t = gamma_tick ^ decision_dt
TD delta = r_real + gamma_t * V(next) - V(current)
GAE_t = delta_t + gamma_t * lambda * GAE_next
```

`lambda` 按 decision transition 使用，不做 `lambda^dt`。

## B4.3 PPO reward

PPO update 只能使用真实 CC4 decision interval 的 accumulated response reward。

WM predicted value：

- 只作为 candidate posterior evidence；
- 不作为 PPO target reward；
- 不覆盖 real reward。

## B4.4 Multi-agent training

一个 LLM variant 对应一个 shared PPO policy，blue_agent_0..4 共用网络；不为每个 agent 单独训练策略，避免参数量与 tuning budget 不公平。

trajectory 按 agent 独立维护 done / GAE boundary，但 update 时合并同一 policy 的 rollout batch。

## B4.5 Pilot / provisional PPO — 先用于 B0.2

在正式全量 training 前：

- 只选一个 development PPO seed；
- train seeds only；
- <=20,000 decision transitions；
- 使用 frozen cache/protocol；
- 只验证 pipeline 与触发 B0.2 OOD audit；
- 结果不进入论文表格。

B0.2 PASS 后，正式 PPO 从初始化重新开始，不继承 provisional weights。

## B4.6 Formal training budget

每个 LLM variant、每个 PPO training seed：

```text
max environment decision transitions = 100,000
```

所有 LLM 相同，不因模型质量不同增加训练预算。

checkpoint 每 10,000 decisions 保存并在 validation seeds 上评估。

训练完整预算后，按 validation **cumulative response return** 选择 checkpoint；tie-break：

1. lower mean attack eradication time；
2. higher official CC4 return。

不得提前看 test。

## B4.7 PPO training seeds

formal 最少 3 个 PPO seeds / LLM variant。

先执行 1-seed engineering pilot；pipeline PASS 后再执行 3-seed formal runs。

所有 LLM 使用完全相同 PPO seed list。

---

# 12. Gate B5 — PPO Stability Gate

每个正式 PPO run 必须记录：

- policy entropy；
- approximate KL；
- clip fraction；
- policy loss；
- value loss；
- explained variance；
- gradient norm；
- selected candidate prior rank；
- selected candidate predicted-value rank；
- action distribution；
- no_op / restore frequency；
- cache hit rate；
- validation response return；
- validation eradication time；
- validation incident-host LWF；
- official CC4 return。

硬失败：

- NaN/Inf；
- policy action outside current candidate count；
- hidden/test leakage；
- long-run policy collapse 到单一 high-level action且 validation objective 同时恶化；
- checkpoint selection 使用 test。

单纯低 entropy 不自动判 FAIL，必须结合 action distribution 与 validation performance。

---

# 13. Step 8 — Fair Comparison Harness

统一 method API：

```text
method.observe(shared_state, raw_visible_obs)
method.plan()
-> requested high_level_action_id
```

统一后处理：

```text
requested action
-> shared target resolver
-> shared CC4 adapter
-> real environment
```

方法：

```text
LWM-RL: LLM prior -> shared WM evaluator -> PPO -> plan[0]
UG-CEM: categorical CEM -> shared WM evaluator -> UG penalty -> plan[0]
CEM-APT: same as UG but beta=0
```

禁止：

- method-specific resolver；
- observation heuristic override；
- risk threshold hard override；
- action vote；
- test-time bandit boost；
- hidden red truth。

---

# 14. Step 9 — Validation / Multi-LLM / Sensitivity / Ablation

## 14.1 Multi-LLM prior-quality experiment

比较：

```text
LLM-H
LLM-M
LLM-L
Uniform non-LLM baseline
```

使用同一 frozen validation state bank 与同一 WM。

输出独立表：quality / diversity / repeatability / latency / cost。

prior-quality 结果本身不决定论文主表 winner。

## 14.2 End-to-end multi-LLM comparison

每个 LLM 独立：

```text
LLM-H + PPO-H
LLM-M + PPO-M
LLM-L + PPO-L
```

禁止 `PPO-H` 直接拿去配 `LLM-M` 作为正式公平比较。

所有模型共享：

- PPO architecture；
- training budget；
- PPO seeds；
- WM/reward；
- train/validation env seeds。

validation 后冻结一个 **primary LWM-RL model configuration**，用于主方法对比；其余 LLM variant 仍作为多模型效果实验完整报告。

## 14.3 UG beta

候选：

```text
0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0
```

beta=0 即 CEM-APT。

## 14.4 CEM compute sensitivity

validation：

```text
N=64 / I=4
N=128 / I=5
N=200 / I=5
```

记录 return / latency / plan diversity。

## 14.5 K sensitivity

主实验继续 K=6。

只在 validation 上做：

```text
K = 3, 6, 10
```

优先评估：

- candidate quality；
- best-of-K predicted value；
- diversity；
- latency / token / cost；
- greedy-G small environment validation。

不因 sensitivity 结果在看到 test 后修改主 K。

由于 PPO network 为 candidate-wise architecture，未来可支持 variable K；若要做 end-to-end K ablation，必须单独重训对应 PPO，不允许直接把 K=6 checkpoint 当正式 K=10 结果。

## 14.6 H sensitivity

主实验继续 H=4。

validation-only：

```text
H = 2, 4, 6
```

报告：

- WM rollout error；
- candidate value spread；
- uncertainty；
- LLM latency/cost；
- greedy-G small validation performance。

H sensitivity 是 foresight vs compounding-error 分析，不在 test 后改 H。

## 14.7 Ours mechanism ablations

至少：

### A. Full LWM-RL

LLM plans + LLM prior + WM value + uncertainty + PPO。

### B. w/o prior preference

保留同一 LLM candidate plans，但：

```text
prior_i = 1/K
```

PPO 必须重训。

作用：只检验 LLM preference score 的贡献。

### C. w/o LLM generator

不调用 LLM；使用 state-independent deterministic uniform plan-space generator + uniform prior。

PPO 必须重训。

作用：检验整个 LLM candidate-generation contribution。

### D. w/o uncertainty

candidate uncertainty feature 固定为 0；其它不变；PPO 重训。

### E. w/o PPO

不训练 posterior，直接选择：

```text
argmax_i predicted_return_i
```

不得加入 heuristic tie-break，除固定 lowest-index deterministic tie-break 外。

### F. optional w/o WM foresight

只有论文篇幅/资源允许才做；必须提前冻结替代 evidence 定义，不得临时启发式实现。

## 14.8 Optional WM ablation

附录可选：

- non-bootstrap ensemble；
- bootstrap ensemble。

主要用于证明 uncertainty quality，而不是重新寻找一个只对 Ours 更好的 WM。

---

# 15. Step 10 — Formal CC4 Test

所有 design/tuning/selection 完成后一次性解锁 test seeds `4000..4019`。

正式 episode：

- five Blue agents；
- 500 scenario ticks；
- same environment seed for paired methods；
- no test-time learning/tuning。

主方法表至少：

```text
Primary LWM-RL
UG-CEM-APT
CEM-APT
```

多 LLM 表：

```text
LLM-H + PPO-H
LLM-M + PPO-M
LLM-L + PPO-L
```

如果资源限制导致三模型不能全部做 3 training seeds ×20 test seeds，必须优先保证 primary main table 与至少 validation-complete multi-LLM study，并在论文明确说明资源限制；不得只报告最好的一次运行。

---

# 16. Step 11 — Metrics / Statistics

## 16.1 Primary response metrics

- Mean Attack Eradication Time；
- incident-host LWF count；
- incident-host weighted LWF penalty；
- cumulative response return。

## 16.2 External metric

- CC4 official team return。

## 16.3 Secondary

- Restore precision；
- action distribution；
- valid-target rate；
- fallback-to-Sleep rate；
- per-agent result；
- planning latency；
- decision_dt。

## 16.4 LLM-specific efficiency

- API call count；
- cache hit rate；
- latency p50/p95；
- token input/output；
- monetary cost when provider exposes enough information；
- fallback / retry rate。

## 16.5 Model diagnostics

- one-step / H4 WM error；
- per-action WM error；
- policy-induced OOD fraction；
- predicted value error；
- uncertainty-error Spearman / quantiles。

## 16.6 Statistical comparison

主方法使用 paired environment seeds。

至少报告：

- mean / std / median；
- 95% paired bootstrap CI；
- two-sided paired permutation test；
- paired effect size。

多组 pairwise secondary comparisons 使用 Holm correction。

统计显著性不能替代 effect size 与 raw distribution。

---

# 17. Step 12 — Final Paper Tables

至少形成：

## Table A — Main comparison

```text
LWM-RL | UG-CEM-APT | CEM-APT
```

## Table B — Multi-LLM prior quality

```text
format reliability
diversity
WM-based candidate quality
repeatability
latency/cost
```

## Table C — End-to-end LLM variants

```text
LLM-H + PPO-H
LLM-M + PPO-M
LLM-L + PPO-L
```

## Table D — Ablation

```text
Full
w/o prior preference
w/o LLM generator
w/o uncertainty
w/o PPO
(optional) w/o WM
```

## Table E — WM / reward quality

```text
aggregate
per-action
OOD audit
uncertainty calibration
reward/value quality
```

## Table F — Efficiency

```text
planning latency
WM forward count
API calls
cache hit
LLM token/cost
```

---

# 18. Implementation order（v2.2 固定）

严格按以下顺序推进：

```text
B0.1  WM per-action final audit
  ↓
B2.1–B2.7  multi-LLM prior protocol + validation state bank
  ↓
B2.5 prior cache implementation
  ↓
B3 regression / provider-neutral cleanup
  ↓
B4.1 PPO network + unit tests
  ↓
B4.5 provisional PPO (train-only)
  ↓
B0.2 policy-induced OOD audit
  ├─ FAIL -> train-only recollect + shared WM/reward retrain + restart PPO
  └─ PASS
       ↓
B4.6 formal PPO training (3 seeds/model)
  ↓
B5 stability
  ↓
Step 8 fair harness
  ↓
Step 9 validation / multi-LLM / sensitivity / ablation
  ↓
freeze everything
  ↓
Step 10 test
  ↓
Step 11 statistics
  ↓
Step 12 tables/paper
```

---

# 19. 实现优先级

P0：

1. B0.1 WM audit；
2. provider-neutral LLM registry / prior cache；
3. fixed validation state bank；
4. formal PPO network/real reward semantics；
5. provisional PPO + B0.2 OOD audit；
6. formal 3-seed PPO training；
7. fair harness。

P1：

8. multi-LLM prior-quality experiment；
9. multi-LLM end-to-end；
10. UG beta / CEM compute；
11. ablations；
12. K/H sensitivity。

P2：

13. optional WM ablation；
14. extra appendix analyses。

LLM cache 不再属于 P2；它是 formal PPO 的必要基础设施。

---

# 20. 禁止项

1. Ours / UG / CEM 使用不同 state / WM / reward；
2. test seeds 调 prompt / model / PPO / beta / K / H；
3. hidden Red truth 进入 policy/planner/LLM prompt；
4. 用 predicted WM return 代替真实 PPO reward；
5. 给某个 LLM 单独使用不同 K/H/prompt 提升成绩；
6. 同一个 PPO checkpoint 跨不同 LLM 直接作为公平 end-to-end 结论；
7. cache 跨 model/split/prompt namespace 复用；
8. cache state quantization 未做 sensitivity 就用于主实验；
9. API failure 被 fallback 掩盖为“live call success”；
10. method-specific host resolver / heuristic override；
11. old A=5 replay/checkpoint；
12. early-warning/action-cost/delay 重新混入正式 objective；
13. 根据 test 结果决定是否重训 WM；
14. 只报告表现最好的一次 PPO seed。

---

# 21. 每阶段审核模板

每阶段必须回答：

1. 本阶段任务是什么？
2. 输入 split 是否正确？
3. 是否触碰 calibration/test？
4. 是否改变冻结 contract？
5. 是否加入 hidden information / heuristic？
6. 是否所有随机 seed 明确？
7. unit/regression tests 是否通过？
8. numerical outputs 是否 finite？
9. output provenance / checkpoint / config 是否记录？
10. PASS/FAIL 条件是否在运行前冻结？
11. 失败后是修 bug、重新开 Gate，还是允许继续？
12. docs/source-of-truth 是否同步？

---

# 22. 当前进度

```text
[x] Gate A
[x] Step 4
[x] Step 5
[x] Step 6
[x] Step 7

[~] Gate B0.1 — WM per-action final audit          <- CURRENT
[ ] Gate B2 — Multi-LLM prior study
[ ] Gate B2.5 — Prior cache
[ ] Gate B4.1 — PPO network
[ ] Gate B4.5 — provisional PPO
[ ] Gate B0.2 — policy-induced OOD audit
[ ] Gate B4.6 — formal PPO
[ ] Gate B5 — PPO stability
[ ] Step 8 — fair harness
[ ] Step 9 — validation / multi-LLM / ablation
[ ] Step 10 — formal test
[ ] Step 11 — statistics
[ ] Step 12 — paper tables
```

---

# 23. 一句话记住 v2.2

```text
先证明 frozen WM 对四动作与 PPO 新分布仍可靠，
再用同一 state bank 公平比较多个 LLM prior，
用 model-specific cache 控制 API 成本与可重复性，
每个 LLM 独立训练同预算 PPO，
最后在完全共享的 CC4/WM/reward/adapter 下与 UG-CEM/CEM 做 paired test。
```
