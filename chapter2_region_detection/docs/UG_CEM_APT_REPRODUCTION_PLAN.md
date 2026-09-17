# UG-CEM-APT / LWM-RL 复现、领域适配与公平对比总任务书（v2.3 Final Execution Contract）

> 仓库：`yang-whu-cnn/aptdetect`  
> 稳定备份分支：`me`  
> 实验分支：`ug-cem-apt`  
> 最后更新：2026-09-17  
> 当前阶段：Gate A、Step 4–7 已 PASS / CLOSED；当前进入 **Gate B0.1 — Frozen World Model Final Audit**。  
> 本文件是后续实现、审核、实验、统计与论文撰写的唯一 source of truth。任何实质性方案变化必须先修订本文件，再修改实现。

---

# 0. v2.3 修订原则与 supersede 规则

v2.3 在 v2.2 基础上完成最终实验路线冻结。核心修订不是更换已通过的领域契约，而是把 PPO 前后所有阶段补全为可执行 Gate：每一阶段都明确 **任务、输入、设计、产物、禁止项、验收条件、失败分支**。

v2.3 的核心决定：

1. 当前 frozen World Model 不因“训练速度快”而直接重训；先完成 per-action 与后续 policy-induced OOD 审计，只有 audit FAIL 才重新开放 WM training。
2. 不再预先把 Gemini 或 OFOX/GPT-5.6 Sol 固定为唯一正式 LLM。最终至少比较 3 个能力/成本层级模型，并加入 uniform non-LLM baseline。
3. LLM prior 与 PPO 解耦：API 只在 cache miss 时调用；PPO gradient update 不重复调用 LLM。
4. 每个 LLM variant 必须独立训练 PPO；不同 LLM 不共享同一个 PPO checkpoint 作为正式公平结论。
5. 所有 LLM、LWM-RL、UG-CEM、CEM-APT 共享同一个 D27/A4、WM、reward predictor、resolver、adapter、seed protocol。
6. 正式 test seeds 在全部模型、prompt、K/H、PPO、beta、ablation 定义冻结前不可使用。

历史 Gemini/OFOX live 接入仅保留工程审计价值；从 v2.3 起它们属于 provider/model candidate history，不自动代表最终主模型。

---

# 1. 最终研究问题

## 1.1 主问题

在相同 CC4 observation、FormalState、动作空间、World Model、response reward、target resolver、adapter 与 paired seeds 下，比较：

```text
LWM-RL
= LLM candidate prior
+ bootstrap ensemble WM foresight
+ PPO posterior plan selection

UG-CEM-APT
= categorical CEM
+ ensemble uncertainty penalty

CEM-APT
= same CEM mechanism with beta=0
```

主要回答：

- 是否缩短 Attack Eradication Time；
- 是否降低 incident-host Host Work Fail；
- 是否提高 cumulative response return；
- 是否保持或改善 CC4 official team return。

## 1.2 多 LLM 子问题

必须回答：

1. 不同 LLM 的 candidate-plan validity、diversity、quality、repeatability 是否不同？
2. 更高的 LLM prior quality 是否转化为更好的 end-to-end PPO performance？
3. WM + PPO posterior 是否能纠正较弱或不完美的 LLM prior？
4. LLM 性能提升相对于 API latency / token / monetary cost 是否值得？
5. Uniform prior / uniform generator 与 LLM prior 的差距多大？

---

# 2. 已冻结领域契约（不得因后续实验改变）

## 2.1 Environment

- CAGE Challenge 4 / CybORG CC4；
- `FiniteStateRedAgent`；
- `EnterpriseGreenAgent`；
- Blue agents=`blue_agent_0..4`；
- `pad_spaces=false`；
- planner 只在 agent ready 的 decision epoch 调用。

## 2.2 FormalState

```text
D = 27
feature 17 = any_valid_observable_target
```

只允许 planner-visible observation / tracker evidence。

禁止 policy/planner/LLM prompt 使用：

- hidden Red sessions；
- true compromise labels；
- future state/reward；
- attack ground-truth labels；
- test information。

## 2.3 High-level action contract

```text
A = 4
0 no_op   -> Sleep            duration=1
1 analyse -> Analyse(host)    duration=2
2 remove  -> Remove(host)     duration=3
3 restore -> Restore(host)    duration=5
```

动作 ID 是 categorical ID，不表示强度。

Targeted action 无合法 observable target 时：

- requested action 保留用于审计；
- runtime 执行 Sleep fallback；
- model-space 使用 shared canonicalizer 转为 executed/canonical action；
- fallback reason 必须记录。

## 2.4 Decision epoch

一个 high-level transition：

```text
s_t
+ requested action
+ actual executed/canonical action
+ real duration dt
+ accumulated real response reward
-> s_next_decision
```

busy agent 不提交 filler action。

## 2.5 Response reward

```text
r_resp
= -1.0 * incident_active_ticks
  + 1.0 * raw incident-host LWF penalty
```

闭合 incident：

```text
T_erad = t_normal - t_compromise
sum(active tick penalties) = T_erad
```

正式 objective 不包含：

- early-warning lead reward；
- fixed action cost；
- lambda_delay；
- checkpoint coverage。

CC4 official team return 始终单独报告。

---

# 3. Seed / split protocol

```text
train       = 1000..1031  (32)
validation  = 2000..2007  (8)
calibration = 3000..3007  (8)
test        = 4000..4019  (20)
```

用途：

- train：replay、WM/reward fitting、PPO training、provisional OOD probe；
- validation：WM audit、LLM prior study、model selection、PPO checkpoint selection、beta/K/H/ablation；
- calibration：UG uncertainty normalizer；
- test：全部冻结后唯一正式 paired evaluation。

严格禁止：

- transition-level random split 代替 episode split；
- calibration/test 更新模型或 normalizer；
- test 调 prompt/model/K/H/PPO/beta；
- 根据 test 结果决定是否重训 WM。

---

# 4. 已完成并冻结的阶段

```text
[x] Gate A — A4/D27/replay/WM/reward/integration
[x] Step 4 — Vectorized SharedRolloutEvaluator
[x] Step 5 — UG-CEM planner
[x] Step 6 — UG uncertainty normalizer
[x] Step 7 — integration smoke
```

Step 7 final smoke：

```text
local_short_calls = 20
local_long_calls = 500
official_seeds = [1000,1001]
official_scenario_ticks = [50,50]
official_post_reset_env_steps = [49,49]
official_decisions = [205,182]
pass = True
```

历史阶段详细记录保留在 `docs/step3-A*.md`, `docs/step4.md`, `docs/step5.md`, `docs/step6.md`, `docs/step7.md`。

---

# 5. 当前 frozen World Model / Reward Predictor

## 5.1 World Model

```text
checkpoint = outputs/world_model_v2/a4_5b/world_model_absolute.pt
state_dim = 27
actions = 4
ensemble M = 5
hidden = 128 x 2
probabilistic diagonal Gaussian
absolute next-state target
independent init/bootstrap/optimizer
train-only normalization
fixed-member deterministic mean rollout
lr = 3e-4
batch = 256
epochs = 50
```

Replay：

```text
train completed transitions      = 9041
validation completed transitions = 2336
validation executed counts:
  Sleep   1641
  Analyse 214
  Remove  233
  Restore 248
```

已有 aggregate validation：

```text
one-step RMSE = 0.132231702 < persistence 0.180830250
H4 RMSE       = 0.191052066 < persistence 0.219133339
H4 uncertainty-error Spearman = 0.687620634
positive validation episodes = 8/8
```

## 5.2 Reward predictor

```text
checkpoint = outputs/world_model_v2/a4_5c/response_reward_predictor.pt
one-step reward RMSE = 2.258064 < train-mean 4.854820
WM+reward H4 value RMSE = 7.626011 < constant baseline 15.522030
H4 value Spearman = 0.670119
positive episodes = 8/8
```

A4.6 integration rerun：

```text
H4 state RMSE = 0.200014838 < 0.219133339
H4 value RMSE = 7.608134616 < 15.522029957
H4 value Spearman = 0.681404826
positive episodes = 8/8
```

当前判断：**planning-ready but not assumed globally accurate**。

---

# 6. Gate B0.1 — Frozen WM Per-Action Final Audit（CURRENT）

## 6.1 任务

在不训练任何参数的前提下，对 frozen WM 做 action-conditioned 与 coverage 审计，确认 aggregate 指标没有掩盖 targeted action 的局部失效。

## 6.2 输入

只允许：

```text
outputs/formal_replay_v2/train.jsonl
outputs/formal_replay_v2/validation.jsonl
outputs/world_model_v2/a4_5b/world_model_absolute.pt
```

train 仅用于 coverage/count provenance；质量 Gate 使用 validation。

## 6.3 设计

对 `no_op/analyse/remove/restore` 分别报告：

### one-step

- train completed count；
- validation completed count；
- WM RMSE / MAE；
- action-specific persistence RMSE / MAE；
- RMSE ratio=`WM/persistence`；
- sample-level epistemic uncertainty vs true error Spearman；
- uncertainty quartile：count、mean uncertainty、mean/median true error。

### H=4 first-action conditioned

按同 agent 连续 decision transitions 构建 H4 window，只以 window 第一个 executed action 分桶：

- window count；
- H4 RMSE/MAE；
- persistence H4 RMSE/MAE；
- uncertainty-error Spearman；
- quartile diagnostics。

### per-agent diagnostic

对 5 Blue agents × 4 actions 报告 count、RMSE、persistence RMSE。小 bucket 只 diagnostic，不单独触发 retraining。

### aggregate reproduction

必须重新计算并复现冻结的：

```text
one-step RMSE
one-step persistence RMSE
H4 RMSE
H4 persistence RMSE
H4 uncertainty-error Spearman
```

## 6.4 禁止项

- 重新 fit normalizer；
- 重新训练 WM；
- 修改 replay；
- 使用 calibration/test；
- 运行后再改 Gate threshold；
- 因某模型后续更喜欢某 action 而对该 action 单独放宽标准。

## 6.5 PASS 条件（运行前冻结）

全部必须满足：

1. checkpoint=`absolute`, D27, A4, M5；
2. validation 四动作 completed count 均 >=150；
3. 所有输出 finite；
4. 每个 action one-step `WM_RMSE/persistence_RMSE <= 1.25`；
5. 至少 3/4 actions 的 one-step WM RMSE <= action persistence；
6. frozen aggregate metrics absolute delta <=5e-4；
7. action family→ID mapping 无 mismatch；
8. 四 action 均有非空 one-step bucket。

H4 first-action bucket 与 per-action Spearman 为 diagnostic：如果出现明显 catastrophic bucket（例如 H4 ratio >2 或 nonfinite），不得静默 PASS，必须人工 reopen review。

## 6.6 FAIL 分支

若硬条件失败且不是实现 bug：

```text
train-only targeted recollection
-> new replay version
-> retrain shared WM
-> 如 reward distribution changed，则同步 retrain reward predictor
-> rerun A4.5/A4.6
-> rerun B0.1
```

不得只给 LWM-RL 单独换 WM。

## 6.7 产物

```text
formal_experiments/evaluation/audit_world_model_final.py
outputs/world_model_v2/b0_1/per_action_audit.json
docs/gate-b-B0-WM-audit.md
tests/test_gate_b0_world_model_audit.py
```

---

# 7. Gate B1 — Response Reward Re-audit

状态：**PASS / CLOSED**。

后续 PPO training reward 必须来自真实 CC4 decision interval 的 accumulated response reward。

禁止：

- predicted WM value 当 PPO reward；
- LLM prior score 当 reward；
- 恢复 action-cost/delay/early-warning reward。

---

# 8. Gate B2 — Multi-LLM Prior Study

B2 目标是先研究 candidate prior，再决定 primary LLM；不允许在实验前把单一 provider/model 写死为最终主模型。

## B2.1 Provider-neutral structural contract

冻结：

```text
A=4
K=6
H=4
planner-visible D27 only
host target not selected by LLM
versioned prompt
strict structured output when provider supports it
local semantic parser always required
duplicate plan -> keep highest score
invalid plan -> reject
shortage -> deterministic neutral fallback
prior scores -> nonnegative linear normalize to sum=1
```

若 provider 不支持某 generation parameter，必须记录为 `unsupported/omitted`，不可伪造或为单个模型单独调优。

## B2.2 Model registry

正式 prior study 前必须创建 model registry，至少 3 个模型：

```text
Tier-H: strong/high-capability
Tier-M: medium-cost
Tier-L: lower-cost/smaller or open-weight-access
```

registry 每项必须记录：

- stable experiment alias；
- provider；
- exact API model ID；
- endpoint family；
- structured-output capability；
- requested/actual temperature；
- tokenizer/usage availability；
- cost source/version if available；
- retry policy；
- model entry freeze timestamp。

当前 OFOX `openai/gpt-5.6-sol` 只是 Tier-H candidate。历史 Gemini-3.1 不自动进入最终三模型集合。

### Registry PASS

- 至少 3 tiers；
- exact IDs 可调用或明确标记 pending-live；
- alias 不可在 state-bank collection 开始后映射到另一模型；
- 所有正式模型共享同一 prompt/A/K/H/parser；
- 无 API key 写入 registry。

## B2.3 Validation state bank

只从 validation seeds `2000..2007` 生成固定 planner-visible D27 state bank。

目标：240 states；选择过程必须 deterministic 且不看 hidden labels/reward。

覆盖约束：

- 5 agents 全覆盖；
- 8 validation seeds 全覆盖；
- 每个 agent×seed 至少一个 state；
- decision order 尽量均匀覆盖 early/mid/late；
- 当某 agent×seed 存在 feature17=0 与1时，两类都应进入 bank；
- duplicate D27 exact states 去重后仍不足目标时，按 deterministic next candidate 补齐。

state bank 保存：

- bank format version；
- seed/agent/decision index/global tick；
- exact D27 float32 state；
- state SHA256；
- source replay/provenance；
- selection config/hash。

禁止保存 hidden truth / future reward / attack labels。

### State-bank PASS

- records=240，除非源数据客观不足并有显式报告；
- seed/agent coverage 完整；
- D27 finite；
- deterministic rerun hash 完全一致；
- no calibration/test；
- no hidden/reward labels。

## B2.4 Prior generation protocol

对每个正式 LLM，在同一 state bank 上生成 K=6/H=4 plans。

默认：

```text
temperature=0.2
max retries=2 per cache miss
retry only on transport/5xx/rate-limit classes
semantic-invalid successful response does not trigger unlimited regeneration
```

API success 与 final PriorBatch validity 分开记录。

## B2.5 Prior-quality metrics

### Reliability

- API success rate；
- schema-valid response rate；
- semantic-valid candidate rate before fallback；
- fallback candidate rate；
- duplicate candidate rate；
- retry/failure rate。

### Diversity

- unique plan ratio；
- per-position action entropy；
- action coverage across candidate set；
- all-no-op / all-restore plan prevalence。

### Frozen-WM candidate quality

使用完全相同 frozen WM/reward evaluator：

- top-prior predicted value；
- best-of-K predicted value；
- mean candidate value；
- value spread；
- mean/max uncertainty；
- prior-score vs predicted-value Spearman；
- top-prior rank under predicted value。

这些指标用于 prior analysis，不等价于真实环境表现。

### Efficiency

- API latency p50/p95；
- input/output token if available；
- monetary cost if available；
- cache hit/miss；
- total live API calls。

## B2.6 Repeatability

从 frozen bank 用 deterministic rule 选 30 states；每模型独立重复 3 次（独立 live generations，不读取第一次 cache）。

报告：

- candidate-set Jaccard/overlap；
- top-prior plan agreement；
- prior-score variance；
- best-of-K predicted-value variance。

repeatability subset 不用于 prompt tuning。

## B2.7 Uniform non-LLM baseline

固定：

```text
state-independent deterministic sample from 4^4 plan space
K=6
uniform prior=1/6
```

使用完全相同 WM evaluation metrics。

## B2.8 B2 完成条件

B2 完成不要求某 LLM “必须赢”，而要求实验完整可靠：

- registry frozen；
- validation state bank PASS；
- 3 LLM + uniform 全部生成完整 report；
- 所有模型 API success/fallback/cost 明确；
- 所有计划最终满足 A4/H4/K6；
- no hidden/test leakage；
- repeatability 完成；
- provider failure 没有被 silent fallback 伪装成 live success。

Primary LLM 只能在 validation prior-quality + 后续 end-to-end validation 完成后确定；prior-quality 单独不能决定最终 winner。

## B2.9 产物

```text
configs/llm_model_registry_v1.yaml
outputs/lwm_rl_v2/b2/state_bank.jsonl
outputs/lwm_rl_v2/b2/state_bank_summary.json
outputs/lwm_rl_v2/b2/<model_alias>/prior_quality.json
outputs/lwm_rl_v2/b2/repeatability/<model_alias>.json
docs/gate-b-B2-multi-llm.md
```

---

# 9. Gate B2.5 — Persistent Prior Cache

Prior cache 是 formal PPO 的必要基础设施，不是可选优化。

## 9.1 Canonical key

主实验禁止 state quantization。Key material 必须包括：

```text
format_version
split
provider
exact model ID
model registry hash
prompt version
A/K/H
generation config
D27 dtype/shape
exact float32 state bytes SHA256
```

最终 key=SHA256(canonical serialized key material)。

## 9.2 Namespace

必须隔离：

```text
train / validation / test
model alias
prompt version
registry version
```

不同 model/split/prompt 不能命中同一 entry。

## 9.3 Value schema

保存：

- cache key + format version；
- state hash；
- split/model/provider/exact model ID；
- prompt version/config hash；
- K/H/A；
- candidate plans；
- raw scores；
- normalized priors；
- candidate source (`llm`/fallback)；
- fallback count；
- API success flag；
- prompt/response hash；
- latency；
- token/cost metadata；
- retry count；
- creation time。

禁止保存：API key、hidden truth、future reward、test labels。

raw response 默认不保存到 formal cache。

## 9.4 Runtime semantics

```text
state -> canonical key
  hit  -> validate entry -> return PriorBatch; provider call count unchanged
  miss -> one provider transaction with bounded retries
       -> validate
       -> atomic temp write + fsync/replace
       -> return PriorBatch
```

API 调用只发生在 cache miss，不发生在 PPO 的每个 gradient epoch。

## 9.5 Corruption / concurrency

- checksum/schema mismatch -> quarantine + hard miss；
- partial file 不得读取；
- atomic replace；
- 同 key 并发写入必须通过 lock 或 deterministic last-equivalent-write；
- incompatible format version hard fail，不 silent migrate。

## 9.6 Cache PASS

- same key deterministic retrieval；
- hit=0 provider calls；
- miss=exactly one logical provider transaction（bounded retries 内部单独计数）；
- model/split/prompt/config change 必须 miss；
- corrupt entry 不可 silent reuse；
- API key serialization scan=0；
- cache roundtrip PriorBatch exact；
- unit tests 覆盖 atomic write/failure recovery。

## 9.7 产物

```text
formal_experiments/ours/prior_cache.py
outputs/lwm_rl_v2/prior_cache/<split>/<model_alias>/...
tests/test_prior_cache.py
docs/gate-b-B2-cache.md
```

---

# 10. Gate B3 — Posterior Candidate Representation

结构已冻结，但 provider-neutral migration 后必须回归。

每 candidate：

```text
state                  27
plan one-hot           16
prior preference        1
predicted value         1
predictive uncertainty  1
--------------------------
feature dim            46
```

```text
value_i = mean(member cumulative response returns)
uncertainty_i = population std(member cumulative response returns)
```

禁止添加：action cost、delay、checkpoint coverage、early-warning、hidden host identity。

B3 regression PASS：shape、finite、permutation-safe inputs、value consistency、prior sum=1、no legacy evidence。

---

# 11. Gate B4 — Formal PPO

## B4.1 Network

采用 candidate-wise shared encoder，避免 flatten K×46 导致 candidate order dependence：

```text
candidate [K,46]
-> shared MLP 46->128 ReLU->128 ReLU

actor: shared scalar head per candidate -> K logits -> Categorical
critic: mean-pool candidate embeddings -> 128 ReLU -> 1
```

性质：

- actor candidate permutation equivariant；
- critic candidate order invariant；
- implementation supports variable K；
- formal main K=6；
- PPO action=candidate index。

初始化：orthogonal；actor output gain=0.01；critic output gain=1.0。

## B4.2 Hyperparameters

```text
lr = 3e-4
rollout_length = 128 decision transitions
update_epochs = 5
minibatch = 64
clip_epsilon = 0.2
GAE lambda = 0.95
entropy_coef = 0.01
value_coef = 0.5
max_grad_norm = 0.5
advantage_normalization = true
gamma_tick = 0.99
```

这些是 formal implementation defaults，不宣称全部来自原论文。

## B4.3 Duration-aware return/GAE

每 transition：

```text
gamma_t = gamma_tick ^ decision_dt
delta_t = real_response_reward_t + gamma_t*V(next) - V(current)
GAE_t = delta_t + gamma_t*lambda*GAE_next
```

lambda 按 decision transition，不使用 `lambda^dt`。

## B4.4 Reward semantics

PPO target 只使用真实 CC4 accumulated response reward。

WM predicted value 只是 observation/evidence，不进入 reward target。

## B4.5 Multi-agent collection

每个 LLM variant 对应一个 shared PPO policy，5 Blue agents 共用参数。

- per-agent 独立 episode/done/GAE boundary；
- update batch 合并；
- busy agent 不产生 transition；
- candidate cache split 必须为 train。

## B4.6 Provisional PPO（用于 B0.2）

只做 pipeline/OOD probe：

```text
one PPO development seed
train environment seeds only
max 20,000 decision transitions
not eligible for paper result
```

provisional weights 在 B0.2 后无论 PASS/FAIL 都不进入 formal training；formal PPO 从新初始化开始。

## B4.7 Formal training

每个 LLM variant × 每 PPO seed：

```text
max 100,000 real environment decision transitions
checkpoint every 10,000
minimum 3 PPO seeds/model
```

所有 LLM 相同 budget 与 seed list。

Checkpoint selection 只看 validation：

主 criterion：mean cumulative response return。

Tie-break：

1. lower mean eradication time；
2. higher official CC4 return；
3. earlier checkpoint（若仍完全相同，避免偏向更多训练）。

不使用 test。

## B4.8 PPO Network/Training PASS

- permutation tests PASS；
- duration-aware GAE reference test PASS；
- no predicted reward leakage；
- finite rollout/update；
- exact seed/config/checkpoint provenance；
- train/validation split enforcement；
- cache model namespace matches policy variant；
- no direct API call during repeated gradient epochs for existing rollout states。

---

# 12. Gate B0.2 — Policy-Induced OOD / Model-Exploitation Audit

在 B4 provisional PPO 后、formal PPO 前执行。

## B0.2.1 Reference distribution

只使用 frozen train/validation replay + frozen WM normalizer。

OOD reference metric：normalized D27 state 的 train kNN distance。

固定：

```text
k = 5
reference threshold = 99th percentile of validation->train kNN distances
```

禁止在 probe 结果出来后移动 percentile。

同时报告 normalized z-RMS distance 作为辅助，不用于唯一 Gate。

## B0.2.2 Probe data

仅 train seeds，由 provisional PPO 在真实 CC4 环境访问并记录真实 transition。

对 probe 计算：

- one-step WM RMSE/MAE；
- per-action count/RMSE；
- state kNN distance；
- OOD fraction above frozen threshold；
- ensemble uncertainty；
- uncertainty-error Spearman；
- top vs bottom uncertainty quartile mean error；
- 能构造时 H4 rollout error。

## B0.2.3 PASS

全部必须满足：

1. train seeds only；
2. no hidden policy input；
3. probe one-step RMSE <=1.25× frozen validation one-step RMSE；
4. uncertainty-error Spearman >0；
5. top uncertainty quartile mean error > bottom quartile；
6. OOD fraction <=10%；
7. 4 action 至少各出现一次；若某 action 缺失，先视为 probe coverage incomplete，不可直接 PASS；
8. real PPO reward 未被 predicted value 替代；
9. no NaN/Inf。

## B0.2.4 FAIL

若为 coverage incomplete：扩大 train-only provisional collection，最多到预先允许的 20k transitions，不改 policy hyperparameter。

若为真实 model shift：

```text
train-only targeted recollection
-> new replay version
-> shared WM/reward refit as required
-> full validation
-> B0.1 rerun
-> discard all provisional PPO/cache entries tied to superseded model-evidence version if evidence changed
-> restart PPO
```

PASS 后当前 WM 正式冻结至最终 comparison 结束。

---

# 13. Gate B5 — PPO Stability

每正式 run 必须记录：

- policy entropy；
- approximate KL；
- clip fraction；
- policy/value loss；
- explained variance；
- grad norm；
- selected candidate prior rank；
- selected predicted-value rank；
- requested/executed action distribution；
- no_op/restore frequency；
- fallback-to-Sleep；
- cache hit rate/API miss count；
- validation response return；
- eradication time/LWF；
- official return。

硬失败：

- NaN/Inf；
- candidate index out of range；
- hidden/test leakage；
- wrong model cache namespace；
- checkpoint selected from test；
- long-run single-action collapse + validation objective simultaneously degrades materially。

低 entropy 本身不是硬失败。

---

# 14. Step 8 — Fair Comparison Harness

统一 method API：

```text
method.observe(shared_state, raw_visible_obs)
method.plan() -> requested high_level_action_id
```

统一后处理：

```text
requested action
-> shared target resolver
-> shared CC4 adapter
-> real environment
```

正式方法：

```text
LWM-RL: LLM/cache -> shared WM evaluator -> PPO -> plan[0]
UG-CEM: Categorical CEM -> same WM evaluator -> UG penalty -> plan[0]
CEM-APT: same as UG, beta=0
```

禁止：method-specific resolver、risk override、action vote、hidden heuristic、test boost。

Harness PASS：相同 seeds/environment config；相同 action canonicalizer/reward; method outputs only requested high-level action; per-method provenance complete。

---

# 15. Step 9 — Validation / Multi-LLM / Sensitivity / Ablation

## 15.1 Multi-LLM prior-quality

```text
LLM-H
LLM-M
LLM-L
Uniform baseline
```

同一 frozen state bank/WM/prompt/A/K/H。

输出独立 prior-quality table；它不单独决定最终 primary model。

## 15.2 End-to-end multi-LLM

每个模型独立训练：

```text
LLM-H + PPO-H
LLM-M + PPO-M
LLM-L + PPO-L
```

共享 PPO architecture/budget/seeds/WM/reward/env splits。

Primary LWM-RL 只能由 validation end-to-end criterion 选出；记录所有 variant，不删除表现差的模型。

## 15.3 UG beta

validation candidates：

```text
0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0
```

beta=0=CEM-APT。

## 15.4 CEM compute

```text
N64/I4
N128/I5
N200/I5
```

记录 performance、latency、plan diversity。

## 15.5 K sensitivity

主实验 K=6；validation-only：K=3/6/10。

先做 prior/greedy-G small validation；如做 end-to-end K ablation，必须单独重训 PPO。

## 15.6 H sensitivity

主实验 H=4；validation-only：H=2/4/6。

报告 rollout error、uncertainty、value spread、latency/cost、greedy-G small environment result。

不因 test 结果改变 H。

## 15.7 Repeatability

30 frozen validation states × 3 independent generations/model。

报告 candidate overlap、top-1 agreement、prior variance、best-of-K value variance。

## 15.8 Ours ablations

必须：

A. Full LWM-RL：LLM generator + prior + value + uncertainty + PPO。

B. w/o prior preference：同 LLM candidates，prior=1/K，PPO 重训。

C. w/o LLM generator：uniform deterministic plan generator + uniform prior，PPO 重训。

D. w/o uncertainty：uncertainty feature=0，PPO 重训。

E. w/o PPO：选择 `argmax predicted_return`，固定 lowest-index tie-break。

可选 F. w/o WM foresight：只有资源允许且替代 evidence 预先冻结才做。

## 15.9 Optional WM ablation

附录可做 bootstrap vs non-bootstrap，仅用于 uncertainty mechanism，不为 Ours 单独寻找更强 WM。

---

# 16. Step 10 — Formal CC4 Test

全部 validation/design 完成后一次性解锁 test `4000..4019`。

正式条件：

- five Blue agents；
- 500 scenario ticks；
- same paired seed/environment；
- no test-time learning/tuning；
- fixed frozen artifacts。

主表至少：Primary LWM-RL / UG-CEM-APT / CEM-APT。

多 LLM end-to-end 表报告所有完成 formal training 的 LLM variants。

---

# 17. Step 11 — Metrics / Statistics

## 17.1 Primary

- Mean Attack Eradication Time；
- incident-host LWF count；
- weighted LWF penalty；
- cumulative response return。

## 17.2 External

- CC4 official team return。

## 17.3 Secondary

- Restore precision；
- action distribution；
- valid-target rate；
- fallback-to-Sleep；
- per-agent；
- planning latency；
- decision_dt。

## 17.4 LLM efficiency

- live API calls；
- cache hit rate；
- p50/p95 latency；
- input/output token；
- monetary cost；
- retry/fallback rate。

## 17.5 Model diagnostics

- aggregate/per-action one-step/H4 WM error；
- policy OOD fraction；
- uncertainty-error calibration；
- value error/rank correlation。

## 17.6 Statistics

paired seeds：

- mean/std/median；
- 95% paired bootstrap CI；
- two-sided paired permutation test；
- paired effect size；
- secondary multiple pairwise comparisons 用 Holm correction。

统计显著性不能替代 effect size/raw distribution。

---

# 18. Step 12 — Final Paper Tables

至少：

- Table A Main: LWM-RL / UG-CEM / CEM；
- Table B Multi-LLM prior quality；
- Table C Multi-LLM end-to-end；
- Table D Ablations；
- Table E WM/reward/OOD quality；
- Table F Efficiency/cost。

论文必须明确：对比 baseline 是 domain-adapted implementation，共享本文 state/action/WM/reward/adapter，而不是原环境逐行复制。

---

# 19. 固定实现顺序

```text
B0.1 WM final per-action audit
  ↓
B2 provider-neutral registry + validation state bank
  ↓
B2.5 persistent prior cache
  ↓
B2 multi-LLM prior-quality/repeatability
  ↓
B3 provider-neutral posterior regression
  ↓
B4.1 PPO network/unit tests
  ↓
B4.6 provisional PPO (train-only)
  ↓
B0.2 policy-induced OOD audit
  ├─ FAIL -> train-only recollect/retrain shared WM/reward -> restart
  └─ PASS
       ↓
formal PPO 3 seeds/model
  ↓
B5 stability
  ↓
Step 8 fair harness
  ↓
Step 9 validation/sensitivity/ablation
  ↓
freeze all configs/checkpoints
  ↓
Step 10 formal test
  ↓
Step 11 statistics
  ↓
Step 12 tables/paper
```

---

# 20. 每阶段强制审核模板

每阶段关闭前必须回答并记录：

1. Task：本阶段精确任务是什么？
2. Inputs：使用哪些 artifact/split？
3. Contract：哪些冻结项不得改变？
4. Leakage：是否接触 hidden/calibration/test？
5. Randomness：所有 seed 是否显式记录？
6. Design：实现是否与任务书一致？
7. Tests：unit/regression 数量与结果？
8. Numerical：所有正式数值是否 finite？
9. Provenance：config/checkpoint/model ID/hash 是否记录？
10. Acceptance：PASS/FAIL 条件是否运行前冻结？
11. Failure branch：FAIL 后是修 bug、reopen Gate 还是扩充数据？
12. Docs：source-of-truth 与阶段文档是否同步？
13. Fairness：是否给某一方法/LLM 增加独享信息、预算或 heuristic？
14. Reproducibility：同输入是否能重现 state-bank/cache/metrics？

---

# 21. 全局禁止项

1. Ours/UG/CEM 使用不同 shared state/WM/reward；
2. test 调 prompt/model/PPO/beta/K/H；
3. hidden Red truth 进入 planner/LLM/PPO；
4. predicted return 代替 PPO real reward；
5. 给单个 LLM 不同 K/H/prompt 以提高成绩；
6. 一个 PPO checkpoint 跨 LLM 作为正式公平结果；
7. cache 跨 model/split/prompt/version 复用；
8. 未验证 state quantization 就用于 formal cache；
9. API failure 被 fallback 伪装成 live API success；
10. method-specific resolver/heuristic override；
11. old A5 replay/checkpoint；
12. early-warning/action-cost/lambda_delay 回流；
13. 根据 test 决定是否重训 WM；
14. 只报告最好 PPO seed；
15. prior-quality WM 指标被解释为真实环境 superiority；
16. provisional PPO 权重进入正式结果。

---

# 22. 当前进度

```text
[x] Gate A
[x] Step 4
[x] Step 5
[x] Step 6
[x] Step 7
[x] Gate B1 response reward re-audit
[x] Gate B3 structural candidate feature contract (provider-neutral regression pending)

[~] Gate B0.1 WM final per-action audit        <- CURRENT
[ ] Gate B2 registry/state bank
[ ] Gate B2.5 prior cache
[ ] Gate B2 multi-LLM prior-quality/repeatability
[ ] Gate B3 provider-neutral regression
[ ] Gate B4 PPO network
[ ] Gate B4 provisional PPO
[ ] Gate B0.2 OOD/model-exploitation audit
[ ] Gate B4 formal PPO 3-seed/model
[ ] Gate B5 stability
[ ] Step 8 fair harness
[ ] Step 9 validation/multi-LLM/sensitivity/ablation
[ ] Step 10 formal test
[ ] Step 11 statistics
[ ] Step 12 paper tables
```

---

# 23. 当前一句话原则

```text
先证明 frozen WM 对四动作可靠，
再用同一 validation state bank 公平比较多个 LLM，
用严格隔离的 persistent cache 控制 API 成本与可重复性，
每个 LLM 用同预算独立训练 PPO，
用 provisional PPO 检查 policy-induced model shift，
最后在完全共享的 CC4/WM/reward/adapter 与 paired test seeds 下比较 LWM-RL、UG-CEM 与 CEM。
```
