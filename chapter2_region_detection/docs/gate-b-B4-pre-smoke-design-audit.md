# Gate B4 — Pre-Smoke Design Consistency Audit

日期：2026-09-17  
状态：**PASS / AMENDMENT FROZEN**

本文件记录在 one-model tiny CC4 pipeline smoke 前，对 v2.3 总任务书、论文研究问题、当前 B0–B4 实现做的一次一致性审计。它只澄清已经实际执行的 multi-LLM 方案与 runtime provenance，不改变论文主研究问题、D27/A4/WM/reward/PPO 核心方法。

## 1. 审计结论

主线一致：

```text
LWM-RL
= LLM candidate prior
+ shared frozen bootstrap world-model foresight
+ PPO posterior candidate-plan selection
```

保持不变：

- CC4 / five Blue agents；
- D27 FormalState；
- A4 high-level action；
- K=6, H=4；
- shared frozen M=5 world model / reward predictor；
- PPO selects candidate index and executes selected plan[0]；
- PPO reward only from real accumulated CC4 response reward；
- train/validation/calibration/test split isolation；
- three LLM variants train independent PPO policies；
- UG-CEM-APT / CEM-APT share the same model-space/environment contracts。

## 2. Amendment A — public agent identifier in prompt/cache identity

实际 B2 prompt 从一开始就包含：

```text
agent_name = blue_agent_0..4
```

该字段是公开 controller identity，不是 hidden Red truth、future reward、attack label 或 host target。B2 prior-quality / repeatability 已在这一 prompt contract 下完成，因此禁止在 B4 临时删除它，否则会改变实验 prompt 并使 B2 与 end-to-end training 不可对应。

正式解释修订为：

```text
planner evidence = D27 only
additional prompt context = public Blue-agent identifier only
```

禁止增加其他环境上下文。

由于 prompt 文本依赖 agent identifier，online prior cache identity 必须同时依赖 agent identifier。实现允许把该字段编码进 cache generation/prompt-context identity；任何两个不同 agent 即使 exact D27 相同，也必须得到不同 cache key。

历史 B2 validation state bank 为 240/240 exact unique D27，因此旧 B2 cache 未发生 cross-agent exact-state collision；已完成 B2 结果无需重跑。后续 online/provisional/formal PPO 必须使用修订后的 agent-aware cache identity。

## 3. Amendment B — registry-driven formal configuration

`configs/lwm_rl_gate_b_v2_1.yaml` 不再把 GPT-5.6 Sol 写成默认 primary model。

正式 source：

```text
configs/llm_model_registry_v1.yaml
primary_model_selected = false
```

三正式 variants：

```text
llm_h_gpt56_sol
llm_m_gpt54_mini
llm_l_gemini35_flash_lite
```

Primary LWM-RL 只能在 end-to-end validation 后选择。

## 4. Amendment C — B0.2 must cover all three LLM-induced policies

共享 WM 对所有正式 LLM variants 都是公平性锚点。因此 policy-induced OOD/model-exploitation audit 不能只检查一个 LLM/PPO policy。

正式 B0.2 probe 改为：

```text
LLM-H + provisional PPO-H
LLM-M + provisional PPO-M
LLM-L + provisional PPO-L
```

每个 variant：

- fresh provisional PPO initialization；
- train environment seeds only；
- same PPO architecture/hyperparameters；
- max 20,000 real decision transitions/variant；
- provisional checkpoint 永不进入 formal result。

B0.2 的 frozen reference threshold 和 PASS threshold 对三个 variant 完全相同。任一正式 variant 出现真实 WM shift/model exploitation，则不能宣布 shared WM 对正式 multi-LLM comparison 全部可靠。

如果某 variant 只是 A4 coverage incomplete，可在自己的 20k 上限内继续 train-only probe，不改 hyperparameter。

## 5. Paper-method alignment

该修订增强而不改变论文因果问题：

1. prior-only B2 比较 LLM candidate quality/repeatability；
2. independent PPO 比较 LLM prior 在 posterior adaptation 后的 end-to-end effect；
3. shared frozen WM 保证 planner evidence 公平；
4. per-variant B0.2 防止某个 LLM/PPO 特有 state distribution 利用 WM blind spot；
5. API cost/latency 仍作为 efficiency outcome，不进入 PPO observation/reward。

## 6. Tiny-smoke entry condition

进入 one-model tiny CC4 smoke 前必须满足：

- B4 core PASS；
- B4 rollout/trainer integration PASS；
- registry-driven config committed；
- agent-aware runtime cache key test PASS；
- selected smoke model 只是工程 probe，不代表 primary selection；
- train seed only；
- no validation/calibration/test；
- real reward/dt collection；
- plan[0] through shared resolver/adapter；
- optimizer epochs do not trigger API calls。
