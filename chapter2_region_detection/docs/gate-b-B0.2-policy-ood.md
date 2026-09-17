# Gate B0.2 — Policy-Induced OOD / Model-Exploitation Audit

日期：2026-09-17  
状态：**SOURCE READY / LOCAL TEST PENDING / LIVE PROBE PENDING**

## 1. Purpose

在 formal PPO 前验证当前 shared frozen WM 是否仍然可靠地覆盖由 LLM+PPO 实际诱导出的真实 CC4 state-action distribution。

本 Gate 不评价 provisional PPO 的任务性能，也不允许 provisional checkpoint 进入论文正式结果。

## 2. Required variants

必须分别审计：

```text
llm_h_gpt56_sol + provisional PPO-H
llm_m_gpt54_mini + provisional PPO-M
llm_l_gemini35_flash_lite + provisional PPO-L
```

共享 WM 对三种正式 LLM variant 都是公平性锚点，因此 **每个 variant 必须单独满足 Gate**。

Pooled union 结果只做 diagnostic；不得用 pooled PASS 覆盖某个 per-model FAIL。

## 3. Frozen reference

只使用：

```text
outputs/formal_replay_v2/train.jsonl
outputs/formal_replay_v2/validation.jsonl
outputs/world_model_v2/a4_5b/world_model_absolute.pt
```

OOD reference：

```text
frozen WM train normalizer
k = 5
validation -> train normalized D27 kNN distance
threshold = validation distance 99th percentile
```

Threshold 在任何 provisional result 出现前冻结。

## 4. Probe input

每个 model variant 的：

```text
outputs/lwm_rl_v2/b4/provisional/<model_alias>/probe_transitions.jsonl
```

只允许 train seeds `1000..1031`。

每条 probe 是真实 CC4 decision transition，包含 planner-visible state/next_state、canonical executed action、真实 reward/dt/done 和 policy/cache provenance；不得包含 hidden Red truth。

## 5. One-step diagnostics

对 action-completed probe transitions：

- overall WM RMSE / MAE；
- per canonical A4 count / RMSE / MAE；
- epistemic uncertainty vs true error Spearman；
- top/bottom uncertainty quartile mean true error；
- normalized state 5-NN distance；
- OOD fraction；
- normalized z-RMS auxiliary diagnostic。

## 6. H4 diagnostic

按：

```text
(model_alias, episode_ordinal, agent_name)
```

构造连续 4 个 completed decision transitions，使用真实 canonical actions 做 fixed-member deterministic H4 rollout。

报告 H4 RMSE / MAE / uncertainty-error Spearman。

H4 为 diagnostic，不新增事后 Gate threshold。

## 7. Frozen hard PASS conditions

每个 model variant 全部必须满足：

```text
train seeds only
probe one-step RMSE <= 1.25 * frozen validation one-step RMSE
uncertainty-error Spearman > 0
top uncertainty quartile mean error > bottom quartile mean error
OOD fraction <= 10%
all four canonical A4 actions have >=1 completed transition
real PPO reward not replaced by predicted reward
all numerical values finite
```

状态定义：

```text
PASS
  hard model-shift checks all pass
  AND A4 coverage complete

COVERAGE_INCOMPLETE
  all hard model-shift checks pass
  BUT one or more A4 actions absent

FAIL
  any hard model-shift check fails
```

`COVERAGE_INCOMPLETE` 才允许按预先冻结 stages 从 2k 扩展到 5k/10k/20k；`FAIL` 不允许靠收更多数据把结果平均掉。

## 8. All-model decision

最终 shared-WM B0.2：

```text
PASS iff H PASS AND M PASS AND L PASS
```

若任一 variant FAIL：overall FAIL。

若无 FAIL 但至少一个 COVERAGE_INCOMPLETE：overall COVERAGE_INCOMPLETE。

## 9. Late-exploitation diagnostic

为避免全程平均掩盖训练后段 drift，额外冻结：

```text
last 25% probe transitions / model
```

重复报告：

- one-step RMSE / validation ratio；
- uncertainty-error Spearman；
- uncertainty quartile error；
- canonical action counts；
- OOD fraction / kNN distribution。

该 tail report **diagnostic only / affects_gate=false**。它用于论文解释或决定是否人工 reopen review，但不能事后改变 frozen numerical PASS threshold。

## 10. Outputs

```text
formal_experiments/evaluation/audit_b0_2_policy_ood.py
formal_experiments/evaluation/audit_b0_2_all_models.py
outputs/lwm_rl_v2/b0_2/<model_alias>.json
outputs/lwm_rl_v2/b0_2/all_models.json
tests/test_gate_b0_2_policy_ood.py
```

## 11. Failure branch

### Coverage incomplete

继续同一 provisional policy lineage 到下一 cumulative stage，最多 20k；不改 PPO hyperparameter / prompt / K / H。

### True model shift / exploitation

```text
stop provisional collection
-> train-only targeted recollection
-> new replay version
-> retrain one shared WM/reward stack as required
-> rerun A4 validation
-> rerun B0.1
-> invalidate evidence/cache tied to superseded WM evidence version where required
-> restart all formal PPO from fresh initialization
```

不得只为表现差的 LLM 换一个更强 WM。
