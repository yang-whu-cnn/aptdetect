# Gate B4 — Multi-LLM Provisional PPO / B0.2 Probe Collection

日期：2026-09-17  
状态：**SOURCE READY / STRICT WRAPPER LOCAL TEST PENDING**

## 1. 目的

本阶段只为 B0.2 policy-induced OOD / model-exploitation audit 产生真实 CC4 policy visitation 数据，不产生正式论文 PPO 权重。

必须分别运行：

```text
llm_h_gpt56_sol + fresh provisional PPO-H
llm_m_gpt54_mini + fresh provisional PPO-M
llm_l_gemini35_flash_lite + fresh provisional PPO-L
```

三个 variant 使用完全相同的 PPO architecture / hyperparameters / train-seed schedule / frozen WM / reward predictor / resolver / adapter / response reward。

## 2. 严格边界

- train seeds only: `1000..1031`；
- 不使用 validation/calibration/test；
- 每个 model variant fresh policy initialization；
- provisional weight 永远 `formal_result_eligible=false`；
- primary LLM 仍不选择；
- API 只在 train cache miss 时调用；
- PPO optimizer epoch 不调用 provider；
- PPO reward 只用真实 CC4 accumulated response reward；
- hidden Red truth 只允许进入既有 incident-response reward bookkeeping，不进入 state/prompt/posterior/policy。

## 3. 预算与 staged probe

任务书硬上限：

```text
max 20,000 real decision transitions / model variant
```

为避免不必要 API 成本，预先冻结 cumulative probe stages：

```text
2,000 -> 5,000 -> 10,000 -> 20,000
```

第一阶段默认只收集到不超过 2,000 transitions。每个 stage 后运行 B0.2：

- 若 PASS：停止该 variant 的 provisional collection；
- 若仅 action coverage incomplete：继续到下一 frozen stage；
- 若发生真实 model-shift hard FAIL：停止，不用更多 API 掩盖失败；
- 任何 stage 都不得超过 20,000。

该 staged rule 在看到任何 B0.2 probe 结果前冻结。

## 4. Episode / seed schedule

- deterministic round-robin train seeds `1000..1031`；
- default scenario length `100` ticks；
- final episode 可缩短，保证 cumulative completed transitions 不超过当前 stage target；
- 每个 episode 的实际 seed / ticks / transitions 必须记录；
- 同 variant resume 时必须验证 frozen protocol manifest 完全一致。

正式执行入口：

```text
formal_experiments/evaluation/run_b4_provisional_stage.py
```

低层 collector：

```text
formal_experiments/evaluation/run_b4_provisional_ppo.py
```

只作为内部实现，不作为正式手工执行入口。

Strict wrapper 不暴露 `--steps` / `--ppo-seed` / PPO 超参覆盖；首次运行写：

```text
outputs/lwm_rl_v2/b4/provisional/<model_alias>/protocol_manifest.json
```

manifest 冻结：scenario steps、PPO seed、stage targets、rollout target、PPO core config、PPO training config、train-seed schedule。resume 时逐字段完全匹配，否则 hard fail。

## 5. Async rollout update barrier

正式 PPO config 仍为 `rollout_length=128` decision transitions。

CC4 为异步 multi-agent，因此禁止在第 128 条 transition 完成时、仍有其他 open decision 的情况下更新 policy。冻结规则：

```text
rollout_target = 128 completed decision transitions
update allowed iff:
  completed_since_last_update >= 128
  AND AsyncPPORolloutBuffer.open_agents == empty
  AND formal replay has no open decision at the same tick barrier
```

若 barrier 时 batch >128，记录 actual batch size / overshoot；所有这些 transitions 必须来自同一 behavior-policy version。

Policy update 后：

- clear only completed rollout steps；
- 保留 per-seed/per-agent next decision index provenance；
- invalidate any prefetched next-state context created under the old critic；
- re-prepare on next decision; prior should hit cache if exact same state/model/agent。

不得让一个 open transition 跨 policy version。

## 6. Probe transition artifact

每个 completed real decision transition 保存到 JSONL，至少包含：

```text
model_alias
policy_version
episode_seed
agent_name
decision_index
global_tick_start/end
state D27
next_state D27
requested_action_id
canonical_action_id
executed_action_family
fallback
selected_candidate_index
selected_plan
real response_reward
decision_dt
done
cache_key/cache_hit
```

这份 artifact 是 B0.2 的唯一 policy-induced probe 输入；不得保存 hidden Red truth。

Strict wrapper 在 stage 结束后重新读取完整 probe JSONL，重算 cumulative requested/canonical action counts、fallback、agent/seed counts、response reward、official reward 与 plan[0] match。不得用“当前进程 this-run counter”代替 cumulative audit summary。

## 7. Training diagnostics

每个 PPO update 记录：

- behavior-policy version；
- batch size / rollout target / overshoot；
- policy/value loss；
- entropy；
- approx KL；
- clip fraction；
- grad norm；
- cache hit/miss；
- live API count；
- requested/canonical action distribution；
- fallback-to-Sleep；
- cumulative real response reward。

## 8. Checkpoint / resume

允许保存 development-only resume checkpoint：

```text
outputs/lwm_rl_v2/b4/provisional/<model_alias>/resume.pt
```

必须标记：

```text
development_only=true
formal_result_eligible=false
```

Checkpoint 只为中断恢复和 B0.2 probe continuation；B0.2 结束后不得复制为 formal PPO 初始化。

任何 stage > 2,000 必须通过 strict wrapper 的 `--resume` 从前一 cumulative checkpoint 继续；不得 fresh-start 到 5k/10k/20k。

## 9. Stage PASS（collection gate）

一个 provisional collection stage 的工程 PASS 只表示数据可用于 B0.2，不表示 WM/B0.2 PASS。必须满足：

1. train-only；
2. requested action == selected plan[0] 100%；
3. PPO/replay reward/dt/done alignment 100%；
4. canonical action 与 shared model-space canonicalizer 一致；
5. finite PPO update；
6. no open transition crosses a policy update；
7. optimizer 前后 API counter 不变；
8. probe JSONL count == completed PPO transitions；
9. current stage budget 未超过；
10. primary model remains unselected；
11. strict protocol manifest exact match；
12. cumulative probe summary count == stage transition count。

## 10. 下一步

每个 model stage collection PASS 后，立即运行 Gate B0.2 audit。B0.2 thresholds 完全沿用 v2.3 总任务书，不根据 probe 结果修改。
