# Gate B — B1–B3 Formal Prior / Posterior Contract

日期：2026-09-16  
状态：**SOURCE READY / LOCAL TEST PENDING**

---

## 1. Scope

本子阶段只冻结 Gate B 的 B1–B3：

- B1 response reward contract re-audit；
- B2 four-action LLM prior v2.1；
- B3 PPO posterior candidate observation。

B4 PPO network / training / checkpoint selection 尚未开始；B5 stability 也尚未开始。

legacy `src/llm_prior.py / src/ppo_posterior.py / src/plan_eval.py / src/env_region.py` 不修改、不复用为正式实现。

## 2. B1 response reward contract

继续使用 Gate A 已冻结 objective：

```text
r_resp = -1.0 * incident_active_ticks
         + 1.0 * raw incident-host LWF penalty
```

不包含：

- early-warning lead reward；
- fixed action cost；
- lambda_delay；
- checkpoint coverage。

`IncidentResponseBookkeeper` 使用 interval-begin presence 计算 active tick：

```text
False -> True at t+1: [t,t+1] 不计 active
True  -> False at t+1: [t,t+1] 仍计 active
```

因此闭合事件满足：

```text
sum(active tick penalties)
= t_normal - t_compromise
= attack eradication duration
```

Gate-B test 使用：

```text
False->True @1
True->False @4
```

实证：累计 active ticks=3，event `T_erad=4-1=3`。

## 3. B2 LLM prior v2.1

新增：

```text
formal_experiments/ours/llm_prior_v2.py
configs/lwm_rl_gate_b_v2_1.yaml
```

正式继承原论文参数、按当前小论文四动作域适配：

```text
K = 6 candidates
H = 4 high-level decisions
thesis model label = Gemini-3.1
current API model ID = gemini-3.1-pro-preview
generation temperature = 0.2
```

正式 action vocabulary：

```text
0 no_op
1 analyse
2 remove
3 restore
```

prompt 只使用 planner-visible FormalState D=27；不选择 host target，host target 仍由后续 shared resolver 决定。

prompt 明确禁止：

- hidden compromise truth；
- future reward；
- attack label；
- test information。

## 4. Parser / duplicate / fallback freeze

LLM 输出格式冻结为 strict JSON：

```json
{
  "candidates": [
    {
      "actions": ["no_op", "analyse", "remove", "restore"],
      "prior_score": 0.8,
      "reason": "..."
    }
  ]
}
```

parser 规则：

- 每条 plan 必须 exact H=4；
- 只接受 4 个 canonical names；
- `prior_score` 必须 finite 且位于 [0,1]；
- unknown action / wrong length / invalid score 直接丢弃；
- exact duplicate plan 去重，保留最高 prior_score；
- valid plans 按 score descending、plan lexicographic tie-break；
- 最多取 K=6。

prior preference normalization：

```text
nonnegative linear normalize to sum=1
```

全部 raw score 为 0 时使用 uniform 1/6。

fallback：

```text
deterministic_uniform_plan_space_sample
plan space = 4^4 = 256
fixed seed = 20260916
state-dependent = false
hidden-truth-dependent = false
fallback raw prior score = 0
```

它不是 risk heuristic，不读取状态，不偏向某个响应动作；只用于 API/parse 不完整时补齐 K。

## 5. B3 posterior observation

新增：

```text
formal_experiments/ours/posterior_features.py
```

PPO action 不是高层 response action，而是：

```text
candidate plan index ∈ {0,...,5}
```

每个 candidate 的正式 posterior feature：

```text
current state             27
candidate plan one-hot    4*4 = 16
LLM prior preference       1
predicted value             1
predictive uncertainty      1
--------------------------------
candidate feature dim      46
```

不增加旧 evidence：

- Cost；
- checkpoint coverage C；
- delay；
- early-warning。

## 6. Value / uncertainty definitions

直接复用 A4.5c 已验证的 H=4 member-return machinery：

```text
value_i
= mean_m member_return[i,m]

uncertainty_i
= population_std_m member_return[i,m]
```

其中 member return 来自：

- exact same v2 absolute WM；
- exact same shared response reward predictor；
- fixed-member H=4 mean rollout；
- duration-aware `gamma_tick=0.99`。

posterior feature builder 强制检查 `expected_return == member_return mean`，防止后续 evaluator 接错。

## 7. B4 pretrain boundary already recorded

Gate-B config 只提前锁定原论文明确给出的 PPO 参数：

```text
learning_rate = 0.0003
rollout_length = 128
update_epochs = 5
clip_epsilon = 0.2
GAE lambda = 0.95
entropy coefficient = 0.01
```

discount 不沿用旧论文前瞻 `gamma=0.9`；当前正式系统继续使用 Gate A 已冻结的：

```text
gamma_tick = 0.99
duration-aware discount = true
```

train seeds=1000..1031；validation=2000..2007；calibration/test forbidden。

尚未由论文明确给出的 PPO hidden/minibatch/value coefficient/network structure 不在 B1–B3 偷偷冻结，留到 B4 正式实现时单独记录。

## 8. Legacy audit

legacy formal-use flags 全部 false：

```text
src_llm_prior_allowed: false
src_ppo_posterior_allowed: false
src_plan_eval_allowed: false
src_env_region_allowed: false
old_a5_checkpoint_allowed: false
```

静态 source audit 确认新的 formal prior / posterior source 中没有：

- control_traffic；
- monitor/light_evidence/heavy_evidence；
- local_mitigate/strong_mitigate；
- action_cost；
- lambda_delay；
- checkpoint_coverage。

## 9. Tests

新增：

```text
tests/test_gate_b_llm_prior_posterior_contract.py
```

共 15 tests：

1. A=4/K=6/H=4/model/temp contract；
2. prompt four-action + D27 contract；
3. valid 6-candidate parse / preference normalization；
4. duplicate highest-score retention；
5. invalid JSON neutral deterministic fallback；
6. unknown action / wrong H / bad score rejection；
7. fallback unique 4^4 formal plan-space sample；
8. plan categorical one-hot 16D；
9. value mean + return-std uncertainty；
10. final candidate feature dim=46；
11. prior / rollout evidence guard；
12. B1 active-tick == T_erad executable proof；
13. no old cost/delay/extra evidence；
14. legacy module exclusion；
15. PPO pretrain seed/gamma/thesis-parameter boundary。

## 10. Local Gate

在 `.venv_cc4`、`chapter2_region_detection`：

```bash
python -m unittest tests.test_gate_b_llm_prior_posterior_contract -v
```

预期：

```text
Ran 15 tests
OK
```

通过后 B1–B3 CLOSED，再进入 B4 PPO network / training pipeline。

## 11. Current conclusion

```text
B1 reward re-audit          : READY
B1 T_erad equivalence test  : READY
B2 four-action prior        : READY
B2 strict parser            : READY
B2 duplicate/fallback       : READY
B3 posterior feature        : READY
B3 value/uncertainty        : READY
Legacy isolation            : READY
Local execution             : PENDING

FINAL STATUS: SOURCE READY / LOCAL TEST PENDING
```
