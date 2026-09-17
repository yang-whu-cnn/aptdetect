# Gate B3 — Provider-Neutral Posterior Candidate Representation Regression

日期：2026-09-17  
状态：**PASS / CLOSED**

---

## 1. Preconditions

Closed before B3：

```text
Gate B0.1 frozen WM audit             PASS / CLOSED
Gate B2 candidate registry            PASS / CLOSED
Gate B2 validation state bank         PASS / CLOSED
Gate B2 persistent cache              PASS / CLOSED
Gate B2 live capability               PASS / CLOSED
Gate B2 240-state prior quality       PASS / CLOSED
Gate B2 repeatability                 PASS / CLOSED
```

Primary LLM remains unselected.

## 2. Purpose

B3 does not add new evidence and does not choose a model. It proves that the posterior observation used by the later PPO remains identical across all LLM variants after the provider-neutral/multi-model migration.

The posterior depends only on：

```text
current planner-visible state
candidate plan
normalized LLM prior preference
shared-WM predicted value
shared-WM predictive uncertainty
```

No provider/model identity is an input feature.

## 3. Frozen candidate feature

Per candidate：

```text
current_state             27
candidate_plan one-hot    16
prior preference           1
predicted value            1
predictive uncertainty     1
--------------------------------
total                     46
```

Formal main：

```text
K = 6 candidates
H = 4 actions/plan
A = 4 categorical high-level actions
feature tensor = [6,46]
```

Column layout is frozen：

```text
[0:27]   repeated D27 current state
[27:43]  H4×A4 categorical one-hot plan
[43]     normalized prior preference
[44]     expected cumulative response return
[45]     predictive epistemic uncertainty
```

## 4. Value and uncertainty semantics

For candidate `i` with M=5 shared bootstrap-WM member returns：

```text
value_i = mean(member_returns_i)
uncertainty_i = population_std(member_returns_i)
```

Population standard deviation means `unbiased=False`.

`rollout_result.expected_return` must numerically equal the member-return mean; otherwise hard fail.

## 5. Provider-neutrality

`build_posterior_candidate_features()` accepts only：

```text
state
plans
prior_preferences
rollout_result
```

It does not accept：

```text
provider
model ID
model tier
API metadata
latency/cost
reason/rationale text
```

Therefore GPT-5.6 Sol, GPT-5.4 Mini and Gemini 3.5 Flash Lite use the same posterior feature construction. Model identity affects the candidate plans/priors generated upstream, not the feature schema downstream.

## 6. Candidate permutation property

If the same permutation is applied to：

```text
plans
prior preferences
member returns / expected returns
```

then the resulting posterior tensor must be exactly the same permutation of candidate rows.

This is required before B4 uses a shared candidate-wise encoder.

## 7. Input hard guards

Hard fail on：

- state shape not `[27]`；
- non-finite state；
- plans shape not `[6,4]`；
- action ID outside `[0,3]`；
- prior shape not `[6]`；
- negative/non-finite prior；
- prior sum not approximately 1；
- expected return shape not `[6]`；
- member returns not `[6,M]` with `M>1`；
- non-finite rollout evidence；
- expected-return/member-mean inconsistency。

## 8. Explicitly forbidden evidence

B3 source contains no model/provider-specific feature and no legacy evidence：

```text
action cost
checkpoint coverage
delay/lambda_delay
early warning
control_traffic
light/heavy evidence
local/strong mitigate evidence
hidden host identity
```

## 9. No new normalizer in B3

B3 is a structural regression gate. It does not introduce a new posterior-feature normalizer after observing the B2 results.

Any later PPO preprocessing change requires an explicit taskbook amendment before changing formal semantics. The frozen B3 evidence definition remains unchanged.

## 10. Source

Production source remains：

```text
formal_experiments/ours/posterior_features.py
```

Regression suite：

```text
tests/test_gate_b3_posterior_regression.py
```

## 11. Local acceptance — PASS

用户本地执行：

```bash
python -m unittest tests.test_gate_b3_posterior_regression -v
```

结果：

```text
Ran 10 tests
OK
```

10 tests 覆盖：

1. exact 27+16+1+1+1=46 contract + config components；
2. exact feature column layout；
3. provider-neutral function signature across all 3 registry models；
4. candidate permutation equivariance；
5. value mean + population-std uncertainty；
6. inconsistent/non-finite rollout rejection；
7. prior probability guards；
8. plan shape/action-ID guards；
9. state shape/finite guards；
10. source scan for provider-specific and legacy evidence。

## 12. PASS conclusion

```text
B2 Multi-LLM Prior Study     PASS / CLOSED
B3 posterior schema          46D / FROZEN
B3 provider-neutrality       PASS
B3 candidate permutation     PASS
B3 local regression          10/10 PASS
Primary LLM selected         NO
Test seeds touched           NO
```

下一阶段：

```text
Gate B4.1 candidate-wise PPO network + unit tests
Gate B4.3 duration-aware GAE + reference tests
```

**FINAL STATUS: PASS / CLOSED**
