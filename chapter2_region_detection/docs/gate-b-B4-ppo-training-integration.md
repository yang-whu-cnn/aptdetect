# Gate B4 — PPO Rollout / Trainer Integration

日期：2026-09-17  
状态：**PASS / CLOSED**

---

## 1. Preconditions

```text
B4 PPO core = PASS / CLOSED
B3 posterior = PASS / CLOSED
B2 cache / multi-LLM = PASS / CLOSED
```

本 Gate 仍不执行 formal PPO training。

## 2. Purpose

把已经冻结的 PPO 数学 core 接到正式 decision-epoch 数据契约上，但先保持 environment/provider 解耦。

Production source：

```text
formal_experiments/ours/ppo_training.py
```

## 3. Asynchronous rollout semantics

每个 Blue agent 独立维护一个 open PPO decision。

在 agent ready 时冻结：

```text
episode_seed
agent_name
decision_index
candidate_features [6,46]
selected candidate index
behavior log-prob
critic V(s)
```

动作执行期间 busy agent 不创建 filler transition。

decision interval 结束后只补入：

```text
real_response_reward
real decision_dt
done
critic V(next)
```

Terminal 必须 `critic_next_value=0`。

禁止 rollout step 保存或使用 predicted WM reward / LLM reward。

## 4. Split boundary

正式/provisional PPO collection 默认 `train_only=true`：

```text
allowed seeds = 1000..1031
```

validation/calibration/test seed 进入训练 buffer 必须 hard fail。

Validation 只用于后续 checkpoint selection/evaluation，不用于 optimizer rollout collection。

## 5. Multi-agent GAE

训练 batch 可以合并 5 agents，但 GAE 必须先按：

```text
(episode_seed, agent_name)
```

分组，并按 decision_index 排序。

禁止：

- 一个 agent 的 GAE continuation 跨到另一个 agent；
- terminal 后继续同一 trajectory；
- decision-index gap 静默通过。

每个 trajectory 使用 B4 core 冻结的 `gamma_tick**dt` GAE；随后可在合并 batch 上做一次 global advantage normalization。

## 6. PPO update isolation

`PPOTrainer.update(batch)` 是纯优化函数，只接收 frozen batch。

它不得：

- 调 OFOX/OpenAI；
- 访问 PriorCache；
- 创建 CC4 environment；
- 修改 candidate plans；
- 重算 LLM prior；
- 使用 hidden truth。

因此 5 update epochs 不产生额外 API transaction。

Optimizer：Adam，lr=3e-4；每 minibatch 做 max-grad-norm=0.5 clipping。

## 7. Unit tests

```text
tests/test_gate_b4_ppo_training.py
```

12 tests：

1. frozen train-seed contract；
2. async begin/complete；
3. busy duplicate / decision-index gap guard；
4. terminal next-value=0；
5. non-train seed / malformed feature/action rejection；
6. per-agent trajectory isolation；
7. non-contiguous / post-terminal rejection；
8. global advantage normalization after per-agent GAE；
9. frozen training defaults；
10. finite optimizer update + parameter change；
11. optimizer has no provider/environment dependency；
12. explicit real-response-reward field / no predicted-reward field。

### Local acceptance record

用户于 2026-09-17 在 `.venv_cc4` 本地环境执行正式测试并报告：

```text
Ran 12 tests
OK
```

因此本 Gate 的 local-test 条件已满足。

## 8. PASS

```text
12/12 tests PASS
train split hard guard PASS
async per-agent contract PASS
terminal bootstrap guard PASS
per-agent GAE isolation PASS
finite optimizer update PASS
no provider/API dependency in optimizer PASS
```

结论：**PASS / CLOSED**。

## 9. After PASS

下一层：

```text
one-model tiny CC4 pipeline smoke
```

该 smoke 需要真实串联：

```text
D27 state
-> train PriorCache (miss may call one exact LLM)
-> frozen WM/reward rollout
-> 46D candidates
-> PPO select candidate
-> execute plan[0] through shared resolver/adapter
-> real CC4 interval reward/dt
-> PPO rollout buffer
-> one tiny PPO update
```

验收重点：

- plan[0] 是实际 requested A4 action；
- fallback/canonical action 仍由 shared resolver/adapter 管理；
- PPO reward 是真实 response reward；
- cache namespace=`train/<matching model alias>`；
- repeated optimizer epochs 不调用 API；
- per-agent boundaries 正确；
- no test/calibration；
- finite update。

通过 tiny smoke 后才允许进入 provisional PPO；根据 2026-09-17 的设计一致性审计，后续 B0.2 将对三个正式 LLM variant 分别做 train-only provisional probe，而不是用单一模型代表全部三种 policy-induced state distribution。
