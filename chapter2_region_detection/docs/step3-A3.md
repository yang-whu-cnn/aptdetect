# Step 3-A3 — Official CybORG / CC4 Four-Action Adapter 汇总审核记录

日期：2026-09-16  
状态：**PASS / A3 CLOSED**

---

## 1. 本阶段目标

A3 的目标是在真实 CybORG / CC4 中冻结正式四动作执行契约，使后续 Ours、UG-CEM-APT、CEM-APT 共享同一真实执行接口。

正式高层动作：

```text
0 no_op   -> Sleep
1 analyse -> Analyse(host)
2 remove  -> Remove(host)
3 restore -> Restore(host)
```

`control_traffic` 不属于 v2.1 正式动作空间。

## 2. 子阶段记录

A3 已完成并分别记录：

- A3.1：`docs/step3-A3.1.md`
- A3.2 / A3.3：`docs/step3-A3.2-A3.3.md`
- A3.4：`docs/step3-A3.4.md`
- A3.5：`docs/step3-A3.5.md`
- A3.6 synchronous：`docs/step3-A3.6.md`
- A3.6 asynchronous extension：`docs/step3-A3.6-async.md`

## 3. 冻结结果

### 3.1 Action contract

四动作均可由共享 adapter 解析到真实 CC4 action。`no_op` 始终映射为 `Sleep`。

### 3.2 Shared host resolver

`Analyse / Remove / Restore`：

- 共用同一个 target resolver；
- 只使用当前 planner-visible observation；
- 只选择 wrapper 当前 valid labels / mask 中的 host；
- deterministic tie-breaking；
- 无合法 observable target 时执行 fallback `Sleep`；
- requested action 与真实 executed action 分开记录。

### 3.3 Duration / decision epoch

冻结真实动作持续时间：

```text
Sleep   : 1 tick
Analyse : 2 ticks
Remove  : 3 ticks
Restore : 5 ticks
```

planner 只在对应 agent 可发起新动作时进入新的 decision epoch；busy agent 不使用 Sleep 作为 filler。

### 3.4 Reward / evaluation bookkeeping source

A3.5 冻结：

- hidden red-presence truth 只允许用于 reward/evaluation bookkeeping；
- 不进入 planner state；
- `t_compromise` 为 incident host 首次 False -> True 的 global tick；
- `t_normal` 为首次 True -> False 的 global tick；
- attack eradication time = `t_normal - t_compromise`；
- 论文 Host Work Fail 对应 CC4 LWF / Local Work Fails；
- 只累计当前 active incident host 的 GreenLocalWork failure。

### 3.5 Multi-agent

覆盖 `blue_agent_0..4`。同步与异步场景均已验证。

异步 scheduler 只依据各 agent 本地已提交动作的 executed duration 判断 readiness；不同持续时间动作可以独立完成并重新进入 decision epoch。

## 4. A3 验收项

- A=4 official adapter：PASS；
- 无固定跨-agent action index 假设：PASS；
- deterministic shared resolver：PASS；
- fallback 语义冻结：PASS；
- duration 冻结：PASS；
- decision-epoch 语义冻结：PASS；
- reward bookkeeping source 冻结：PASS；
- five-agent synchronous integration：PASS；
- five-agent asynchronous readiness：PASS；
- hidden ground truth 不进入 planner：PASS。

## 5. 最终结论

```text
A3 — Official CybORG / CC4 Four-Action Adapter

Action contract          : PASS
Shared target resolver   : PASS
Fallback semantics       : PASS
Decision-epoch duration  : PASS
Reward bookkeeping source: PASS
Multi-agent sync         : PASS
Multi-agent async        : PASS
No planner leakage       : PASS

FINAL STATUS: PASS / A3 CLOSED
```

A3 关闭后进入 A4 — Formal State / Replay / Bootstrap WM / Response Reward。