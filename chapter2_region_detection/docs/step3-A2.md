# Step 3-A2 — Gate A Local-online 动作适配审核记录

> 项目：UG-CEM-APT  
> 分支：`ug-cem-apt`  
> 阶段：Gate A / A2 — Local-online Action Adapter  
> 审核日期：2026-09-15  
> 最终状态：**PASS / 已完成**

---

## 1. 本阶段目标

A2 的目标是让 Gate A1 已冻结的 5 类高层动作在开发期 local-online 环境中具有**彼此区分、可学习、且不依赖动作 ID 数值大小**的 transition semantics。

A2 分为：

```text
[x] A2.1  Local action semantics
[x] A2.2  Shared local-online transition client
```

A2 不负责：

- 正式 CybORG / CC4 action mapping；
- BlockTraffic 的真实 subnet 参数；
- 正式 reward；
- action cost；
- multi-tick duration locking；
- world-model 重训；
- PPO / CEM / uncertainty 逻辑。

---

## 2. Git 范围审核

相对 A1 完成点，本阶段只新增：

```text
chapter2_region_detection/shared/local_action_adapter.py
chapter2_region_detection/shared/local_online_client.py
chapter2_region_detection/tests/test_gate_a_local_action_adapter.py
chapter2_region_detection/tests/test_gate_a_local_online_client.py
```

审核确认：

- 未修改 `chapter2_region_detection/src/`；
- 未修改 UG 官方源码；
- 未修改 Categorical CEM；
- 未修改 UG uncertainty；
- 未修改 world model；
- 未修改旧 `src/cc4_client.py`；
- 未提前混入 CybORG adapter；
- 未提前冻结 reward / action cost。

因此阶段边界符合 A2 要求。

---

## 3. A2.1 — Local action semantics

新增：

```text
shared/local_action_adapter.py
```

内部私有状态：

```text
auth_pressure
scan_pressure
process_pressure
outbound_pressure
visibility
```

这些变量只属于 local simulator 内部，不属于正式 8 维 planner state。

五类动作语义：

### no_op

```text
不主动修改 latent threat state
```

自然攻击演化由 local environment 自己负责。

### analyse

```text
visibility ↑
```

不直接清除 auth / scan / process / outbound threat。

这对应：

```text
analyse = intrusion investigation
```

### control_traffic

主要降低：

```text
scan_pressure
outbound_pressure
```

不直接清除：

```text
auth_pressure
process_pressure
```

### remove

主要降低：

```text
auth_pressure
process_pressure
```

不直接改变：

```text
scan_pressure
outbound_pressure
```

### restore

主要降低：

```text
auth_pressure
process_pressure
outbound_pressure
```

但不直接消除外部：

```text
scan_pressure
```

---

## 4. 不把 action ID 当连续强度

A2 不再使用旧 local client 中类似：

```python
abs(action_id - target_action)
```

或：

```python
eff[action_id]
```

来表达动作相似度/强度。

A2 先通过 A1：

```python
get_action(action_id)
```

获得动作语义，再按：

```text
no_op / analyse / control_traffic / remove / restore
```

分别处理。

因此：

```text
0 < 1 < 2 < 3 < 4
```

不具有连续控制强度含义。

审核结果：**PASS**。

---

## 5. A2.2 — Shared local-online client

新增：

```text
shared/local_online_client.py
```

核心 transition 顺序固定为：

```text
current latent state
    ↓
natural threat evolution
    ↓
apply high-level action
    ↓
generate observable events
    ↓
derive risk_proxy from observable events
    ↓
next observation
```

即 replay 语义可以稳定表示为：

```text
(s_t, a_t, s_{t+1})
```

这为后续重新采集 action-consistent world-model replay 提供了基础。

---

## 6. Partial observability / latent-state isolation

A2 明确区分：

```text
private latent simulator state
```

与：

```text
planner-visible observation
```

正式 observation 仅包含：

```text
t
events
risk_proxy
region_id
region_name
meta
```

不会返回：

```text
auth_pressure
scan_pressure
process_pressure
outbound_pressure
visibility
latent_state
```

`info` 同样不暴露这些 latent variables。

测试代码为了白盒验证 transition 可以读取 `_latent_state`，但这不是 planner observation 接口。

审核结果：**PASS**。

---

## 7. analyse 与 visibility

A2 将 `analyse` 实现为“提高可见性”，而不是“直接降低真实威胁”。

事件生成概率使用：

```text
latent pressure × visibility factor
```

因此 analyse 后有可能观察到更多异常事件，甚至使 observation-level `risk_proxy` 上升。

该现象表示：

> 调查动作发现了之前不可见的威胁，而不是调查动作制造了威胁。

另外 visibility 会向配置的 floor 衰减，避免一次 analyse 产生永久完全观测。

审核结果：**PASS**。

---

## 8. risk_proxy 不直接泄漏真实 threat

A2 的 `risk_proxy` 由可观测 events 的：

```text
mean severity
max severity
event density
```

组合得到。

它没有直接使用：

```text
mean(auth_pressure, scan_pressure, process_pressure, outbound_pressure)
```

作为 planner 输入。

因此 local simulator 保留了部分可观测性，避免把真实 latent threat 直接泄漏给规划器。

审核结果：**PASS**。

---

## 9. StateSummary 兼容性

A2.2 生成的 event type 与当前统一状态摘要兼容：

```text
auth_fail
port_scan
proc_spawn
outbound_conn
```

经：

```text
SlidingWindowSummarizer
```

仍得到：

```text
D = 8
```

的统一状态：

```text
cnt_auth_fail
cnt_port_scan
cnt_proc_spawn
cnt_outbound_conn
avg_severity
unique_src
unique_dst
risk_proxy
```

A2 没有修改正式 state dimension。

审核结果：**PASS**。

---

## 10. Reward 边界

A2.2 的：

```python
reward = 0.0
```

是 Gate B 前的显式 placeholder。

同时 `info["reward_status"]` 标记：

```text
placeholder_zero_until_gate_b
```

A2 没有复制旧 local client 中依赖：

```text
risk improvement
action-ID mismatch
action cost
overreaction
early-warning
```

的 reward。

正式 reward 仍由 Gate B 冻结。

审核结果：**PASS**。

---

## 11. 可复现性

A2.2 使用 seed 控制 `numpy.RandomState`。

本地测试覆盖：

- 同 seed reset 一致；
- 同 seed + 同 action sequence trajectory 一致；
- 不同 region profile 具有不同 latent 初始状态。

审核结果：**PASS**。

---

## 12. 测试

用户本地执行 Gate A 测试：

```bash
python -m unittest discover -s tests -p "test_gate_a_*.py"
```

结果：

```text
Ran 31 tests
OK
```

组成：

```text
A1      9
A2.1    9
A2.2   13
----------------
total  31
```

A2.1 覆盖：

- no-op 无主动效果；
- analyse 只增加 visibility；
- control_traffic 网络定向效果；
- remove 用户级失陷效果；
- restore 主机恢复效果；
- 动作不是数值强度级别；
- 输入 state immutable；
- 非法 latent state；
- 非法 adapter config。

A2.2 覆盖：

- observation schema；
- latent-state leakage；
- reset reproducibility；
- full action-sequence reproducibility；
- natural dynamics；
- analyse semantics；
- control_traffic semantics；
- remove semantics；
- restore semantics；
- Gate B reward placeholder；
- episode termination；
- StateSummary D=8 compatibility；
- region distinction；
- invalid config。

GitHub source audit 与用户报告的测试设计一致。

---

## 13. 非阻塞加固项

当前没有阻塞 A3 的缺陷。

后续可选工程加固包括：

- 对 `threat_drift`、`threat_noise_std`、`burst_scale`、`event_rate` 增加显式 finite 校验；
- 更严格拒绝可被 `int()` 静默截断的非整数 config；
- 后续集成时避免正式 planner/experiment 直接读取带下划线的 `_latent_state`。

这些属于健壮性增强，不改变当前 A2 transition contract，因此不阻塞 A2 PASS。

---

## 14. A2 不代表正式 CybORG 环境

A2 的 local-online simulator 只用于：

- 单元测试；
- 联调；
- replay 重新采集前的 transition 验证；
- CEM / uncertainty / MPC smoke test。

最终论文正式结果仍必须使用统一 CybORG / CAGE Challenge 4 环境。

A2 中人为定义的 local transition 参数不能被描述为 CC4 官方动作动力学。

---

## 15. 尚未完成的 Gate A

当前：

```text
[x] A1  Shared Action Contract
[x] A2  Local-online Action Adapter
[ ] A3  CybORG / CC4 Action Adapter
[ ] A4  Replay / World-model Compatibility + Cost/Delay
[ ] A5  Gate A Final Review
```

下一阶段 A3 必须解决：

- `no_op` 的真实 CybORG no-operation execution；
- `analyse` host target；
- `remove` host target；
- `restore` host target；
- `control_traffic` 的真实 BlockTraffic / subnet-pair 参数；
- action availability / action-space label 验证；
- Ours 与 UG 共用同一个 official adapter。

---

## 16. 最终审核结论

```text
A2.1 action semantics             PASS
A2.2 transition client            PASS
Categorical action treatment      PASS
Latent-state isolation            PASS
Observed risk construction        PASS
StateSummary compatibility        PASS
Reward boundary                   PASS
Seed reproducibility              PASS
Legacy src isolation              PASS
Stage scope                       PASS
Gate A tests (31)                 PASS

FINAL STATUS: PASS
```

**Gate A2 正式关闭。**

当前下一阶段：

```text
Gate A3 — CybORG / CC4 Action Adapter
```


---

## 17. 2026-09-15 方案变更说明（v2.1）

由于正式高层动作空间从 A=5 修订为 A=4，本文件记录的五动作 local-online adapter PASS 仅作为历史审计记录。

仍可复用的语义：

- no_op；
- analyse；
- remove；
- restore；
- partial observability；
- latent-state isolation；
- reproducibility。

需要在 A2R 删除：

- control_traffic transition；
- control_traffic tests；
- A=5 action sequence assumptions。

本阶段状态更新为：

```text
HISTORICAL PASS / SUPERSEDED BY A2R
```

后续以 `UG_CEM_APT_REPRODUCTION_PLAN.md v2.1` 与 `step3-A2R.md` 为准。
