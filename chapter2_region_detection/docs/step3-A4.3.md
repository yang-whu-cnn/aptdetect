# Step A4.3 — Incident Bookkeeping + Response Reward 审核记录

日期：2026-09-16  
状态：PASS

## Git 范围

实现提交：

`d1468237167c5338963613ab414a49b065fcb687`

相对 A4.2 完成点仅变更：

- `chapter2_region_detection/formal_experiments/data_collection/incident_response.py`（新增）
- `chapter2_region_detection/formal_experiments/data_collection/decision_replay.py`
- `chapter2_region_detection/tests/test_gate_a_incident_response.py`（新增）
- `chapter2_region_detection/tests/test_gate_a_decision_replay.py`

## 源码审核

A4.3 正式冻结：

- 每个 Blue agent / region 独立维护 incident bookkeeping；
- 每个 host 使用 False -> True 打开新的 incident event；
- True -> False 关闭当前 host event；
- 同一 host 再次 compromise 会获得新的 incident ordinal / event ID；
- 多 host 并发 compromise 作为多个独立 incident，同时累计 incident-active ticks；
- tick 区间使用 interval-start presence 计时，使累计 active ticks 与已完成事件的 `t_normal - t_compromise` 对齐；
- 只有 active incident host 的 GreenLocalWork failure 计入 incident-host LWF；
- 其他 host LWF 不计入当前 incident-specific response objective；
- response reward 为 `-lambda_time * incident_active_ticks + lambda_failure * raw_LWF_penalty`，其中 CC4 raw LWF penalty <= 0；
- hidden compromise truth 明确限定于 reward/evaluation bookkeeping，不进入 FormalState / LLM / PPO / CEM / UG-CEM planner state；
- replay 现在累计 `incident_event_ids` / `incident_host_ids`，可以审计跨多 tick、并发 incident 的 decision interval。

## 测试

用户报告结果与冻结预期一致：

- A4.3 incident response：15 tests，OK；
- A4.2 replay（含新增 incident-ID case）：16 tests，OK；
- 完整 Gate A：110 tests，OK。

## 方案确认

该实现与当前小论文的 response objective 一致：

- attack eradication time；
- normal operation failure / Host Work Fail；
- 四动作与 decision-epoch 执行语义保持不变。

A4.3 不把 official CC4 aggregate team reward 当作训练 response objective；official reward 仍单独保留用于外部评价。

## 结论

FINAL STATUS: PASS

下一步：A4.4 Bootstrap Probabilistic Ensemble World Model。
