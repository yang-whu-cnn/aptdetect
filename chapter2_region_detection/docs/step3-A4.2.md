# Step A4.2 — Decision-Epoch Replay Schema / Collector 审核记录

日期：2026-09-16  
状态：PASS

## Git 范围

实现提交：

`ba3acf83f9ac6fb4659bab42201d8983bbb6b183`

相对 A4.1 完成点仅新增：

- `chapter2_region_detection/formal_experiments/data_collection/decision_replay.py`
- `chapter2_region_detection/tests/test_gate_a_decision_replay.py`

未修改 formal state、action adapter、CEM、UG uncertainty、WM 或 PPO。

## 源码审核

A4.2 已冻结 decision-epoch replay 的结构语义：

- replay 单位是“一个高层决策到下一 decision epoch”，不是逐 tick policy transition；
- Sleep / Analyse / Remove / Restore 的执行区间分别对应 1 / 2 / 3 / 5 tick；
- multi-agent 可异步维护独立 open decision；
- busy agent 不重复开启 decision；
- requested high-level action 与 executed low-level action 同时保存；
- fallback 时使用 executed action duration，而不是 requested nominal duration；
- 非 terminal 情况只能在真实 next decision epoch 关闭 transition；
- terminal mid-action 可保存 actual decision_dt < nominal duration，并标记 action_completed=False；
- state / next_state 固定为 A4.1 的 27 维 float32 finite vector；
- interval official reward 与 response-reward bookkeeping 字段按 tick 累加；
- JSONL 可序列化输出。

## 测试

新增 15 个 replay 单元测试，覆盖：

- Sleep dt=1；
- Analyse dt=2；
- fallback Restore -> Sleep 使用 executed duration=1；
- 非终止 busy action 提前关闭拒绝；
- terminal mid-action；
- final tick accounting；
- multi-agent asynchronous epochs；
- interval reward / LWF 聚合；
- 非法 LWF raw penalty；
- duplicate open；
- contiguous decision index；
- resolution agent mismatch；
- state copy；
- JSON serialization；
- state dimension validation。

用户报告 A4.2 专项测试与完整 Gate A 回归均为 OK。

## A4.3 延迟冻结项

A4.2 中的：

- incident_active_ticks；
- incident_host_lwf_count；
- incident_host_lwf_raw_penalty；
- response_reward；

当前只是 replay 容器字段，不负责生成语义。

`incident_event_id / incident_host_id` 以及“多个并发 incident 如何映射到每个 decision interval”的正式语义，必须在 A4.3 Incident Bookkeeping + Response Reward 中一起冻结后再接入 replay。这样避免在尚未定义并发 incident 规则前把错误的一对一 event schema 固化。

## 结论

A4.2 的 decision-epoch / asynchronous replay collector 通过。

FINAL STATUS: PASS

下一步：A4.3 Incident Bookkeeping + Response Reward。
