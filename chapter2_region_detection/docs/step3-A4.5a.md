# Step A4.5a — Formal CC4 Replay Collection 审核记录

日期：2026-09-16  
状态：IMPLEMENTATION PASS / FORMAL COLLECTION PENDING

## Git 范围

实现提交：

`a27fbeb4061460fd54a228297716ae7ce80787fa`

相对 A4.5a protocol 冻结点仅新增：

- `chapter2_region_detection/formal_experiments/data_collection/collect_cc4_formal_replay.py`
- `chapter2_region_detection/tests/test_gate_a_formal_replay_collection.py`

未修改 adapter、resolver、formal state、incident reward、decision replay、world model、CEM、UG uncertainty 或 PPO。

## 源码审核

正式 collection runner 满足冻结要求：

- 使用真实 `EnterpriseGreenAgent` + `FiniteStateRedAgent` + `BlueFixedActionWrapper`；
- 不使用 probe-only false-positive、service-reliability 或 deterministic attack injection；
- planner-visible 路径仅使用 Blue observation -> ObservableHostEvidenceTracker -> FormalStateEncoder / observable_host_scores；
- hidden controller red-session truth 仅进入 IncidentResponseBookkeeper；
- GreenLocalWork failure 仅用于 LWF reward bookkeeping；
- 五个 Blue agent 独立 scheduler-local async decision epochs；
- executed duration 从当前 underlying action object 的 `.duration` 获取；
- requested action 使用 deterministic 4-action round-robin exploration；
- targeted action 只用 observable host evidence，无法合法 target 时真实 fallback Sleep；
- replay 同时记录 requested / executed action、fallback、decision_dt、response reward、official reward、incident IDs、next state；
- calibration/test split 无法通过 A4.5a CLI 正式采集；
- train / validation / calibration / test seed pools 相互独立。

## 单元 / 回归测试

用户报告：

- A4.5a 单元测试：7 tests，OK；
- 完整 Gate A：133 tests，OK。

## 真实 CC4 smoke

seed=1000，steps=100：

- transitions = 320；
- completed = 317；
- terminal incomplete = 3；
- requested action count = 80 / 80 / 80 / 80；
- executed = Sleep 248 / Analyse 21 / Remove 23 / Restore 28；
- targeted requested = 240；
- valid targeted = 72；
- valid target rate = 0.30；
- fallback = 168；
- fallback rate = 0.525；
- incident hosts = 23；
- all five Blue agents have replay transitions。

Full JSONL consistency audit：

- state / next_state all 27D finite；
- per-agent decision_index contiguous；
- per-agent decision intervals contiguous；
- chained next_state == next transition state；
- completed transition dt == executed duration；
- fallback always executes Sleep；
- incomplete transitions only occur at terminal；
- response_reward exactly matches frozen incident-time + LWF equation；
- smoke provides 312 valid H=2 and 302 valid H=4 complete contiguous windows。

## Optimization decision

No collection-policy optimization is required before formal collection.

Reason:

- requested actions are exactly balanced by construction；
- executed imbalance is a real consequence of partial observability and no-valid-target fallback, not a collection bug；
- using hidden attack truth to increase targeted-action execution would violate the no-leakage contract；
- one smoke episode already provides nonzero Analyse / Remove / Restore execution and usable H=4 trajectories；
- final action coverage must be judged after all 32 train episodes rather than tuned on one episode.

## Remaining A4.5a requirement

Implementation and smoke are complete, but A4.5a is not finally closed until the frozen full replay is collected and audited:

- train seeds 1000..1031 (32 episodes)；
- validation seeds 2000..2007 (8 episodes)；
- steps=100；
- calibration/test remain untouched。

FINAL IMPLEMENTATION STATUS: PASS  
A4.5a FORMAL DATA STATUS: PENDING
