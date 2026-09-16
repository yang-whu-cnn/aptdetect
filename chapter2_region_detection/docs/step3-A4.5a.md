# Step A4.5a — Formal CC4 Replay Collection 审核记录

日期：2026-09-16  
状态：PASS

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

## Frozen formal replay results

Train split（seeds 1000..1031, 32 episodes, steps=100）：

- transitions = 9041；
- completed = 8952；
- terminal incomplete = 89；
- requested = 2237 / 2260 / 2276 / 2268；
- executed Sleep / Analyse / Remove / Restore = 6114 / 925 / 984 / 1018；
- valid-target rate = 0.430188；
- fallback rate = 0.428824；
- all five Blue agents covered；
- incident-host coverage = 97。

Validation split（seeds 2000..2007, 8 episodes, steps=100）：

- transitions = 2336；
- completed = 2315；
- terminal incomplete = 21；
- requested = 577 / 584 / 588 / 587；
- executed Sleep / Analyse / Remove / Restore = 1641 / 214 / 233 / 248；
- valid-target rate = 0.395111；
- fallback rate = 0.455479；
- all five Blue agents covered；
- incident-host coverage = 81。

Train / validation requested-action distributions remain essentially identical by construction. Executed-action distributions are also close enough for held-out validation: validation Sleep is about +2.62 percentage points vs train, while each targeted family differs by roughly 0.6–1.1 percentage points. No data-collection policy change is justified.

Terminal incomplete rates are below 1% in both splits and are excluded from standard dynamics training by A4.4 contract.

Calibration seeds 3000..3007 and test seeds 4000..4019 remain untouched.

## Final conclusion

A4.5a formal replay collection is complete.

No collection-policy optimization is required before A4.5b. Action imbalance is treated as a real partial-observability / fallback property and will be diagnosed per executed action during WM validation instead of being corrected with hidden-truth-guided collection.

FINAL STATUS: PASS

Next: A4.5b held-out WM validation / rollout / uncertainty calibration.
