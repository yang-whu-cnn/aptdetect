# Step A4.1c — Valid Observable Target Availability Correction 审核记录

日期：2026-09-16  
状态：PASS

## Git 范围

实现提交：

`a449ce6a795ce2375f11420be5131ef16bd21959`

相对 corrective-design 冻结点仅修改：

- `chapter2_region_detection/shared/cyborg_target_resolver.py`
- `chapter2_region_detection/shared/formal_state.py`
- `chapter2_region_detection/formal_experiments/data_collection/collect_cc4_formal_replay.py`
- `chapter2_region_detection/tests/test_gate_a_cyborg_target_resolver.py`
- `chapter2_region_detection/tests/test_gate_a_formal_state.py`

未提交 replay、checkpoint、A4.6a evaluator 或其它输出。

## 源码审核结论

- FormalState 维度保持 D=27；
- feature 17 从 `any_observable_target` 修正为 `any_valid_observable_target`；
- feature 17 只由 current wrapper action_labels/action_mask 与 ObservableHostEvidenceTracker host scores 推导；
- FormalStateEncoder 本身仍不读取 env/controller；
- resolver availability 与 production resolver 共用 `valid_actions()` 过滤语义；
- availability 判断使用 observable scored hosts 与当前 valid targeted actions 的交集；
- collector 在 decision state 与 next decision state 两处均使用新 availability；
- collector 对 requested action 与 production adapter fallback 做 exact runtime assertion；
- Analyse/Remove/Restore availability 若出现 family divergence，collector fail-fast，不强行压成一个 bit；
- hidden Red truth / incident truth 仍只用于 IncidentResponseBookkeeper 与 reward/evaluation；
- 其它 26 维 FormalState 语义保持原 contract；
- resolver tests 14 个，FormalState tests 24 个。

官方 frozen CC4 BlueFixedActionWrapper 对所有其它 host-based commands 使用相同 host-exists + Blue-session validity rule，因此在当前环境中使用一个 shared valid-target bit 与 wrapper 语义一致；若未来 family availability 分歧，代码会显式失败。

## User-side smoke runtime evidence

seed=1000, steps=100：

- transitions = 320；
- requested = 80 / 80 / 80 / 80；
- executed Sleep/Analyse/Remove/Restore = 248 / 21 / 23 / 28；
- valid_target_rate = 0.300；
- fallback_rate = 0.525；
- incident_host_count = 23。

该统计与旧 seed=1000 collection 相同，说明 corrective state semantic change 没有改变 exploration / resolver / async scheduling trajectory。

独立 replay mapping audit：

- count = 320；
- overall accuracy = 1.0；
- targeted count = 240；
- targeted accuracy = 1.0；
- feature=1 but fallback = 0；
- feature=0 but non-fallback = 0；
- nonbinary feature count = 0；
- Analyse / Remove / Restore 均 80/80 canonical mapping correct。

重点 tests 与完整 Gate A 均由用户报告 OK。

FINAL STATUS: PASS

## 下一步

旧 formal replay、A4.5b WM checkpoint 与 A4.5c reward checkpoint 因 feature 17 语义改变而 superseded，不得与新 replay 混用。

先重新采集完整 train 1000..1031 与 validation 2000..2007，并在训练前对新 replay 做全量 exact requested→executed mapping audit。之后重新运行 A4.5b 的 absolute/delta selection。不得预设 absolute 一定再次获选；只有 A4.5b 新结果确定 selected target mode 后，才进入 A4.5c rerun。
