# Step A4.1a — Observable State Contract Probe 审核记录

日期：2026-09-16  
状态：PASS

## Git 范围

相对 A3 完成点，本阶段只新增：

`chapter2_region_detection/formal_experiments/probes/probe_cc4_observable_state_contract.py`

未提交 probe JSON，也未修改 adapter、resolver、CEM、UG uncertainty、WM 或 PPO。

## 代码审核

Probe 使用真实 CC4 `BlueFixedActionWrapper` raw Blue observation 作为 planner-visible 信息来源。

允许来源：

- reset observation 中的 host inventory / IP->hostname；
- post-reset observation 中的 Processes；
- Processes[*].Connections；
- Files。

明确禁止 planner 使用：

- controller true state；
- red sessions；
- true compromise labels；
- future information；
- A3 synthetic host scores。

Probe-only controller access 仅用于把 EnterpriseGreenAgent 的 false-positive rate 设为 1.0，以稳定地产生真实 Monitor observation；该信息不进入 planner evidence。

## 真实结果

seed 42 / 43 / 44，五个 Blue agent 均获得真实 Monitor host-level evidence。

每个 seed 中，Analyse target 均来自刚刚观察到的 alert host，未使用 true-state target。

汇总：

- runs=3；
- agents_per_run=5；
- all_agents_observed_monitor_evidence=True；
- analyse_used_only_observable_targets=True；
- hidden_truth_used_for_planner=False。

## 重要发现

reset observation 中正常 host 自带初始 Processes，因此：

`reset Processes != threat evidence`

正式 `ObservableHostEvidenceTracker` 必须把 reset observation 仅作为 inventory/baseline；只有 reset 之后的 Monitor / Analyse observation 才能产生 threat evidence。

## 论文一致性

当前设计与论文方法一致：论文以告警、日志及关联主机形成 incident information / vector；A4.1 将其落实为真实 CC4 Blue-observable state。四动作、LLM candidate plans、world-model rollout、PPO posterior plan selection 均未改变。

observable-only 限制属于实现上的防信息泄漏约束，不改变论文方法。

CC4 multi-tick action 的“下一时刻重规划”落实为 next decision epoch，是对真实动作 duration 的环境适配。

FINAL STATUS: PASS

下一步：A4.1b 正式 ObservableHostEvidenceTracker + FormalStateEncoder。
