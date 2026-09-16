# Step 3-A3.5 — Incident Bookkeeping Probe 审核记录

日期：2026-09-16  
状态：PASS

## Git 范围

相对 A3.4 完成点，本阶段只新增：

`chapter2_region_detection/formal_experiments/probes/probe_cc4_incident_bookkeeping.py`

没有提交 probe JSON，也没有修改 adapter、resolver、CEM、UG uncertainty、WM 或 PPO。

## 代码审核结论

Probe 明确区分：

- hidden ground truth：仅 reward / evaluation bookkeeping 使用；
- planner observation：禁止读取 ground-truth red presence；
- 当前 incident host 的 LWF；
- 其他 host 的 LWF；
- official CC4 aggregate team reward。

IncidentBookkeeping 冻结：

- host red presence False -> True：记录 t_compromise；
- host red presence True -> False：记录 t_normal；
- attack eradication time = t_normal - t_compromise；
- 仅 hostname == incident_host_id 的 GreenLocalWork failure 计入 incident-host LWF；
- 其他 host 的 LWF 不改变当前 incident count / penalty。

## 真实 probe 结果

seed=42：

- incident host: restricted_zone_a_subnet_user_host_0
- t_compromise=1
- t_normal=8
- attack_eradication_time=7
- incident_host_lwf_count=1
- incident_host_lwf_raw_penalty=-1.0
- other_host_lwf_official_team_penalty=-1.0
- other_host_lwf_counted_for_incident=False
- official team return over probe ticks=-3.0

Restore duration=5；前 4 ticks red presence 保持 True，第 5 tick 完成后变为 False。

## 冻结语义

论文中的 Host Work Fail 对应 CC4 源码 LWF (Local Work Fails)，触发来源为 GreenLocalWork failure。

正式指标只累计当前 incident host 的 LWF count / weighted penalty。ASF、RIA、其他 host 的 LWF 与 aggregate team reward 不得混入该 incident-specific metric。

Official CC4 team return 单独保留为外部评价指标。

FINAL STATUS: PASS

下一步：A3.6 multi-agent integration。
