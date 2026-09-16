# Step 3-A3.6 — CC4 Multi-Agent Integration 审核记录

日期：2026-09-16  
状态：REOPENED

## Git 范围

相对 A3.5 完成点，本阶段仅新增：

`chapter2_region_detection/formal_experiments/probes/probe_cc4_multi_agent_integration.py`

未提交 JSON 输出，也未修改 adapter、resolver、CEM、UG uncertainty、WM 或 PPO。

## 代码审核

Probe 在真实 CC4 `BlueFixedActionWrapper` 中覆盖：

- blue_agent_0 ... blue_agent_4；
- seed 42 / 43 / 44；
- pad_spaces=False / True；
- no_op / analyse / remove / restore 四个 high-level actions；
- 同一 `CybORGActionAdapter` 和 resolver；
- mask=True 后的独立 probe oracle；
- targeted action 的 deterministic synthetic observable scores；
- wrapper label / mask / underlying action object / hostname 一致性；
- multi-tick controller behavior 与 decision availability。

Probe 中固定的 82 / 242 action-space size 仅作为 A3.1 已验证后的测试 oracle，不进入 production adapter。

## 真实结果

总计：

- 6 runs；
- 24 joint action cases；
- 120 adapter resolutions；
- all_pass=True。

pad_spaces=False：

- blue_agent_0..3: action space 82；
- blue_agent_4: action space 242。

pad_spaces=True：

- 五个 Blue agent 均为 action space 242。

blue_agent_4 同时覆盖 admin / office / public_access 三个子网，无需 adapter 特判。

四动作 decision_dt 保持：

- Sleep 1；
- Analyse 2；
- Remove 3；
- Restore 5。

所有 targeted resolution 均无意外 fallback，最终底层 action family / target 与 wrapper 一致。

## 复核更正：异步 multi-agent readiness 仍缺失

后续 Gate 复核发现，本 probe 当前只验证了“五个 Blue agent 同时提交同一个 action family，并在后续 busy ticks 统一 actions={}”的同步场景。

A3.4 已冻结的正式要求还包括异步 multi-agent readiness：不同 agent 执行不同 duration 的动作时，某个 agent 一旦先完成，应能独立进入下一 decision epoch，而其他 busy agent 必须继续省略，不能用 Sleep 作为 busy filler。

因此原先 A3.6 PASS 结论过早。已有同步测试结果仍然有效，但不足以关闭 A3。

必须补充同一 probe 的 async extension：

- blue_agent_0: Sleep -> Restore；
- blue_agent_1: Analyse -> Remove；
- blue_agent_2: Remove -> Analyse；
- blue_agent_3: Restore -> Sleep；
- blue_agent_4: Analyse -> Restore；
- scheduler readiness 只能由本地已提交动作的 executed duration 推导；
- controller actions_in_progress / action 只能作为 oracle assertion，不能决定何时提交；
- fallback 时 readiness 使用 executed action duration；
- busy agent 不提交任何 action；
- 每次新 launch 都重新读取 labels/mask，并重新走 adapter；
- seed 42/43/44 × pad false/true 共 6 个 async runs。

预期 launch schedule：

- t=0：五个 agent 首次 launch；
- t=1：blue_agent_0 第二次 launch；
- t=2：blue_agent_1、blue_agent_4 第二次 launch；
- t=3：blue_agent_2 第二次 launch；
- t=5：blue_agent_3 第二次 launch。

## 结论

同步 multi-agent integration：PASS。

异步 multi-agent readiness：PENDING。

FINAL STATUS: REOPENED

A3.6 async extension 通过前，不正式关闭 A3；已完成的 A4.1-A4.4 实现保留，不需要回退。
