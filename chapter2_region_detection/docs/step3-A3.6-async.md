# Step A3.6 Async Extension — CC4 Multi-Agent Readiness 审核记录

日期：2026-09-16  
状态：PASS

## Git 范围

实现提交：

`8356d3fab2513828f4730ae146ec0a332623f547`

相对 reopened taskbook point 仅修改：

- `chapter2_region_detection/formal_experiments/probes/probe_cc4_multi_agent_integration.py`

未修改 production adapter、resolver、formal state、replay、reward、WM、CEM、UG uncertainty 或 PPO。

## 源码审核

异步 extension 已满足 A3.4 / A3.6 冻结要求：

- mixed queues：
  - blue_agent_0: Sleep -> Restore；
  - blue_agent_1: Analyse -> Remove；
  - blue_agent_2: Remove -> Analyse；
  - blue_agent_3: Restore -> Sleep；
  - blue_agent_4: Analyse -> Restore；
- scheduler readiness 由 scheduler-local `active + ready_at` 决定；
- `ready_at` 使用 underlying executed action 的真实 `.duration`；
- controller `actions_in_progress` / `action` 不参与 scheduler decision，仅在 launch decision 之后或 step 之后作 oracle assertion；
- busy agent 不提交 filler action；
- 每次 ready launch 都重新读取当前 labels/mask/actions，并重新走 production adapter；
- targeted action 若无合法 target，允许真实 adapter fallback Sleep，且 readiness 自动按 executed Sleep duration=1 计算；
- 保留原 synchronous 4-family integration probe，不回退已有覆盖；
- seed 42/43/44 × pad_spaces false/true 全覆盖。

## 真实运行结果

6 个 synchronous runs：

- adapter resolutions = 120；
- joint action cases = 24；
- sync_all_pass = true。

6 个 async runs：

- 每 run 10 launches；
- 每 run 10 completions；
- 共 60 launches / 60 completions；
- 每 run final scheduler time = 7；
- launch schedule：
  - t=0: blue_agent_0..4；
  - t=1: blue_agent_0；
  - t=2: blue_agent_1, blue_agent_4；
  - t=3: blue_agent_2；
  - t=5: blue_agent_3；
- async_all_pass = true；
- all_pass = true。

## 结论

A3.6 synchronous + asynchronous multi-agent integration 均通过。

A3 Official CybORG / CC4 Four-Action Adapter 正式关闭。

FINAL STATUS: PASS
