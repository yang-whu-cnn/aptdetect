# Step A4.1b — Formal Observable State Encoder 审核记录

日期：2026-09-16  
状态：PASS

## Git 范围

实现提交：

`b43287d735cede725b549e4d664e2725c3bdb206`

仅新增：

- `chapter2_region_detection/shared/formal_state.py`
- `chapter2_region_detection/tests/test_gate_a_formal_state.py`

## 源码审核

正式实现冻结：

- 27 维 fixed-D formal state；
- reset observation 仅建立 host inventory / IP->hostname baseline；
- reset Processes 不计为 threat evidence；
- post-reset Processes / Connections / Files 才更新 observable evidence；
- unknown host 不自动扩展进入 inventory；
- successful Restore 清除该 host 历史 observable evidence；
- successful Remove 不等价于 host normal，因此不清除历史 evidence；
- formal state 不读取 controller true state、Red sessions、true compromise labels 或 future information；
- host score 只用于共享 target ranking，不是 attack probability、PPO reward 或 world-model value；
- 输出 dtype=float32，固定 shape=(27,)，并检查 finite；
- encoder deterministic。

## 测试

用户本地正式回归：

- `tests.test_gate_a_formal_state`: PASS；
- 全部 `test_gate_a_*.py`: 79 tests，PASS。

最终输出：

`Ran 79 tests ... OK`

## 结论

A4.1a + A4.1b 全部通过。

FINAL STATUS: PASS

下一步：A4.2 Decision-Epoch Replay Schema / Collector。
