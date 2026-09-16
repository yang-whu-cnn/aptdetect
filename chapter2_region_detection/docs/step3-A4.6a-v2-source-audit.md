# Step A4.6a v2 — Model-Space Action Consistency Source Audit

日期：2026-09-16  
状态：**RUNTIME PASS / SOURCE AUDIT BLOCKED**

---

## 1. 背景

A4.6a v2 runtime 已通过，详见 `docs/step3-A4.6a-v2-runtime.md`。

运行结果已经满足冻结 hard Gate：

- train requested→executed mapping accuracy = 1.0；
- validation requested→executed mapping accuracy = 1.0；
- feature=1 but fallback = 0；
- feature=0 but non-fallback = 0；
- H=4 state RMSE 优于 persistence；
- H=4 value RMSE 优于 constant baseline；
- aggregate value Spearman > 0.3；
- 8/8 validation episodes value Spearman 为正。

之后 commit `843e2ecdf979c31f9c34205da2b83c293805dba5` 新增：

```text
formal_experiments/evaluation/audit_model_space_action_consistency.py
tests/test_gate_a_model_space_action_consistency.py
```

因此“源码尚未 push”的旧状态已经过期。

## 2. Source contract 审核

当前 evaluator 已实现：

- `FORMAL_STATE_DIM` shape guard；
- `N_ACTIONS` range guard；
- 通过 `FORMAL_STATE_FEATURE_NAMES.index("any_valid_observable_target")` 获取 availability feature；
- `no_op -> Sleep`；
- targeted + unavailable -> Sleep；
- targeted + available -> keep requested；
- scalar / tensor canonicalization；
- real replay requested→executed exact mapping audit；
- fallback reason 检查；
- H=4 requested-plan integrated rollout；
- fixed world-model member through horizon；
- response reward predictor；
- duration-aware discount；
- H4 state/value quality Gate；
- per-episode Spearman；
- uncertainty-error correlation diagnostic。

配套单元测试文件包含 7 个 `test_*`，覆盖 canonicalization、threshold、scalar/tensor consistency、invalid action 与 quality gate。

当前 GitHub commit 没有 CI workflow/status，因此本记录不把“测试文件存在”等同于“GitHub CI 已执行通过”。

## 3. 阻塞问题：默认 artifact 路径仍指向旧 A4.5 产物

当前 source 中 CLI 默认值仍为：

```text
outputs/formal_replay/{train,validation}.jsonl
outputs/world_model/a4_5b/world_model_absolute.pt
outputs/world_model/a4_5c/response_reward_predictor.pt
outputs/world_model/a4_6a/action_consistency_report.json
```

但 corrective v2 已明确要求：

```text
outputs/formal_replay_v2/{train,validation}.jsonl
outputs/world_model_v2/a4_5b/...
outputs/world_model_v2/a4_5c/...
```

A4.5b-v2 文档已明确 selected world model 位于 `outputs/world_model_v2/a4_5b/world_model_absolute.pt`。

因此当前 evaluator 的算法逻辑与 v2 runtime contract 基本一致，但**默认执行入口存在 legacy artifact drift**。

如果直接无参数运行脚本，有可能重新读取已被 supersede 的旧 replay / checkpoint，从而破坏 A4.6a“必须使用 v2 artifacts”的冻结要求。

## 4. 必须修复后才能关闭 A4.6a

至少完成以下一项正式方案，并写入测试/文档：

### 方案 A：v2 defaults

把 A4.6a evaluator 默认路径统一改为 v2 artifact 目录，并新增测试或 source assertion，保证默认路径不再指向 legacy A4.5 产物。

### 方案 B：禁止隐式 defaults

要求 formal Gate 运行必须显式传 train replay、validation replay、selected world model、response reward predictor、output path；缺失正式 artifact 参数时直接拒绝运行。

无论采用哪一种，都必须确保 A4.6a formal command 不可能静默读取 superseded artifacts。

## 5. 当前审核结论

```text
A4.6a v2

Runtime hard Gate       : PASS
Evaluator source pushed : PASS
Core canonical logic    : PASS
Unit-test source present: PASS
GitHub CI execution     : NOT AVAILABLE
Formal artifact binding : BLOCKED (legacy default-path drift)

FINAL STATUS: NOT CLOSED
```

A4.6a 的下一动作不是重新调 WM 或 reward predictor，而是修正 formal artifact binding，然后再次执行同一 v2 command / tests，确认报告数值与已记录 runtime PASS 一致。

完成后才可进入 A4.6b formal v2.1 config freeze + legacy isolation。