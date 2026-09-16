# Step A4.6a v2 — Model-Space Action Consistency Source Audit

日期：2026-09-16  
状态：**RUNTIME PASS / SOURCE FIXED / LOCAL RERUN PENDING**

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

## 3. 已修复：formal artifact binding

已完成源码修复：

- commit `444ad85e4584f455c087c146c2e90571078c8af3`：把 A4.6a evaluator 默认输入/输出绑定到 v2 artifact；
- commit `5dc197d00838d53d8b25079c5782fd4e3186bc40`：新增 formal v2 default-path regression tests。

当前正式默认路径为：

```text
outputs/formal_replay_v2/train.jsonl
outputs/formal_replay_v2/validation.jsonl
outputs/world_model_v2/a4_5b/world_model_absolute.pt
outputs/world_model_v2/a4_5c/response_reward_predictor.pt
outputs/world_model_v2/a4_6a/action_consistency_report.json
```

源码中已不存在 `outputs/formal_replay/` 或 `outputs/world_model/` legacy 默认路径字面量。

默认路径已提取为模块常量：

- `DEFAULT_TRAIN_REPLAY`；
- `DEFAULT_VALIDATION_REPLAY`；
- `DEFAULT_WORLD_MODEL`；
- `DEFAULT_REWARD_MODEL`；
- `DEFAULT_REPORT_OUT`。

新增两个单元测试：

- `test_formal_defaults_use_v2_artifacts`；
- `test_formal_defaults_do_not_bind_legacy_artifacts`。

该测试文件现共有 9 个 `test_*`。

## 4. 当前仍需完成：本地 rerun

GitHub 仓库未提交 formal v2 replay / checkpoint 大产物，因此 connector 端无法执行真实 A4.6a numerical rerun。

必须在保存 v2 artifacts 的本地实验环境执行：

```bash
cd chapter2_region_detection
python -m unittest tests.test_gate_a_model_space_action_consistency -v
python -m formal_experiments.evaluation.audit_model_space_action_consistency --device cpu
```

无参数 evaluator 现在应直接读取 v2 artifacts。

rerun 后必须确认：

- 9 个 unit tests 全部 PASS；
- train mapping accuracy = 1.0；
- validation mapping accuracy = 1.0；
- feature=1 but fallback = 0；
- feature=0 but nonfallback = 0；
- H4 state RMSE 仍优于 persistence；
- H4 value RMSE 仍优于 constant baseline；
- H4 value Spearman > 0.3；
- 至少 5/8 validation episode Spearman > 0；
- 结果与已记录 v2 runtime 数值无异常漂移。
## 5. 当前审核结论

```text
A4.6a v2

Runtime hard Gate       : PASS
Evaluator source pushed : PASS
Core canonical logic    : PASS
Unit-test source present: PASS
GitHub CI execution     : NOT AVAILABLE
Formal artifact binding : FIXED
Local unit-test rerun   : PENDING
Local numerical rerun   : PENDING

FINAL STATUS: SOURCE FIXED / RERUN PENDING
```

A4.6a 当前不需要重新调 WM 或 reward predictor。下一动作仅是在本地 v2 artifacts 上运行上述 9 个 tests 与 evaluator；结果确认后即可把 A4.6a 正式 CLOSED，并进入 A4.6b formal v2.1 config freeze + legacy isolation。