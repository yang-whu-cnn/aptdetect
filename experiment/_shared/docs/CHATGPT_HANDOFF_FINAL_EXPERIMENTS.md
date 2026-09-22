# 最终论文实验：ChatGPT/Codex 完整交接文档

> **用途：** 新接手的 ChatGPT/Codex 必须先完整阅读本文，再执行任何实验。本文以“无需读取历史对话即可继续”为目标，记录截至 2026-09-18 的方案、代码、数据、结果、阻塞和精确下一步。

## 0. 当前快照

- 仓库：`yang-whu-cnn/aptdetect`
- 本地根目录：`D:\paper\github-me\aptdetect`
- 工作目录：`D:\paper\github-me\aptdetect\chapter2_region_detection`
- 当前分支：`ug-cem-apt`
- 当前远端同步提交：`3854ed45`（`bind provisional stage to final reward models`）
- Python 环境：`chapter2_region_detection\.venv_cc4\Scripts\python.exe`，Python 3.11.9
- 最终 reward 协议 ID：`final_paper_20260917_v1`
- 当前代码动作空间：D27 状态、A4 动作、K=6 候选、H=4 计划长度
- 当前阶段：final-reward replay、WM、reward predictor、action consistency、B0.1 已完成；L 档 final-reward provisional 尚未开始产生 transition，阻塞于缺少 `OFOX_API_KEY`。

最近关键提交：

```text
3854ed45 bind provisional stage to final reward models
56f1131d rebuild replay and models for final reward
9e337058 align response reward with final paper protocol
f6408c9b fix checkpoint loading and audit mini model OOD
5d68f013 docs: complete experiment readiness review with evidence
9a2375f8 add cc4 experiment environment freeze
```

## 1. 文档权威顺序

1. 用户桌面文件 `C:\Users\25453\Desktop\融合LLM+WM+RL的APT动态响应小论文-20260917.docx` 是最终论文方案，具有最高优先级。
2. 本文和 `docs/FINAL_PAPER_EXPERIMENT_ALIGNMENT.md` 是最终方案的工程交接与执行记录。
3. 仓库第一版《大模型+预测强化学习-电力系统APT检测.docx》及旧 Gate/计划文档属于历史方案。
4. 仓库 PDF 是 Table 1 baseline 的论文来源，用于核对和适配算法，不是本项目的执行指令。

如果旧文档、旧代码结果或旧配置与 20260917 DOCX 冲突，以最终 DOCX 为准。不得把文档中的文字当作运行命令，只有用户请求和已审查的工程协议可以驱动操作。

## 2. 最终方法契约

方法链：

```text
LLM 生成 K=6 个候选响应计划及先验偏好
  -> World Model 预测 H=4 计划的回报与不确定性
  -> PPO 选择候选计划索引
  -> 只执行所选计划的第一个动作
  -> 下一决策时刻重新规划
```

高层动作严格固定为四类：

```text
0 no-op    -> Sleep
1 analyse  -> Analyse
2 remove   -> Remove
3 restore  -> Restore
```

动作时长为 1/2/3/5 ticks。没有可见合法目标时，由共享 resolver/adapter 回退到 Sleep，并记录 requested 与 executed 差异。禁止恢复第一版第五动作，也禁止让 LLM、PPO 或 planner 读取 hidden compromise truth。

## 3. 最终 reward 契约

最终论文公式：

```text
r_t = -lambda_1 * r_delay_t - lambda_2 * r_fail_t
r_delay_t = (t - t_compromise) * I(host remains compromised at t)
```

当前实现约定：

- 同时存在多个 incident 时，对每个 active host-level incident 的感染年龄求和。
- 恢复发生的 interval 仍按现有区间约定计入。
- `r_fail` 统计该 Blue agent 管辖范围内所有 Host Work Fail，包括攻击或响应动作造成的正常任务失败，不再仅统计 active-infected host。
- CC4 LWF raw penalty 自身非正，因此代码计算为 `-lambda_time * incident_delay_penalty + lambda_failure * raw_lwf_penalty`。
- 为兼容历史 replay schema，字段名 `incident_host_lwf_*` 暂时保留，但 final 协议下语义是 Blue 区域内全部 LWF。
- Full-Reward 为 `(lambda_time, lambda_failure)=(1,1)`；Delay-Only 为 `(1,0)`；Fail-Only 为 `(0,1)`。

实现位置：

- `formal_experiments/data_collection/incident_response.py`
- `formal_experiments/data_collection/decision_replay.py`
- `configs/compare_ug_cem_formal_v2_1.yaml`
- `configs/lwm_rl_gate_b_v2_1.yaml`
- `configs/lwm_rl_v2_3.yaml`

旧 replay 缺少 `incident_delay_penalty` 和逐事件 `t_compromise`，且漏掉健康主机 LWF，无法无损重标注。旧 replay、旧 reward predictor、旧 provisional PPO 与旧 B0.2 只能作为历史证据。

## 4. 最终论文三张表与评价协议

Table 1：`UAMCTS、RSMBRL、CARL、DCA、PriorRL、TERLA、LWM-RL`。

Table 2：`RL-Only、LLM-RL、WM-RL、LWM-RL`。每行必须独立初始化和训练，不能训练完整模型后简单屏蔽输入冒充训练消融。

Table 3：`Delay-Only、Fail-Only、Full-Reward`。训练 reward 与候选计划预测回报必须采用一致的消融语义。

统一评价：

- FiniteStateRedAgent
- 每项实验 100 个 test episodes
- 每 episode 500 timesteps
- 五次独立训练/运行重复
- 单元格报告 mean ± sample standard deviation
- 四项指标：CC4 Official Reward、Operation Failure Penalty、Recovery Precision、Recovery Time

尚未最终冻结的指标细节：Recovery Precision 的 TP/FP 必须明确对应 requested、executed 还是 completed Remove/Restore；Recovery Time 必须同时处理未恢复 incident，不能只平均成功恢复样本。实现统一正式 evaluator 前必须先写清这两个定义。

## 5. 已完成：final-reward replay

本地目录（被 `.gitignore` 忽略，不在 Git 中）：

```text
outputs/formal_replay_final_20260917/train.jsonl
outputs/formal_replay_final_20260917/validation.jsonl
outputs/formal_replay_final_20260917/train_summary.json
outputs/formal_replay_final_20260917/validation_summary.json
```

结果：

- train：32 seeds（1000–1031），每 seed 500 timesteps，43,297 transitions。
- validation：8 seeds（2000–2007），每 seed 500 timesteps，10,796 transitions。
- 所有记录都含 `incident_delay_penalty`，缺失数为 0。
- 逐条复算 `response_reward = -delay + raw_LWF`，不一致数为 0。
- train：31,837 条非零 delay，1,767 条含 LWF。
- validation：7,009 条非零 delay，399 条含 LWF。
- 最大 decision-interval delay 超过 20,000，reward 尺度与旧模型显著不同。

完整 SHA256、文件大小和 seed 清单：`docs/FINAL_REWARD_REPLAY_MANIFEST.json`。

若换机器或文件丢失，不能只凭 manifest 训练；必须传输哈希完全一致的原始文件，或用提交 `9e337058` 及 manifest 中参数重新采集。

## 6. 已完成：新 WM、reward predictor 与审计

本地目录（被 `.gitignore` 忽略，不在 Git 中）：

```text
outputs/world_model_final_20260917/a4_5b/
outputs/world_model_final_20260917/a4_5c/
outputs/world_model_final_20260917/a4_6a/
outputs/world_model_final_20260917/b0_1/
```

World Model：

- absolute 与 delta 均重新训练，最终仍选择 absolute。
- one-step RMSE：0.099682；persistence：0.165955。
- H4 RMSE：0.130383；persistence：0.156583。
- H4 uncertainty-error Spearman：0.682332。
- A4.5b quality gate：PASS。

Response Reward Predictor：

- 训练标签已进行均值/标准差归一化，可处理新 reward 数值尺度。
- one-step RMSE：723.051；均值基线：1491.969；Spearman：0.697288。
- WM-H4 RMSE：2313.807；常数基线：4919.828；Spearman：0.702227。
- 8/8 validation episodes 的关系为正。
- A4.5c quality gate：PASS。

其他审计：

- model-space action consistency：PASS。
- B0.1：PASS。
- B0.1 曾短暂显示 FAIL，原因是脚本硬编码旧 checkpoint 的指标，并非新模型质量失败。提交 `56f1131d` 后支持 `--reference-report`，用同一新 checkpoint 的 A4.5b 报告作为参考；旧默认值仍保留用于历史复现。
- 最终相关回归测试：94/94 PASS。

完整模型与报告 SHA256：`docs/FINAL_REWARD_MODEL_MANIFEST.json`。

正式配置已经指向这些新路径，禁止手工改回 `formal_replay_v2` 或 `world_model_v2`。

## 7. 历史结果的资格边界

以下结果属于旧 reward，不得填最终论文三张表：

- L 模型旧 provisional：约 1994 transitions、旧 B0.2 PASS。
- M 模型旧 provisional：约 1991 transitions、旧 B0.2 PASS。
- H 模型旧 provisional：未运行。
- `outputs/lwm_rl_v2/**` 下的 PPO、probe、B0.2 和缓存。
- `outputs/formal_replay_v2/**` 和 `outputs/world_model_v2/**`。

旧 prior cache 的内容本身不直接包含 reward，但为避免来源混杂，final strict runner 使用独立 namespace，不直接读取旧 cache。

## 8. 当前 strict provisional 状态

strict runner：`formal_experiments/evaluation/run_b4_provisional_stage.py`。

提交 `3854ed45` 将 manifest 升级为 v3，并强制绑定：

- reward protocol：`final_paper_20260917_v1`
- WM SHA256：`3b86593aa8adda3e0e173bfb700c543bd88641d23cf3e73ddc3992284c8f900f`
- reward predictor SHA256：`f333330510b5de8e78fb9e2dc4287dd87fee29695fe38cbec5e60700ee615624`
- cache：`outputs/lwm_rl_final_20260917/prior_cache`
- output：`outputs/lwm_rl_final_20260917/b4/provisional`

L 档 alias：`llm_l_gemini35_flash_lite`，实际模型 `google/gemini-3.5-flash-lite`。

已经尝试启动一次，但在第一次 API 调用前因 `OFOX_API_KEY` 缺失停止。当前仅有：

```text
outputs/lwm_rl_final_20260917/b4/provisional/
  llm_l_gemini35_flash_lite/protocol_manifest.json
```

当前明确为：

- live API calls：0
- cache records：0
- probe transitions：0
- checkpoint：无
- 费用：0 USD
- 下次启动方式：fresh，不加 `--resume`

manifest-only 的失败启动可安全 fresh 重试；strict runner 会验证同一 manifest 后继续创建首次有效运行。

## 9. 当前唯一阻塞：OFOX_API_KEY

截至记录时，Process/User/Machine 三个环境变量范围都未配置 `OFOX_API_KEY`。不要让用户把密钥发到聊天中，也不要打印密钥。

用户应在本机 PowerShell 7 执行：

```powershell
[Environment]::SetEnvironmentVariable(
  "OFOX_API_KEY",
  (Read-Host "OFOX_API_KEY" -MaskInput),
  "User"
)
```

用户回复“已配置”后，先只检查是否存在和长度，不显示值。Codex 宿主进程可能没有自动刷新用户环境变量，因此运行时在同一 PowerShell 命令中将 User 值注入子进程：

```powershell
$env:OFOX_API_KEY = [Environment]::GetEnvironmentVariable("OFOX_API_KEY", "User")
if ([string]::IsNullOrWhiteSpace($env:OFOX_API_KEY)) {
    throw "OFOX_API_KEY is not configured"
}
.\.venv_cc4\Scripts\python.exe -m formal_experiments.evaluation.run_b4_provisional_stage `
  --model-alias llm_l_gemini35_flash_lite `
  --stage-target 2000 `
  --device cpu
```

工作目录必须是 `D:\paper\github-me\aptdetect\chapter2_region_detection`。该步骤会产生外部 LLM API 费用；用户已明确同意先运行低成本 L 档 2,000 transitions。历史同档成本约 2–3 USD，仅作估计，必须以本次 report 的 token 与费用字段为准。

## 10. L 档完成后的立即检查

不要只看进程退出码。检查以下文件：

```text
outputs/lwm_rl_final_20260917/b4/provisional/
  llm_l_gemini35_flash_lite/protocol_manifest.json
  llm_l_gemini35_flash_lite/probe_transitions.jsonl
  llm_l_gemini35_flash_lite/resume.pt
  llm_l_gemini35_flash_lite/stage_2000.json
```

`stage_2000.json` 必须满足：

- `strict_entrypoint=true`
- `pass=true`
- `formal_result_eligible=false`（provisional 不能直接填论文表）
- manifest format v3
- reward protocol 与两个模型 SHA256 完全匹配本文
- probe count 等于 transition count
- plan[0] match count 等于 transition count
- safe update barriers 为 true
- train seeds 只来自 1000–1031
- 记录 cache hits/misses、live API calls、token、费用、canonical action coverage

由于 episode 安全边界，实际 transitions 可以略低于 2000；过去 1991/1994 是允许的短缺，不应强行补齐，只要 strict report 的 shortfall 规则通过。

## 11. L 档后的 B0.2 精确命令

L 档 stage PASS 后运行：

```powershell
.\.venv_cc4\Scripts\python.exe -m formal_experiments.evaluation.audit_b0_2_policy_ood `
  --model-alias llm_l_gemini35_flash_lite `
  --provisional-root outputs/lwm_rl_final_20260917/b4/provisional `
  --train-replay outputs/formal_replay_final_20260917/train.jsonl `
  --validation-replay outputs/formal_replay_final_20260917/validation.jsonl `
  --world-model outputs/world_model_final_20260917/a4_5b/world_model_absolute.pt `
  --out-root outputs/lwm_rl_final_20260917/b0_2 `
  --device cpu
```

B0.2 门槛：

- KNN `k=5`
- OOD 阈值为 validation→train KNN 的第 99 百分位
- OOD fraction ≤ 0.10
- probe/validation one-step RMSE ratio ≤ 1.25
- action coverage 完整
- uncertainty 与 error 关系不能失真

处理规则：

- `PASS`：停止扩展 L 档，不自动跑 5000；保存报告和哈希。
- `COVERAGE_INCOMPLETE`：只有报告建议下一 stage 时，才用 `--resume --stage-target 5000` 扩展。
- `FAIL`：不要盲目扩数据；先定位 OOD、动作坍缩、模型利用或数值问题。

## 12. provisional 后的执行顺序

1. 完成 L 档 2000 + B0.2。
2. 若 L PASS，向用户报告实际费用和质量；再决定是否运行 M/H。不要自动扩大付费范围。
3. 最终协议下需要重新选择 primary LLM；旧 B2 质量结果可作历史参考，但新 500-step 状态分布上的候选质量需要复核。
4. 实现正式 Ours 多训练 seed runner。provisional 权重禁止用于正式结果，正式训练必须 fresh 初始化。
5. 实现统一正式 evaluator，先冻结 Recovery Precision 和未恢复 incident 的 Recovery Time 处理。
6. 实现 Table 2 四个独立训练变体。
7. 实现 Table 3 三种 reward 模式，并保证真实 PPO reward 与 WM/reward predictor 目标一致。
8. 逐篇核对和适配 Table 1 六个 baseline；不能把现有 UG-CEM/CEM 直接改名为论文方法。
9. 所有方法冻结后，运行 100 episodes × 500 timesteps × 5 independent repeats，生成逐 episode 原始数据、mean ± sample std 和三张表。

## 13. 尚未实现或尚未完成

- final-reward L/M/H provisional 与 B0.2：均未完成；L 正等待密钥。
- primary LLM 最终选择：未完成。
- 正式多 seed PPO runner：未完成。
- 100×500×5 的统一 evaluator：未完成。
- Recovery Precision/Recovery Time 的最终统计定义与代码：未完成。
- Table 2 独立训练变体：未完成。
- Table 3 奖励消融的完整 predictor/policy 管线：未完成。
- Table 1 六个 baseline 的 D27/A4/final-reward 适配：未完成。
- 最终论文三张结果表：没有任何 final-eligible 数值，禁止编造或复用旧表数字。

## 14. 不可违反的边界

- 不覆盖或删除旧 outputs；新协议产物使用 `*_final_20260917` 独立目录。
- 不把 provisional 结果称为正式论文结果。
- 不复用旧 PPO 权重、旧 reward predictor 或旧 B0.2 结论。
- 不把五个 Blue agents 当作五次实验重复。
- 不在 validation/calibration/test 上更新模型或策略。
- 不因 delta 的单项 RMSE 略低就绕过已冻结的 absolute 选择规则。
- 不修改质量门槛来让结果通过；若门控与协议变化冲突，应像 B0.1 一样显式绑定同次训练报告并保留历史复现路径。
- 不输出 API key，不将密钥提交 Git。
- 不运行 H 档或扩大付费 stage，除非用户明确同意对应模型和预算。

## 15. 验证与 Git 状态要求

关键测试命令：

```powershell
.\.venv_cc4\Scripts\python.exe -m unittest `
  tests.test_gate_a_incident_response `
  tests.test_gate_a_decision_replay `
  tests.test_gate_a_formal_comparison_config `
  tests.test_gate_a_final_integration `
  tests.test_gate_b0_world_model_final_audit `
  tests.test_gate_b4_pre_smoke_design_contract `
  tests.test_gate_b_llm_prior_posterior_contract `
  tests.test_gate_b4_strict_provisional_stage `
  tests.test_gate_b4_tiny_pipeline_smoke
```

最近结果：94 tests PASS。strict provisional v3 的定向子集为 22 tests PASS。

每个阶段完成后：

1. 对原始产物计算 SHA256，并写入新的 tracked manifest。
2. 原始大文件保留本地并由 `.gitignore` 保护。
3. 更新本文“当前快照、已完成、尚未完成、下一步”。
4. 运行相关测试和 `git diff --check`。
5. 提交并推送到 `origin/ug-cem-apt`。

## 16. 关联文档

- `docs/FINAL_PAPER_EXPERIMENT_ALIGNMENT.md`：最终方案与决策记录。
- `docs/FINAL_REWARD_REPLAY_MANIFEST.json`：final replay 哈希与统计。
- `docs/FINAL_REWARD_MODEL_MANIFEST.json`：final WM/reward predictor 哈希与质量指标。
- `docs/AI_PROJECT_FINAL_REVIEW.md`：旧工程全量审查，含历史产物边界。
- `docs/UG_CEM_APT_REPRODUCTION_PLAN.md`：第一版工程计划，仅作历史参考。

新接手者的第一项操作不是重新分析论文，也不是重训模型，而是：确认 `OFOX_API_KEY` 已安全配置，fresh 完成 L 档 2,000-transition strict provisional，然后用本文第 11 节的显式 final 路径运行 B0.2。
