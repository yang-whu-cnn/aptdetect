# APT 动态响应项目代码与实验审查报告

审查日期：2026-09-18。仓库：`yang-whu-cnn/aptdetect`，分支：`ug-cem-apt`。
审查基线：`9a2375f873fa3a34705a97a766cb8cffe68546db`；本次通过 `git ls-remote` 确认远端分支与本地起点一致。

## 1. 结论与审查边界

当前项目已经具备 LLM 候选计划、冻结世界模型评估、PPO 候选选择以及真实 CC4 决策回放的工程链路，并保存了两个模型的 provisional 运行结果。**尚不能从现有产物直接生成论文 Table 1、Table 2、Table 3 的正式比较结果。** 主要缺口是多模型 B0.2 尚未闭环、正式训练与统一评价调度未落地、模块消融未接入、reward 消融未贯通，以及论文和仓库实验协议不一致。

本报告完成的是原对话未闭环问题的代码—配置—结果映射审查，不代表所有算法逐行证明正确或论文实验完成。本次没有运行付费 LLM 实验，没有重训模型，没有修改算法和实验协议，也没有把历史通过标记当成本次运行验证。

依据包括：原任务《GitHub读写操作指南》的相关用户要求；其附件《融合LLM+WM+RL的APT动态响应小论文-20260917.docx》和 `AI_PROJECT_CONTEXT`；本分支源码、配置、设计文档、结果 JSON/JSONL；本次离线测试。论文附件读取了正文和表格内容，未进行版式审查。用户说明的仓库外 DCA/EURKA/LLM_priori 代码未取得路径，因而未审查其实现。

文中路径均相对于 `chapter2_region_detection/`，除非另有说明。配套 [审查证据清单](AI_PROJECT_REVIEW_EVIDENCE.json) 记录文件数量、结果状态、来源校验和测试范围。

## 2. 项目演化与目录修正

依据用户明确说明，`chapter2_region_detection` 是齐泽第一版方案的演化仓库，后续叠加 CC4、LLM Prior、World Model、PPO 和比较实验改造。不能把目录内所有结果视为同一版本实验。

- `src/`、`experiments/`：历史训练、部署、指标处理路径；有既有数据和代码价值，但不自动符合当前正式契约。
- `formal_experiments/ours/`：当前候选先验、缓存、46 维候选特征、PPO、运行时和 provisional 协议。
- `formal_experiments/data_collection/`：真实 CC4 replay、异步决策 transition、incident reward 记账。
- `formal_experiments/training/`：bootstrap probabilistic WM、response reward predictor。
- `formal_experiments/evaluation/`：Gate 检查、模型审计、B2、B4 provisional 和 UG-CEM smoke。
- `shared/`：D27 编码、A4 动作、可见目标解析、真实动作适配和共享模型 rollout。
- `baselines/ug_cem_apt/`：categorical CEM、UG uncertainty、normalizer 和 planner。

旧 `AI_PROJECT_CONTEXT_REVIEW.md` 提到的顶层 `ours/`、`evaluation/` 不是当前实际目录；应使用上述 `formal_experiments/` 路径。旧文档“B0.2 已完成”的表述必须限定到已有报告的模型。

## 3. 当前正式架构与代码证据

执行链为：可见 D27 + 公共 agent ID → 模型及 split 隔离的 prior cache → K=6、H=4 候选计划 → 冻结 WM/reward predictor → 每候选 46 维特征 → PPO 选择候选索引 → 仅执行 plan[0] → 共享 resolver/adapter → CC4 真实反馈。

主要入口是 `formal_experiments/ours/lwm_runtime.py:LWMDecisionRuntime.prepare/select`。其 `prepare()` 必须调用 prior 和 evaluator，并没有 `use_llm/use_world_model` 开关。

`posterior_features.py` 将 27 维状态、16 维计划 one-hot、1 维 prior、1 维预测回报、1 维 ensemble 回报总体标准差拼接为 46 维。`ppo_core.py` 实现候选间共享 actor/critic；`ppo_training.py` 实现异步 rollout、按 agent/episode 分组和 duration-aware GAE。训练 reward 来自真实 replay，不能用 WM value 或 prior score 替代。

冻结契约见 `configs/lwm_rl_v2_3.yaml`、`configs/compare_ug_cem_formal_v2_1.yaml`：

- 状态 D27；五个 Blue agents；动作 no_op/analyse/remove/restore，ID 为 0/1/2/3。
- 动作时长为 1/2/3/5 ticks；无可见有效目标时回退为 Sleep，并记录 requested/executed 差异。
- WM ensemble=5，absolute target，共享 reward predictor；gamma_tick=0.99。
- train seeds 1000–1031；validation 2000–2007；calibration 3000–3007；test 4000–4019。
- PPO 配置：学习率 0.0003、rollout target 128、5 epochs、minibatch 64、clip 0.2、GAE lambda 0.95、entropy 0.01、value coefficient 0.5、gradient clip 0.5。

注意：这些是已有源码和配置事实；配置中的 formal_training 段本身不等于正式批量训练已实现。

## 4. 正式实验入口与调度能力

现有入口的职责可以确认如下：

1. `collect_cc4_formal_replay.py`：采集指定 split 的 CC4 replay。
2. `training/bootstrap_world_model.py`、`training/response_reward_predictor.py`：训练共享模型；对应 validation、action-consistency、B0.1 审计脚本。
3. `build_llm_validation_state_bank.py`、`run_b2_multi_model_preflight.py`、`run_b2_prior_quality.py`、`run_b2_repeatability.py`：候选先验质量与稳定性，不能代替端到端策略比较。
4. `run_b4_tiny_pipeline_smoke.py`：小规模 live pipeline；其单元测试使用 fake client，二者不是同一运行。
5. `run_b4_provisional_stage.py`：严格 provisional 入口，支持 model alias、stage target、device、resume；调用 `run_b4_provisional_ppo.py`。
6. `audit_b0_2_policy_ood.py`：单模型或 union 审计；`audit_b0_2_all_models.py`：逐模型 Gate，加 pooled/tail 辅助诊断。pooled 通过不能覆盖单模型失败。
7. `calibrate_ug_normalizer.py`、`run_ug_cem_step7_smoke.py`：UG normalizer 校准及真实环境 smoke。

provisional stage 为 2000→5000→10000→20000 条真实 decision transitions，不是强制全部跑满。预算末尾根据剩余量缩短 episode，出现 1994/1991 而不是恰好 2000 是已有安全调度设计，不能据此单独认定失败。

正式配置要求 fresh initialization、每模型每次最多 100000 transitions、每 10000 checkpoint、至少 3 个 PPO seeds、validation 选 checkpoint，且禁止复用 provisional 权重。源码清单及入口搜索未发现完整的正式多模型多 seed runner、统一四指标 test harness 或 Table 1/2/3 生成器。`experiments/eval_deploy_jsonl.py` 等有历史 mean/std 汇总，但不构成当前正式表格协议的实现。

## 5. 已有结果清点及可用性

### 5.1 当前链路结果

- `outputs/lwm_rl_v2/b2/`：240 个 validation states；H/M/L 与 uniform_non_llm 的 prior-quality 文件及 summary 标记通过。repeatability 为每模型 30 states × 3 次生成，summary 通过。可以作为先验质量补充材料，不能填端到端主表。
- `outputs/lwm_rl_v2/b4/tiny_pipeline_smoke.json`：保存的历史 pass=true；属于工程验证。
- `outputs/lwm_rl_v2/b4/provisional/llm_l_gemini35_flash_lite/`：1994 transitions，8 episodes，13 updates，stage pass=true；formal_result_eligible=false。记录成本约 2.2661 USD，为历史报告估算，不是当前价格或下一次预算。
- `outputs/lwm_rl_v2/b4/provisional/llm_m_gpt54_mini/`：1991 transitions，8 episodes，12 updates，stage pass=true；formal_result_eligible=false。记录成本约 3.5399 USD。
- 两者 probe JSONL 行数与 stage 一致，episode seeds 均为 1000–1007；报告中的 WM、reward predictor 和 registry SHA256 与当前文件相符。
- `outputs/lwm_rl_v2/b0_2/llm_l_gemini35_flash_lite.json`：已有 PASS、coverage_complete=true，1994 条 probe；probe/validation RMSE 比约 1.0466，OOD fraction 约 2.357%。这是历史报告，本次未重新计算其数值。
- `outputs/lwm_rl_v2/b0_2/llm_m_gpt54_mini.json`：2026-09-18 离线审计 PASS、coverage_complete=true，1991 条 probe；probe/validation RMSE 比约 1.0855，OOD fraction 约 4.470%。不需要扩到 5000，也没有重新调用 LLM。
- 未发现 H 模型 B4 provisional 结果或 all_models B0.2 汇总。因此“所有模型 B0.2 完成”仍不成立。
- `outputs/world_model_v2/a4_5b/`：训练/验证 replay 计数 9041/2336，包含 absolute checkpoint；`a4_5c/` 包含 reward predictor；`a4_6a/` 为动作一致性报告；`b0_1/per_action_audit.json` 保存 pass=true。可作模型/契约证据。
- `outputs/ug_cem_v2/step6/`：2301 个 calibration states、500 次 planner calls 的 normalizer 报告通过；`step7/step7_smoke_report.json` 通过。仅证明适配基础和 smoke，不是多 seed 正式 baseline 结果。

### 5.2 其他 outputs 的归属

- `world_model/`：历史 WM 与 reward predictor，不能与 `world_model_v2/` 混用。
- `formal_replay_v2/`：当前模型依赖的数据，**不是可以随意删除的无用调试文件**。它被 `.gitignore` 排除且未跟踪，需要独立存档、hash manifest 或可复现重建流程。
- `formal_replay/`：旧 replay；`formal_replay_smoke/`、`formal_replay_v2_smoke/`、`cc4_probe/`：历史采集和 smoke 证据，保留来源再判断复用范围。
- `prior_cache/`、`prior_cache_repeatability/`：候选缓存，不是独立实验样本，不能当成统计重复。
- `metrics/`、`metrics_logs/`、`official_eval/`、`local_online_deploy/`、`cc4/`、`outputs/` 根目录的 end2end JSONL：历史部署/指标/权重。需要逐次核对 action/reward/state/seed/episode 长度来源后才能考虑重算；当前不能直接进入新主表。
- `local_online_wm/`、`outputs/docs/`：历史运行标记和说明。

配套 JSON 清点所有 outputs 顶层目录的文件数和体积。这里“不可直接填表”是协议资格判断，不代表删除或否定这些历史结果。

## 6. 论文三张表与仓库计划的差异

20260917 论文附件的三表共用四列：CC4 Official Reward、Operation Failure Penalty、Recovery Precision、Recovery Time。正文声明每个实验重复五次并报告 mean/std。

Table 1 行为 UAMCTS、RSMBRL、CARL、DCA、PriorRL、TERLA。Table 2 行为 RL-Only、LLM-RL、WM-RL、LWM-RL。Table 3 行为 Delay-Only、Fail-Only、Full-Reward。

相比之下，`docs/UG_CEM_APT_REPRODUCTION_PLAN.md` 的主比较为 LWM-RL / UG-CEM-APT / CEM-APT，另外规划 Table A–F；其模块消融主要是去 prior preference、去 LLM generator、去 uncertainty、去 PPO。**这两套计划不能仅通过改表格标题合并。** 仓库允许最少 3 PPO seeds，论文五次重复要求更强；建议正式实验采用五个独立训练 seeds，并预先冻结具体 seed 列表及统计单位。

TERLA 与 LWM-RL 的论文命名对应、UAMCTS/RSMBRL 与现有 UG-CEM 的算法对应都没有在本次证据中闭环。不能擅自将 UG-CEM 改名为 UAMCTS/RSMBRL，也不能把 CEM 当作 CARL/DCA。

### 6.1 Table 1 主比较

现有 UG-CEM 适配与 CEM(beta=0) 可作为最先跑通共享 harness 的工程比较对象。论文原表中的方法仍需各自实现/适配、参数验证和正式重跑。仓库内 `baselines/` 未提供 DCA/EURKA/CARL 等完整当前契约适配；用户提到的仓库外代码仍需后续定位。

第一版五动作、旧 reward 的结果只能作为历史参照。新主表必须统一 CC4 环境版本、D27/A4、resolver、episode、paired test seeds 和外部指标。保留论文原行清单，先完成 Ours，再适配 baseline；UG-CEM 上游完整算法审查按原用户要求延后，本次只核实本仓库适配和产物边界。

### 6.2 Table 2 模块消融

当前完整 LWM-RL 的候选生成与 WM 特征构造是必经路径，没有可直接执行四行消融的配置开关。建议在保持 candidate selector 架构和训练预算一致的条件下定义：

- RL-Only：固定、可复现且覆盖动作的非 LLM 候选生成；uniform prior；WM value/uncertainty 置零，且不调用 WM。
- LLM-RL：保留 LLM candidates 与 prior；WM value/uncertainty 置零，且不调用 WM。
- WM-RL：同 RL-Only 的非 LLM 候选生成与 uniform prior，保留 WM evidence。
- LWM-RL：现有完整链路。

以上是待实现的方案，不能称已支持。需要在论文说明 RL-Only 是同架构候选选择 PPO；若论文指 D27→A4 的直接动作 PPO，则应另设 direct-action 对照，不能把两种定义混写。每行独立初始化和训练；不能仅在已训练完整策略上置零然后当作训练消融结果。现有 B2 uniform_non_llm 只证明先验评估控制组存在。

### 6.3 Table 3 reward 消融

`formal_experiments/data_collection/incident_response.py:ResponseRewardConfig` 已实现：

`r = -lambda_time × incident_active_ticks + lambda_failure × incident_host_lwf_raw_penalty`

LWF raw penalty 本身非正。Delay-Only 对应 (1,0)，Fail-Only 对应 (0,1)，Full-Reward 对应 (1,1)。底层允许零权重，不需要重写 reward 公式。

但 `run_b4_provisional_ppo.py` 创建 `IncidentResponseBookkeeper` 时使用默认 reward config；现有 CLI 没有贯通三种模式。需要将权重传到采集、训练、manifest、checkpoint 与汇总，避免不同目标混用。

还必须决定 WM evidence 中 reward predictor 的目标：若保持 full-reward predictor 不变，只能称 PPO 训练 reward 消融；若消融整个目标，应从 replay 的两项标签分别训练预测头或按模式训练 predictor，再组合预期回报。否则 Delay-Only 仍可能从 full-reward WM 特征获得失败项信息。无论采用哪种定义，三行最终都报告同一外部四指标，不能把不同训练 reward 直接相减比较。

## 7. 四项论文指标的实现缺口

1. **CC4 Official Reward**：现有 replay/provisional 保存 official reward；正式汇总必须明确团队 reward 在多 agent 中是否重复，以环境每 tick 团队值定义验证一次，不能直接认定所有 agent 的和就是论文列。
2. **Operation Failure Penalty**：已有 incident-host LWF count/raw penalty。需要冻结符号、是否加权、按 episode 总和还是均值；不要把所有 ASF/RIA/team reward 混进该列。上下文所称“响应导致”的因果含义尚不能仅凭 incident-host 筛选证明。
3. **Recovery Precision**：设计文档列出 Restore precision，但本次未发现覆盖三表的统一正式汇总。需要预先定义分母是 requested restore、executed restore 还是 completed restore，分子是成功动作还是实际清除 incident；零分母报告 NA，并一同报告分子分母。旧指标函数不能未经语义核对直接复用。
4. **Recovery Time**：bookkeeper 的 `mean_completed_attack_eradication_time` 只平均已完成 incident，无完成事件返回 None。未清除事件被忽略会偏向低清除率策略；必须同时报未清除率、完成数和预先约定的删失处理。不能填 0，也不能拿上下文第一版 3.18 当当前实验数值。

建议每 episode 保存 method、model alias、training seed、environment seed、scenario ticks、artifact hashes、四指标原始计数、未完成事件以及统一 full response return。先对每 training seed 的同一批 20 个 test episodes 汇总，再报告五次独立训练的 mean/std；同时保留 paired episode 数据用于置信区间。事先写清 std 的计算约定，不把五个 agents 或缓存命中当成五次重复。

## 8. 本次复现检查与发现

首次审查及随后恢复验证得到以下结论：

- `.venv_cc4/pyvenv.cfg` 指向 Python 3.11.9。首次在受限沙箱中启动失败；2026-09-18 在允许执行用户目录解释器后确认 Python 3.11.9、torch 2.14.0+cpu、numpy 2.4.6 均可用。因此这是执行权限限制，不是 venv 再次丢失或损坏。
- 使用目标环境独立运行 `tests/test_gate_b4_tiny_pipeline_smoke.py`：10 tests 全部通过。
- PyTorch 2.6+ 默认 `torch.load(weights_only=True)`，与本项目包含 NumPy normalizer 状态的 checkpoint 不兼容。`bootstrap_world_model.py` 和 `response_reward_predictor.py` 已对本项目可信 checkpoint 显式使用 `weights_only=False`；两个 checkpoint round-trip 测试通过，随后 M 模型 B0.2 通过。
- 可用替代解释器为 Python 3.13.3，torch 2.14.0+cpu，numpy 2.5.0；它不同于冻结环境。
- 使用替代解释器执行 `python -m unittest discover -s tests -p 'test_*.py'`：484 个测试条目，10 errors，1 skipped。不能报告整套通过。
- 5 个 errors 来自 `test_gate_b0_world_model_final_audit.py` 未传新版 `b0_1_gate()` 必需的 `first_action_h4`；1 个来自 `test_gate_b_llm_prior_posterior_contract.py` 仍要求已被 registry-driven 配置替代的 `llm_prior.provider`。这是源码/测试契约不一致的明确证据，应更新测试而非倒退新设计。
- 替代解释器全量测试中的 2 个 checkpoint round-trip errors 已由上述可信 checkpoint 显式加载修复并在目标环境定向验证；其余错误仍未处理。
- 另外 2 个 errors 分别为 CybORG→Ray 导入缺 filelock，以及 OFOX HTTP client 缺 httpx。
- 独立运行 tiny 单元测试也在 CybORG→Ray→filelock 导入阶段失败；本次没有重现原用户环境中的 tiny 通过。全量测试受加载顺序/其他测试替身影响，不能代替独立入口验证。

`requirements_cc4_freeze.txt` 已跟踪，但它不是完整复现闭环。CybORG 通过 `CYBORG_ROOT` 注入外部源码，默认 `D:\python1\cage-challenge-4`；freeze 未固定其代码版本。本地该目录未能取得 Git commit，需要来源版本或源码归档 hash。还应补充 Python 版本、安装平台、torch 安装渠道和关键 import 验证。当前 freeze 出现 httpx2/httpcore2，而代码 import httpx；应在目标环境核实发行包与实际模块是否一致，不应仅凭包名判断已满足依赖。

日志见 `docs/review_evidence/unittest_20260918.log`、`docs/review_evidence/tiny_20260918.log`。它们用于解释本次审查限制，不作为算法性能数据。

## 9. 下一步执行顺序

1. 已确认目标 Python 3.11 环境可用，修复 checkpoint 加载兼容问题，并通过独立 tiny 和两个 checkpoint round-trip 测试。保留现有环境和实验结果，不执行清理删除。
2. 已使用保存的 1991 条 M 模型 probe 完成离线 B0.2，结果 PASS；没有重新调用 LLM。
3. H 模型从严格 provisional 2000 阶段开始，再单模型 B0.2；只有覆盖不足且协议允许时逐级扩量。该阶段会调用付费 API，执行前应确认预算。L、M 已有 PASS，不应无理由重跑付费阶段。
4. 汇总 all_models Gate；全部模型通过后才把共享 WM 冻结到正式比较。若 FAIL，按既有协议修复，不能靠 pooled 平均掩盖。
5. 先补正式 Ours runner、validation checkpoint selection、五 training seeds 与指标落盘；随后实现 Table 2 和 Table 3 的已声明变体。每个 LLM 必须独立 PPO，不共用一个 checkpoint。
6. 适配 Table 1 baseline；优先利用已有 UG/CEM 检查共享 harness，再处理论文原表方法。外部代码定位后再判断复用量。
7. 统一论文命名、统计重复与 episode 协议。仓库计划规定正式 test 为 500 scenario ticks，而 provisional/共享开发配置常为 100；记录 reset tick 与实际 step 数，不直接复用 100-tick smoke 作为正式测试。
8. validation 完成后锁定所有方法、模型及参数，一次性执行 test 4000–4019；输出原始结果、统计摘要与三表。不要用 test 选 primary LLM、checkpoint 或 reward 权重。

可复用的下一步命令如下；需要先解决本节第 1 步，以下命令本次未执行：

```powershell
python -m formal_experiments.evaluation.audit_b0_2_policy_ood --model-alias llm_m_gpt54_mini --device cpu
python -m formal_experiments.evaluation.run_b4_provisional_stage --model-alias llm_h_gpt56_sol --stage-target 2000 --device cpu
python -m formal_experiments.evaluation.audit_b0_2_policy_ood --model-alias llm_h_gpt56_sol --device cpu
python -m formal_experiments.evaluation.audit_b0_2_all_models --device cpu
```

## 10. 对原对话未确认项的最终答复

- 正式入口与种子：已有 provisional/审计入口和固定 split；完整正式 batch runner 与论文表格导出未找到。
- 四种模块消融：已有可复用组件，尚不能直接配置切换运行。
- 三种 reward：底层系数支持，训练/预测目标/manifest 尚未打通。
- outputs：已清点归属；现有 B4 不能作为正式论文结果。
- B0.2：L、M 均有 PASS；H 缺 provisional 和报告；全模型闭环未完成。
- baseline：UG/CEM 有本仓库适配与 smoke；其他论文方法及用户仓库外代码仍待适配核查。
- 环境：freeze 已提交，目标 Python 3.11 venv 已确认可运行，tiny 与 checkpoint 定向测试通过；替代解释器全量测试仍有旧测试契约和依赖错误。
- 论文表格：原表要求与仓库计划存在明确差异，本文已给出保留原表的实现路线，没有擅自替换方法行或填入未经验证的数据。

后续会话应先读本报告和证据 JSON，再读 `UG_CEM_APT_REPRODUCTION_PLAN.md`。不要把“源码已就绪”“历史 Gate 通过”“所有模型通过”“正式论文结果完成”混为同一进度。
