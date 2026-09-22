# CC4 v3 repeat-1 全量冲刺交接（2026-09-22 04:00 +08:00）

## 1. 当前唯一目标与权威约束

- 当前只完成三张表的 **12 个唯一物理结果的 repeat_1**；不启动 repeat 2--5。
- 每个 repeat 仍必须有完整训练、validation、test seeds `4000..4099`、每 episode 500 ticks、原始事件、可重算指标、manifest、SHA 与 eligibility。单 repeat 统一标记 `PROVISIONAL`，不得声称满足五-repeat 论文终值。
- Table 1 Ours、Table 2 LWM-RL、Table 3 Full-Reward 只复用一个 LWM-RL 物理 run。
- 实验执行统一使用 `gpt-5.6-luna/xhigh`；固定 15 分钟只读核验统一复用 `gpt-5.6-luna/high` 代理 `repeat1_readonly_monitor`。主窗口负责设计和最终验收。
- 禁止删除、覆盖、reset、clean、restore、停止健康 writer、在线 LLM、test 调参、缩短 seeds/ticks 或自动启动 repeat 2--5。

本窗口已完整重读的权威文件及 SHA256：

- `docs/USER_MANDATORY_REQUIREMENTS.md`: `5450bc5d692f4441497737f291121923693d15f6a27f92b5f55d0c24d91026a8`
- `CC4_V3_0917_FINAL_EXPERIMENT_TASKBOOK.md`: `6d862e9475c92a668358f69d9e09e9b32b7d23cd83679dcd4bd0d3847eae00a5`
- `docs/CHATGPT_HANDOFF_FINAL_EXPERIMENTS.md`: `9b8b146fc24c4ceefeb8b925439535bb3fc6b239828274403178cd557af2c75e`
- `docs/FINAL_REWARD_REPLAY_MANIFEST.json`: `b7aae1a9189afad9717be7ebfff97ef2016bb25634ae8e8c83c254987653f8ee`
- `docs/FINAL_REWARD_MODEL_MANIFEST.json`: `0a8a2c5c196dee74eda94cca24e4904cf93badf9008bf6ddcc27e6b854503931`
- `docs/EXPERIMENT_BACKUP_PROTOCOL.md`: `aa094b2be57e69221aacc5cb6da41a409428a62cc57ec169ed98683ed68de5b5`
- 上一交接 `docs/SESSION_HANDOFF_20260921_2352.md`: `8f22acf08adbb665c6d79efd541836971a175c99fefb57941fb20a9f5bacfa73`

## 2. repeat1 完成度

唯一物理结果共 12 个；截至本交接：

- 完整并通过当前 repeat 门：DCA、RSMBRL、RL-Only、PriorRL，**4/12**。
- 正在运行：LWM-RL 正式 test、Fail-Only train，**2/12**。
- 尚未完成：LLM-RL、WM-RL、Delay-Only、TERLA、CARL、UAMCTS，**6/12**。

DCA 既有五 repeats 与 RSMBRL repeat1 保持只读；不得重跑或覆盖。RSMBRL corrected backup manifest SHA 为 `c35fb2eb6b526cd0c0da8786cfe3f9652917326e32145ecc4fcc7c1a25828b36`。

## 3. 本窗口新完成的正式 repeat

### 3.1 Table 2 RL-Only repeat1：PASS / PROVISIONAL

源目录：

`outputs/formal_v3/table2/rl_only_trueppo_v2/repeat_1`

验收：100 episodes，seeds `4000..4099`，每 episode 500 ticks；policy seed 51001；Full-Reward；provider calls/cache misses/test-time updates 均为 0；冻结 artifact before/after 相同；原生 eligibility PASS；独立指标复算误差不超过 `1e-9`；writer lock `RELEASED`。

关键 SHA：

- raw manifest: `7e1ddd80266aeb3461f83b7213226cb49e7873886ef160d3b31921a0fa020806`
- raw eligibility: `62d6a18d3d08d4e20a351d73648152245245bccb3a929873e9aabc651621dd12`
- official aggregate: `8758cbe891d307d7ff70d683524ab44b06047744b0ca3fde33a8c7f6810ecf47`

非破坏性准入桥：

`outputs/formal_v3/admission_derived/table2/rl_only/repeat_1`

- bridge manifest: `a0d9876d3282831ef702c0c8e80fb1ff8db6034e75535c8accb8ec129223fd80`
- admission report: `4301336e25b06a6a2f5de9793c2be97f324e5c6318238504264f9c0ec54d8400`
- 状态：`PROVISIONAL`、`formal_repeat_eligible=true`、`paper_row_eligible=false`

repeat1 四项指标：

- CC4 Official Reward: `-309.63`
- Operation Failure Penalty: `18.18`
- Recovery Precision: `0.8582064935676924`
- Recovery Time（censored mean）: `74.63995499437429`

辅助：completed-only `54.37284586889999`；unrecovered rate `0.22377797224653082`。

### 3.3 已完成结果的不可覆盖 PASS 备份

`formal_experiments/evaluation/backup_outputs.py` 已增加两个严格、只读的 completed-formal PASS adapter；没有向源目录补造 `run_status.json`：

- true-PPO 必须通过 `audit_true_ppo_source`、官方 aggregate PASS、`RELEASED` lock、完整 SHA/冻结性和 100×500 门。
- generic formal 必须通过 `validate_run_directory`、eligibility schema/status/eligibility 三重 PASS、manifest formal、100×500 和逐文件 SHA 绑定。

定向测试 7/7 PASS；既有备份测试 49 PASS / 1 skipped；唯一环境错误是已有 Windows 真实依赖锁测试的 WinError 5。`py_compile` 与 `git diff --check` PASS。

两个真实备份均已依次通过 dry-run、create、verify：

- RL-Only：`C:\aptdetect_experiment_backups\20260921T201436Z-table2-rl-only-trueppo-repeat1-formal-pass.finalized`；84 files / 571,406,142 bytes；backup manifest SHA（sidecar 内容相同）`030637fb29438aedd0e87b997312db6d1637b5a9eb65603a61550a001d513ff7`。
- PriorRL：`C:\aptdetect_experiment_backups\20260921T201436Z-table1-priorrl-ppo-repeat1-formal-pass.finalized`；211 files / 238,646,232 bytes；backup manifest SHA（sidecar 内容相同）`1115851ee87bedf302a20b0d472b27ba863cc1dc98b421245815d2297902ecc8`。

两份备份均记录 `status=PASS`、`formal_result_eligible=true`，源 before/after 与 payload 的文件数、字节数和逐文件 SHA 一致；这不改变单 repeat 的 `PROVISIONAL` 边界。

官方 backup dry-run 因 true-PPO 源没有通用工具要求的 `run_status.json`/safe-stop report 而 exit 2；未改 raw 源、未伪造状态、未手工复制。必须补一个 fail-closed 的 true-PPO PASS 备份兼容路径后再备份。

### 3.2 Table 1 PriorRL-PPO-CC4 repeat1：PASS / PROVISIONAL

训练资产：

`outputs/formal_v3/training/priorrl_ppo_cc4/repeat_01`

训练审核：alpha `0.01`；train seeds `1000..1031`×500；policy 51001；test=false；WM=false；provider/online calls=0。

- alpha selection SHA: `166acee32371d57c71b47506561d35003d317141cdc27f0c9f50253be7066cb9`
- checkpoint SHA: `6cbd471dd27cf87375248b866a6adc4ce3365c584cf1b6b04527770028b8406f`
- training manifest SHA: `4a284cffb203c28ecdbb86925c96f452cb79e91d07b0dcc70cb8329ea1b25552`
- canonical prototype SHA: `e4359df9b6703a767ebe0e52ad09d2b208b8ddd19bd11e3820a887a249c96d09`

为满足 evaluator 的 canonical worktree 路径，训练目录被无覆盖复制到 `D:\w\priorrl\chapter2_region_detection\outputs\formal_v3\training\priorrl_ppo_cc4\repeat_01`；6 files / 1,301,918 bytes，逐文件 SHA 和字节完全一致。原始主仓训练目录保留。

正式结果：

`outputs/formal_v3/table1/priorrl_ppo_cc4/repeat_1`

验收：100 episodes，seeds `4000..4099`，每 episode 500 ticks；generic validator 与 eligibility 均 PASS；代码 commit `e949939411a529cf0e8519f43fe917999b4003e7` clean；artifact SHA 全匹配；200 个 per-seed recovery files 保留。

关键 SHA：

- manifest: `3ff96ecd36cbc595841d44521b90b6c0e6a13222d4cd6b3adc749c126c8d9a1b`
- eligibility: `1bf9e93d425db9925713444a5bb94d2708edab8aeaa4348026c53f71d083539b`
- metrics: `66f0abb4a326ed8ad77708cb77031266906766ba4f96ec86d368b907fef99f5b`

repeat1 四项指标：

- CC4 Official Reward: `-309.63`
- Operation Failure Penalty: `18.18`
- Recovery Precision: `0.28066537863437163`
- Recovery Time（censored mean）: `74.63995499437429`

辅助：completed-only `54.37284586889999`；unrecovered rate `0.22377797224653082`。

## 4. 当前活动 writer（04:00 实时快照）

### 4.1 LWM-RL true-PPO v3 repeat1 正式 test

- target: `outputs/formal_v3/table2/lwm_rl_trueppo_v3/repeat_1`
- worktree: `D:\w\trueppo_integrated`, HEAD `87a52483fbda4c4c9f383d86adc38a260d81cbc2`, clean
- 进程链：venv shim PID `23976` -> runtime PID `34972`
- 阶段：`test`；lock `ACTIVE`
- 身份：Full-Reward、policy 51001、CUDA、provider0、test `4000..4099`×500
- selected checkpoint epoch24 SHA: `416125efa240feea2556d87585d936e34563334d97961fdadc2941ffdfd72ba6`
- validation SHA: `98054670da7d8f6022e521b1da8cf6bde66d347de68645400f0f6ce5fa16dcf9`
- prepare-test auth SHA: `f5f14a3915925ca56dde89e920cc5d7182a49ddc25aefded5a6ff3db53d12fed`
- 03:50 runtime CPU time 约 56 分钟且持续增长；runner 在 test 结束前不写 episode parts，禁止因目录不增长而判死或重启。

完成后必须先审原生 eligibility，运行官方 `--stage aggregate`，再创建 `admission_derived/table2/lwm_rl/repeat_1`；该物理结果同时服务 Table1 Ours、Table2 LWM-RL、Table3 Full-Reward，禁止新建独立 Full-Reward run。

### 4.2 Table 3 Fail-Only true-PPO repeat1 train

- target: `outputs/formal_v3/table3/fail_only_trueppo_20260922_repeat1`
- worktree: `D:\w\t3fail`, HEAD `b094ae9ba13cf20add2c0353c9457b20fccc6395`, clean
- 进程链：outer pwsh PID `35764` -> venv Python PID `37220` -> runtime PID `49692`
- 阶段：`train`；lock `ACTIVE`
- 身份：variant `lwm_rl`、reward `Fail-Only`、policy 51001、train seeds `1000..1031`×500、CUDA、test=false
- 04:00 首个安全 pending checkpoint 已存在，seed 1000 正在执行。

本窗口先完成 calibration：seeds `3000..3007`、4000 samples、500 ticks、provider0、policy_updated=false、Blue-visible-only、lock `RELEASED`。正常化器与当前 LWM v3 相同：

- normalizer SHA: `e50df37c17cf7c49d2c3127800144a948a1a709c5a8396e773d2fb71f8765025`
- target calibration manifest SHA: `8b16d4d4d903135cddd411f39f8824549e3dedef7372db675ffda1c924ac1d98`
- input fitted calibration manifest SHA: `fe0a7eb95b1fc5ff02775b85b05135f3be8555118880d68a1d0dd1db14c356d9`

启动器最初误绑定 RL-Only v2 的旧 normalizer SHA `5db617...`，因此 calibration 后曾 fail-closed，未启动 train。主窗口比较 JSON 后确认统计量、raw input 与 LWM v3 完全一致，只是 v2/v3 calibration manifest 身份不同；随后只修正门值并直接复用已通过 calibration，未重跑 calibration。

最终启动器：

- `scripts/table3_fail_only_trueppo_20260922_repeat1.ps1`: `dd13dcc2128d9fef035aa26b4a412d4a83769da15a5a022bf0440705fe4dc02d`
- `scripts/table3_delay_only_trueppo_20260922_repeat1.ps1`: `e2591ad7b05c26e4c0be4f7d9fcd080688af24f6ff59aa1db585e61b1b527b79`

两个脚本默认 inert，支持 stage-aware dry-run，无 `Remove-Item`、reset、clean、restore。Delay 尚未启动。

## 5. true-PPO 非破坏性准入桥

新增：

- `formal_experiments/evaluation/true_ppo_admission.py`
- `tests/test_true_ppo_admission.py`
- 修改 `validate_formal_run.py`、`generate_paper_figures.py`

设计：不覆盖 runner-owned manifest/eligibility；只在 `outputs/formal_v3/admission_derived/...` 创建 SHA 绑定桥；active writer、缺 aggregate、篡改或覆写目标均 fail closed；raw source 在 discovery 中只标 `source_only`。

验证：新增 4/4 PASS；formal manifest/figure 回归 31/31 PASS；`py_compile` PASS；`git diff --check` PASS。只有官方 aggregate PASS 且 writer 释放后才可建桥。单 repeat 仍为 `PROVISIONAL`。

## 6. 保留的失败与恢复证据

- PriorRL 第一次 train 启动使用主仓 prototype，因“outside canonical frozen artifact”在创建 target 前失败；日志 `outputs/launcher_logs/priorrl_ppo_cc4_repeat1_train_20260922T0300Z` 保留，不计 writer。
- 正确 PriorRL train 日志：`outputs/launcher_logs/priorrl_ppo_cc4_repeat1_train_20260922T0302Z_canonical`。
- PriorRL test 日志：`outputs/launcher_logs/priorrl_ppo_cc4_repeat1_test_20260922T0321`。
- RL-Only test 日志：`outputs/launcher_logs/rl_only_trueppo_v2_repeat1_test_20260922T0255Z`；aggregate 日志：`outputs/launcher_logs/rl_only_trueppo_v2_repeat1_aggregate_20260922T0344`。
- Fail calibration stdout/stderr 位于 `outputs/launcher_logs/table3_fail_only_trueppo_20260922_repeat1_calibration.*.log`；其首次 post-calibration 门失败是启动器 SHA 预期错误，不是算法或数据失败。没有重跑 calibration。
- UAMCTS 旧正式路径在 seed 4004 超出冻结支持半径；禁止调宽、test tuning 或 resume 旧失败 run。必须保留 BLOCKED 证据并审查是否存在合规 fresh 路径。

## 7. Git 与工作树状态

04:00 快照：

- 主仓 HEAD `fa52f8064527971cc9a2c0fe2288927f025ebb20`，dirty（16 条）；含既有用户/前序文档改动、准入桥、新 handoff 与两个新 launcher。禁止 reset/clean/restore，不 commit/push，除非用户明确要求。
- `D:\w\trueppo_integrated` HEAD `87a52483fbda4c4c9f383d86adc38a260d81cbc2`, clean。
- `D:\w\trueppo_rlonly` HEAD `28372df20ebf544771dae5b117e23cf7b57399e8`, clean。
- `D:\w\priorrl` HEAD `e949939411a529cf0e8519f43fe917999b4003e7`, clean。
- `D:\w\t3fail` HEAD `b094ae9ba13cf20add2c0353c9457b20fccc6395`, clean。

`terla_backup_source_20260921_clone/` 是已有未跟踪恢复资产；不得清理。

## 8. 活跃代理与 15 分钟核验

- `table3_next_executor`: `gpt-5.6-luna/xhigh`，已启动 Fail train；完成后回到 idle。
- `terla_repeat1_executor`: `gpt-5.6-luna/xhigh`，正在只读审计并补齐 TERLA train/validation/test 路径；默认不启动长任务。
- `repeat1_readonly_monitor`: `gpt-5.6-luna/high`，固定只读监控；必须复用同一代理，不创建新监控窗口。
- heartbeat `cc4-15` ACTIVE，每 15 分钟唤醒当前主任务并要求把核验交给上述 Luna/high 代理。正常无变化保持安静。

## 9. 下一接管窗口的精确顺序

1. 先重读本交接与 `USER_MANDATORY_REQUIREMENTS.md`；实时复核 PID，不能把本文件 PID 当现状。
2. 用同一个 Luna/high monitor 核验 LWM test 与 Fail train；不得重复 writer。RAM <1.5 GiB、OOM风险、异常退出、身份漂移或 15 分钟无合理 CPU/进度时上报主窗口。
3. LWM test 完成后：主窗口审核 100×500、原生 eligibility、冻结 before/after、指标重算；Luna/xhigh 用完全相同参数运行官方 aggregate；再创建/验证 `admission_derived/table2/lwm_rl/repeat_1`。不要启动独立 Table3 Full-Reward。
4. Fail train 完成后：审核32/32连续 train seeds、curve/journal/checkpoint SHA 与 Fail-Only 双 reward 通道；建立不可覆盖 dependency backup；再单独授权 validation，选定 checkpoint 后 prepare-test 与 test。不得自动越级。
5. PriorRL 与 RL-Only 已完成 PASS 备份；后续只做聚合发现/最终三表消费测试，不重跑、不重复备份。
6. 每个后续完成 repeat 使用同一严格 backup adapter 执行 dry-run/create/verify；不得伪造 `run_status.json`。
7. TERLA 代理返回后由主窗口审 diff/测试；只有原论文 reward、hidden-truth 隔离和 stage path 全部 PASS 才启动。
8. 资源允许时下一 GPU 序列建议为 Fail validation/test，然后 Delay calibration/train；LLM-RL 与 WM-RL 使用各自独立 policy，不能复用 LWM checkpoint。
9. CARL 需高 RAM，等待活动 writer 释放后启动；UAMCTS 保持 BLOCKED，除非出现不改变冻结半径/协议的 fresh 合规路径。
10. 每个完成 repeat 都要主窗口复算、备份、更新此交接；只在 12/12 唯一物理 repeat1 都有四项最终指标后生成第一版全量三表和图，并清楚标 `PROVISIONAL`。

## 10. 05:04 增量状态（覆盖第 4、8、9 节中的旧实时快照）

当前完成度仍为 `4/12`；活动正式 writer 已扩展为三条，均由动态 PID 核验，历史 PID 只作线索：

- LWM-RL test：runtime PID `34972`，`ACTIVE/test`，05:00 CPU time `2:01:08`，唯一 writer；test 仍在一次性计算，目录仍为 77 files / 109,540,168 bytes，不能因尚未写最终 episode parts 而重启。
- Fail-Only train：runtime PID `49692`，`ACTIVE/train`；05:00 已 durable 完成 seeds `1000..1014`（15/32），pending `1015`，唯一 writer。
- LLM-RL train：05:03 使用主仓只读 launcher 指向 clean worktree `D:\w\t2llmready` 成功启动；链 `8680 -> 16012 -> 14760`，writer PID `14760`，`ACTIVE/train`；首轮核验已完成 seeds `1000,1001` 并增长。不得终止承载 launcher 的工具会话 `65240`。

LLM-RL 本次**没有重跑 calibration**。此前目标已只有通过的 calibration 三件套与 `RELEASED/calibration` lock；normalizer SHA 为 `e50df37c17cf7c49d2c3127800144a948a1a709c5a8396e773d2fb71f8765025`。训练身份为 `llm_rl`、Full-Reward、policy 51001、train `1000..1031`×500、CUDA、provider/test updates 0，且 runner 参数不含 world-model/reward predictor：

- launcher：`scripts/table2_llm_rl_trueppo_20260922_repeat1.ps1`
- launcher SHA（加入 dead ACTIVE/train fail-closed 回收门后）：`b9d062bc52ec6b0985bfd439c163f9c657476a233b66becb527d63c4c9f50412`
- clean worktree HEAD：`669859819cda21119f2b1fb5906613b917c3d6b8`
- launcher log root：`outputs/launcher_logs/table2_llm_rl_trueppo_20260922_repeat1`
- train record stem：`train_20260921T210339216Z`

05:03 启动后资源约为：free RAM 3.38 GiB、CPU 26.4%、GPU 92%、VRAM used 1.64/8.19 GiB；这是健康 writer 启动后的负载，不应按事后瞬时门终止。全局报警门仍为 free RAM `<1.5 GiB`、OOM 风险、重复 writer、身份漂移或 15 分钟无合理活动。

监控要求已同步至 heartbeat `cc4-15`：继续每 15 分钟复用同一个 `repeat1_readonly_monitor`（Luna/high），现在同时核验 LWM、Fail、LLM 三条链。正常时保持安静；完成或异常时唤醒主窗口。`terla_integration_executor` 与 `carl_repeat1_executor` 均为 Luna/xhigh，只准备/审计启动链，尚未授权长实验。

后续优先级：

1. 任一活动阶段完成，先由主窗口验收；禁止自动跨阶段。
2. LWM 完成时最高优先执行同参 aggregate、admission bridge、正式备份；此物理 run 同时计 Ours/LWM/Full-Reward。
3. Fail/LLM train 各自完成时先核对 32/32、journal/curve/checkpoint、冻结资产和 provider/test=0，备份训练依赖；再逐一授权 validation。
4. 资源释放前不再增加 GPU writer；WM-RL、Delay-Only、TERLA、CARL、UAMCTS 继续等待各自审计与资源门。

### 10.1 LLM-RL 05:17 中断与原地恢复门

LLM writer PID `14760` 在 05:16:49 后意外消失，launcher 没有 exitcode，stdout 为空、stderr 仅有 Gym/lz4 warning；保留 stale `ACTIVE/train` lock，没有手工清锁或覆盖。中断点已由主窗口与 Luna/xhigh 双重只读验收：

- durable completed seeds 为连续 `1000..1021`（22/32）；training curve、train summary、journal index 都为 22，22 个 journal 文件的 SHA/size 均匹配。
- pending 为 index `23` / seed `1022` / phase `pending`；文件 `pending/seed_0023.pt` 的记录 SHA 与实算 SHA 同为 `ca77accf1497ed2777b1115e2016cad49b0164df470b29a1395081c1240fc8c7`。
- `provider_calls=0`、`test_started=false`、`on_policy_ppo_used=true`，worktree HEAD/runner/normalizer 无漂移。
- runner 原生语义只拒绝仍存活的 ACTIVE writer；对 dead lock 会先标 RECLAIMED，再严格验证 resume/pending/journal 后从 seed 1022 重算，不覆盖已提交 22-seed 前缀。

launcher 仅增加一条同语义门：dead `ACTIVE/train` 必须 schema、stage、variant、reward、policy seed 全匹配且 PID 确实不存在；live ACTIVE 或其他状态仍 BLOCKED。PowerShell parse PASS。05:24 单次 dry-run 因 GPU util 恰为 `90%`（门要求 `<90%`）而保持 WAITING；没有重试风暴、没有启动恢复。下一次 15 分钟监控确认资源门转绿后，用**独立隐藏 outer PowerShell**承载同步 launcher，避免代理工具会话结束再次杀死健康子进程。

05:31 监控资源门转绿（GPU 24%、free VRAM 6496 MiB、free RAM 4.80 GiB、CPU 20.5%），随后一次性恢复成功：独立隐藏链 `outer pwsh 50416 -> venv shim 41180 -> runtime 16364`，新 lock 为 `ACTIVE/train`/PID `16364`。05:34 已从 pending seed 1022 正确继续到 25/32（last seed 1024）；旧 22-seed curve 前缀及 journal checkpoint22 的 SHA 均保持不变，provider0、test false、on-policy true。外层日志 stem 为 `outputs/launcher_logs/table2_llm_rl_trueppo_20260922_repeat1/recovery_outer_20260921T213224125Z`，runner stem 为 `train_20260921T213226732Z`。该链已独立于代理工具会话，禁止终止。

## 11. 06:30 增量状态（最新；覆盖第 10 节实时进度）

### 11.1 当前三个唯一 writer

- LWM-RL：runtime PID `34972`，`ACTIVE/test`，06:30 CPU time `3:24:46`，WS约0.81 GiB；100×500 test 仍在一次性计算，未完成。
- LLM-RL：runtime PID `47968`，`ACTIVE/test`，独立链 `outer 9768 -> shim 19544 -> runtime 47968`；train/validation 已完成，100×500 test 正在运行，无最终 test artifacts/aggregate。
- Fail-Only：runtime PID `31940`，`ACTIVE/validation`；train 已32/32完成，validation `2000..2007`×500 正在运行；禁止自动 prepare-test/test。

06:30 资源为 CPU 27.4%、free RAM 3.45 GiB、GPU 92%、free VRAM 6449 MiB。并发已经达到当前安全上限；CARL test 需 RAM>=4 GiB、TERLA test 需 RAM>=4.5 GiB，均保持 WAIT。不得降低门值。

### 11.2 LLM-RL 已验收链与当前 test

Train PASS：连续 `1000..1031`（32/32）、16000 steps、curve/journal均32、public checkpoints恰为8/16/24/32、provider0/on-policy、test false。完整75文件/110,139,228 bytes SHA inventory：

- `outputs/launcher_logs/table2_llm_rl_trueppo_20260922_repeat1/train_complete_20260921T214659Z.sha256_inventory.json`
- inventory SHA `97f1a1c7...`，records digest `df931a4a...`（需要最终报告前补记完整值）。

官方 dependency backup 因 completed-train 尚无通用 `run_status` 且主仓 dirty 而 fail-closed；没有伪造/手工覆盖。完整 repeat 后再使用正式 adapter 备份。

Validation PASS：records SHA `17a4259fa75a843620d315050425950a2ac91f62d02c17dcb012c60909f5a901`；4候选仅 epoch8/16/24/32，各自只用 `2000..2007`×500，无 test。冻结排序 `max_score_then_min_epoch_then_min_checkpoint_sha256` 选择 epoch24（score `-3422729.625`），checkpoint SHA `52a25465c059b7df4de9a13ff828d5f20afc1872b41e9c53a82dd932a3998c51`。

主窗口审核后使用 runner 专用 `approve_prepare_test_authorization` 对既有未批准文件做原子状态转换；选择 SHA 保持 `852c8b3809deba516674fd1b8a35e305502ff983facd6884fc1d3a348cf66f0d`，批准后 authorization SHA `3c59ea608b98561427da9a006cc02567257aacf05ec930b6fc51a7944c4c8d5d`，self manifest `c52513852470766e3ac7ed33f58c2a380d561a45cd90558393c7e18f3c69af58`。test 命令无 WM/reward 参数，provider0。日志 stem：`outputs/launcher_logs/table2_llm_rl_trueppo_20260922_repeat1/test_outer_20260921T222506550Z`。

### 11.3 Fail-Only train 完成

Train PASS：32/32、16000 steps、curve/journal32、public checkpoints8/16/24/32、provider0/on-policy、test false、Fail-Only 双 reward 通道一致。关键 SHA：curve `0c777cde817730834f199007b455a91f95d11e8504afd6ece6201136ef279bc2`；checkpoint32 `e6990d7543a72b4e9ec1acadb23e9212de41bc3ca7477c94aab180a3700eb6a2`；完整75文件/110,116,492 bytes inventory：

- `outputs/launcher_logs/table3_fail_only_trueppo_20260922_repeat1_train_inventory_20260921T221125Z.json`
- inventory SHA `6e9844cea3cb38a0b2740a3ee1cd6fe3fb5057132c2a9010894bcc67d5c53509`

正式 dependency backup 同样因主仓 tracked dirty/缺通用安全状态而未创建，未伪造。当前只运行 validation。

### 11.4 TERLA-A4

专用分支 `C:\Users\25453\.codex\worktrees\a553\aptdetect` HEAD `250acdca0287f233a218eafb4473f5055f06ac0b` clean。已修复 TERLA resolver 只用 graph canonical `2*p+n`、validation/test 真 dry-run 与 writer lease；相关61 tests PASS。validation 已完成 `2000..2007`×500、lease `RELEASED`、policy hash始终 `2d924cb48dbde59aa1dd56a597a32e99a1a78d87fe76570caed06b0b5a7cd20e`、provider0、source SHA不变。target：`outputs/formal_v3/table1/terla_a4/repeat_1_eval_v1`；episodes SHA `503079f087a9af4b345b1185193b6432f80ec6e475708110b9ea156685da9688`；decisions SHA `32f43172e93e395f4f3a6d4fe425616b613b3f2c16759c649141a9cc67c7f1a4`；state SHA `de36e0442f1fa1fe95931bc8a0e5ab2f61ae4322949cc8707e1e742b9e549d92`。

launcher PowerShell 等待/重定向换行 bug 已修，当前主仓 launcher SHA `c6804dcf1346c7f608f2104125383487954bca366a4e29ba9e9d63004d68b5fb`。test 尚未授权；在启动前主窗口必须处理/验收“validation 与 test 不得混入同一 root episodes.jsonl”输出布局风险，并确认 source checkpoint/optimizer/manifest immutable。

### 11.5 CARL-CC4 adapted

专用 `D:\w\carl` HEAD `6459a142640a3e1baaa746e38a5419688b26b1a7` clean。signed CAICS reward 已修（incident clearance -1 -> +2；terminal clearance 1002）；按任务书明确披露 requested H256/effective H4、`ADAPTED_TRUNCATED`、frozen shared WM only validated to H4；8:1 synthetic、hidden-truth=false、test unused。Train/validation selection 已完成：32 train seeds、8 validation seeds、500 ticks、policy51001；checkpoint SHA `a018b472e39fe21dcc2883c84a84cf0c6ff0b61c092d72faeeb7f318d8abd3bc`；selection SHA `38ab430090a9de99cadc23eeacf094d551448e9842a0c19661656c6f6ae2ee21`；training manifest SHA `4317e4eda27a820501fe39afecdc4fe236159f876e4ffb02f32da8ff89fa43de`。repeat1为PROVISIONAL、paper_row_eligible=false。

首次 test 在写 episode 前因 selection CRLF/raw SHA 与 evaluator LF重序列化 SHA 不一致而安全退出；旧空 target/log保留。已做严格 CARL-only 修复：直接绑定 selection 文件原始 bytes SHA，仍保留schema/content/seeds/WM/git检查；`run_method_episode.py` SHA `60ca319e1c16b548bd2433041227b81531de91e04b9c32715602d8347816cdb4`。fresh retry target `repeat_1_retry1` 与日志 `test_retry1` 已绑定，但06:30 RAM不足，尚无 retry writer。

### 11.6 UAMCTS

旧 canonical fresh lineage 在 seeds4004/4009 超过 frozen radius；fallback equivalence为BLOCKED且改变语义。禁止扩大旧radius、fallback、resume或test tuning。新增的 bounded support-contract 草案只使用train/validation/calibration、拒绝test/hidden truth/provider，且不改planner/potential/radius；但它不能证明旧prototype覆盖失败test状态，因此 UAMCTS 仍 `BLOCKED`，不得启动误导性正式run。草案位于 `D:\w\uamctsfreshsupport`，4文件未提交；不纳入正式结果。

### 11.7 下一步（产出即衔接）

1. 复用 Luna/high 每15分钟核验 LWM、LLM test、Fail validation；任一完成立刻主窗口验收。
2. Fail validation完成：审8/8与选择，生成未批准 prepare-test；主窗口审核后再test。
3. LLM test完成：审100×500、eligibility/freeze/provider0，随后同参aggregate、admission bridge、正式不可覆盖备份。
4. RAM>=4 GiB时优先启动CARL fresh retry test；RAM>=4.5 GiB且TERLA输出布局修复/验收后启动TERLA test。
5. 之后启动 WM-RL 与 Delay-Only；两者不得复用其他变体policy。UAMCTS保持BLOCKED直到出现不依赖test的科学合规支持域证明。

### 11.8 WM-RL 与 Delay-Only 已就绪但等待资源

WM-RL：clean worktree `D:\w\t2wmready` HEAD `3009d7d9a7fa215f29d817b7a47a3b64666ec18c`；launcher SHA `b9e02e402c016b0878e41e1ab009e774d48d3aae051815d9fab8a2d7553e8c04`；runner SHA `ec2de89ebb2e0894df4cf7d4194afdb502503e9f94bf121915cf114ed6282a6e`。preflight 与35+7项定向回归 PASS；身份为独立 `wm_rl`/Full-Reward/policy51001、uses_world_model=true、uses_llm_prior=false、provider0、K6/H4 non-LLM plans。target/log 均 absent。当前只因 RAM<4GiB/GPU瞬时>=90而 `READY_WAIT_RESOURCE`；下一步最多先 calibration，不得自动train。

Delay-Only：clean worktree `D:\w\t3delay` HEAD前缀 `a3c09bcd`；launcher SHA `e2591ad7b05c26e4c0be4f7d9fcd080688af24f6ff59aa1db585e61b1b527b79`；同一 runner SHA `ec2de89e...6282a6e`。true-PPO formal preflight PASS；身份为独立 `lwm_rl + Delay-Only`/policy51001/provider0，frozen Delay predictor SHA前缀 `192efbe6512f...0e683f`、manifest前缀 `fb9fdd64a6f5...31eaf`。32/33测试通过；唯一失败是无关旧 `run_table23_formal` 缺 legacy sidecar，专用 true-PPO 路径 PASS，未复制不匹配 sidecar。target/log absent。资源门恢复后也只先 calibration。

## 12. 07:35 增量状态（当前最新；覆盖第 11 节实时进度）

### 12.1 完成度与三表口径

按 12 个唯一物理结果计：`5/12` 已产生正式 repeat1 测试结果，`3/12` 正在运行，`3/12` 已就绪/待后续阶段，`1/12`（UAMCTS）仍为真实 BLOCKED。LWM-RL 是唯一物理结果，并只读服务 Table1 `lwm_rl`、Table2 `lwm_rl`、Table3 `full_reward` 三个逻辑行。

- Table1 当前已出结果：DCA、RSMBRL、PriorRL；CARL/LWM 正在 test；TERLA validation 已完成但 test 尚未启动；UAMCTS BLOCKED。
- Table2 当前已完整闭环：RL-Only、LLM-RL；LWM test 中；WM-RL 未启动。
- Table3 尚无最终闭环行；Full-Reward 依赖当前 LWM test，Fail-Only validation 中，Delay-Only 未启动。

RSMBRL repeat1 源结果已经 PASS，不得重跑。它只因旧 manifest 缺新版聚合器要求的 `method_artifacts.normalizer_sidecar_sha256` 而尚未进入当前严格 discovery；应创建只读、SHA 绑定的兼容准入派生物，不能修改源 manifest/eligibility。

### 12.2 LLM-RL repeat1 已完整闭环 PASS

源目录：`outputs/formal_v3/table2/llm_rl_trueppo_20260922_repeat1/repeat_1`。100 seeds `4000..4099`×500，writer/lock 已 `RELEASED/aggregate`；独立官方 `metrics_v3` 重算与存储逐项差值均为 0：

- Official Reward: `-5740.44`
- Operation Failure Penalty: `4774.51`
- Recovery Precision: `0.9191983434403195`
- Censored Recovery Time: `233.313819699077`

关键绑定：checkpoint `52a25465c059b7df4de9a13ff828d5f20afc1872b41e9c53a82dd932a3998c51`；aggregate `0d8655afc639379b3f55a485637135b9f5a5ca008ed546619e9683d082e62d8d`；source manifest `574c04a6ae5cdc76cd52587eb4d637bedfb1bc85da7cb15182f3da296ace2dcc`；source eligibility `5a76940b24cb1cd0056173375f993afc714f6bb82ead92cde1a7faef5b00cb76`。

准入桥：`outputs/formal_v3/admission_derived/table2/llm_rl/repeat_1`，generic validator/discovery PASS，状态 `PROVISIONAL`、`paper_row_eligible=false`；bridge manifest `3e73ee3e7bdbe8f658f8dc7cb489b163c0dd5280e6570bd378681d54b31b5c8e`，report `6c623ac4dbd276057ade1c39547077ac169b6fabe102654f12442212dfa958ca`。

不可覆盖备份已通过 dry-run/create/verify：`C:\aptdetect_experiment_backups\llm_rl_trueppo_20260922_repeat1_pass_20260921T233000Z.finalized`，84 files / 659,046,883 bytes；backup manifest SHA `fb6d15eab8cf7d9d145626de52554305ea7cfcacb066881d5ebef92d191f728f`。

### 12.3 当前活动 writer 与资源快照

07:29:56 只读快照：CPU `20.8%`；free RAM `5.43/15.29 GiB`；GPU `88%`，VRAM used/free `1310/6639 MiB`。

- LWM-RL test：链 `23976 -> 34972`，runtime CPU time `4:20:37`，`ACTIVE/test`，唯一 writer。
- Fail-Only validation：链 `48940 -> 31940`，runtime CPU time `1:01:40`，`ACTIVE/validation`，唯一 writer。
- CARL test retry1 已在资源门通过后启动：clean HEAD `6459a142640a3e1baaa746e38a5419688b26b1a7`；outer PID `47716`，runner PID `41856`，child Python `43452`；CPU、policy51001、seeds `4000..4099`×500、H4 `ADAPTED_TRUNCATED`；target `outputs/formal_v3/table1/carl_cc4/repeat_1_retry1`，writer `ACTIVE`。未启动 H256/repeat2-5。

### 12.4 TERLA 启动前主审新增阻断

TERLA 已将 validation 与纯 test root 分离，候选 clean HEAD `c5d1f8735daff0e631ac69c8777236b41bf66cb9`，相关测试通过；validation 三个关键 SHA 仍保持不变。但主窗口发现生产 test 路径当前自行写出的 eligibility 缺 generic discovery 要求的 `validator/input_sha256`，且 generic manifest 缺冻结 Table1 TERLA identity 字段，因此 test 暂未启动。必须先补齐 Table1 identity，并由 `validate_run_directory(..., formal=True)` 生成唯一 canonical eligibility，证明 `discover_formal_runs` 接受后再授权。该修复不允许改变已有 validation 字节。

### 12.5 聚合与后续并行

已委派 Luna/xhigh 实现独立 repeat1 provisional 三表 writer；它必须与现有五重复正式 writer 分离，非 DCA 行只接受 repeat1，DCA 复用不可变五重复，缺失/BLOCKED 行显示 `—`，所有产物醒目标记 `PROVISIONAL / NOT PAPER ELIGIBLE`。当前不生成伪全量数值。

下一个空代理槽优先处理 RSMBRL 只读兼容准入；下一轮资源核验若 CARL 启动后仍安全，再决定 WM-RL/Delay-Only calibration。禁止为提高利用率而降低 RAM/GPU 门值。

## 13. 08:05 增量状态（覆盖第 12 节相关项）

### 13.1 RSMBRL repeat1 compatibility admission 已 PASS

RSMBRL 原始正式实验早已完成，本轮未重跑且未修改源/既有备份。新增 fail-closed adapter 在独立 clean worktree `D:\w\rsmadmit`，HEAD `3a78ea11a5a043e5dce6f8bceeb6160fbe3e6cfb`；34 项测试 PASS。它只修复旧 manifest 相对当前 schema 唯一缺失的 `normalizer_sidecar_sha256`，该值由真实 frozen sidecar 计算为 `c0db1c4cef39abd9db28bcb6c6e18ebc365cbaf3b3968f8b4e29bf3150cc0d49`，并同时绑定原 manifest/eligibility、immutable backup、normalizer、D27/H4 与 train/validation/calibration split。

实际只读 bridge：`outputs/formal_v3/admission_derived/table1/rsmbrl_cc4/repeat_1`；`discover_formal_runs` accepted=1/rejected=0。bridge manifest SHA `b67038ad540e933886d3095f498fe6b35072f782435132aa41b4e532f84b58a6`；report SHA `4f2735e6e27fc4c8444b6ea3163c5b6caa881b5adf73b74dd0696dc178cf480f`。100 seeds `4000..4099`×500、policy51001、206522 decisions、provider/cache/test updates=0；指标重算与存储逐项差值0：reward `-6227.94`，failure `5156.49`，precision `0.2161925601750547`，censored time `330.90568539698324`。

### 13.2 CARL repeat1 test 完成但 eligibility FAIL

CARL fresh retry1 已完整执行 100 seeds `4000..4099`×500，原始资产保留于 `outputs/formal_v3/table1/carl_cc4/repeat_1_retry1`；writer 已退出。manifest clean commit/CPU/policy51001/ADAPTED_TRUNCATED 身份匹配，但 official eligibility `passed=false`，唯一错误为 `recovery precision denominator is zero; formal review required`。正式指标文件记录：reward `-6550.27`，failure `5430.13`，precision `null`，censored time `353.43640426263323`，5818 incidents 全部未恢复。

只读审计 decisions 共129399条，执行动作只有 Analyse 120850、Sleep 8549；Remove/Restore 为0，episodes 中 recovery_actions 为0。因此不能把该结果接入 provisional 数值表，也不能填造 precision。正在只用 train/validation 证据审查 action mapping、selection 与策略退化；严禁根据 test seeds 调参或直接重跑。若存在训练/验证集可证明的实现缺陷，必须新建 fresh lineage/target；否则 CARL 保持 BLOCKED。

### 13.3 资源与 TERLA

07:58 快照时 CARL 已释放，free RAM 5.12 GiB、GPU85%，但随后 TERLA executor 启动前瞬时复核 free RAM 降为3.57 GiB，故按4.5 GiB门禁保持 WAIT，没有创建 test target。既有 `validation_v1` launcher 仅支持 preflight/validation；另一个 stage launcher已有旧日志 root，不能覆盖。下一步必须提供 fresh test log root/明确 test execute 入口后再启动，validation `repeat_1_eval_v1` 保持字节不变。

### 13.4 provisional writer 主审

独立实现位于分支 `codex/repeat1-provisional`，HEAD `b6bbea1b93a4f63ce43c72f3dfc3c2f3af68eeb7`，9项相关测试 PASS，未改五重复正式聚合器。主审发现合入当前已支持 admission bridge 的 discovery 后，writer 的补充 bridge 扫描可能重复发现同一物理 bridge；合入前必须去重并增加真实 discovery 集成测试，同时报告 12 个物理结果与14个逻辑行的完成数。不得直接用当前版本生成最终临时表。

## 14. 09:11 增量状态（当前最新；覆盖第 13 节实时项）

### 14.1 已准入完成度与 TERLA 兼容门

当前严格准入仍为 `5/12` 个唯一物理结果：DCA、RSMBRL、PriorRL、RL-Only、LLM-RL。TERLA 的正式计算已经完整完成，100 seeds `4000..4099`×500、policy51001、provider0，独立 `metrics_v3` 重算逐项一致，但当前 generic eligibility 因唯一显示名兼容问题保持 FAIL：manifest 为 `method=TERLA-A4 (adapted)`、`method_slug=terla_a4`，旧 canonical 表只接受无 adapted 后缀，错误为 `method does not match formal method_slug name`。不得重跑 TERLA；只能新增严格、SHA 绑定的兼容准入，且不得放宽其他方法或 seed/tick/provider/update/lineage 门。

TERLA 结果目录：`outputs/formal_v3/table1/terla_a4/repeat_1`。关键不变证据：manifest `68ead04be527699af33a8f32214434979d3aa1ea78972353a63dbf5db67d5668`；metrics `b30dd78fdd764aa1dc667bf51003bdea17ea0a5f5d1c14c121b1bd5410516465`；episodes `9766e354538df19a01b590af020cfd9e103cf2d689ef35ff823c747764e75893`；decisions `1905c1bb075de1bedf412824ea73802195dfd820184f3739d79796ff21a6ea01`；checkpoint `7c5e823f741ae805f751ccfa02444931f07e92f7687bcd39ca5000304c4ab783`；optimizer `5a153cf8f26ad1259f08743b6800f7576b88a25edb01dcc268e6801917e791f3`；training manifest `a54ba4e6df63a7f7b03e78044e192240d77242fc239fc0db1cb8f38eaf915921`。四项指标为 reward `-6579.24`、failure `5428.7`、precision `0.368987480781902`、censored time `298.9814518708027`。专用 Luna/xhigh 正在只读兼容 admission、discovery 与正式备份收尾；没有新实验。

### 14.2 09:08 单次只读运行快照

- LWM-RL：`ACTIVE/test`，PID `34972`，累计 CPU `21276.22s`，train32/32；尚无最终 test summary/metrics/eligibility/aggregate，stderr 只有 Gym/lz4 警告。禁止重启或并行重复 seeds。
- Fail-Only：validation 已 `RELEASED`，PID `31940` 退出，train32/32、validation records 已落盘，provider0/test=false。已立即委派完整 8-seed 验收与未批准 prepare-test；主窗口批准前不得 test。
- WM-RL：`ACTIVE/train`，PID `30284`，累计 CPU `395.81s`，train1/32，provider0/test=false；独立政策和完整 Full-Reward 身份保持。
- 全机 CPU `19.77%`、free RAM `3.68 GiB`、GPU `89%`、显存 used/free `1539/6410 MiB`。Delay-Only calibration 因 free RAM 未达冻结门 `4 GiB` 暂不启动；不得降低门值。下一次资源释放事件即复核，不等待固定15分钟。

### 14.3 CARL 与 provisional 集成

CARL 旧 `repeat_1_retry1` 失败结果继续保留且不准入。修复候选位于 clean `D:\w\carl` HEAD `28c8bdbd690268403642f4cfed24aab2dcb4bd8a`：已淘汰把所有动作固定为 `incident_change=0` 的离线 validation proxy，改为每个候选只在 validation seeds `2000..2007` 的真实环境选择，并要求 recovery denominator>0 且 TP>0；当前主审仍要求把每候选/每 seed 的原始 episode/decision 或等价完整可复算证据及 SHA 持久化后才允许 fresh train。严禁用旧 test 结果选参。

repeat1 provisional writer 当前 clean HEAD `2a20a9b55b422dbba994bdf43c8cf4d59e6cc9c5`，已增加 native/supplemental bridge 去重；RSM adapter clean HEAD `3a78ea11a5a043e5dce6f8bceeb6160fbe3e6cfb`。后续必须在独立 clean integration worktree 合并 true-PPO admission、RSM adapter 与 provisional writer，不得覆盖主仓 dirty 用户资产。只有12/12物理行都有合法数值时才能称 `COMPLETE_PROVISIONAL`；缺失行必须继续显示 `BLOCKED/—`。

### 14.4 Fail-Only validation PASS，test 已批准但等待资源

Fail-Only train/validation 主审 PASS：train seeds `1000..1031`×500，validation 每个候选均为 `2000..2007`×500；候选 epoch 8/16/24/32 的 validation score 均为 `-5407.0`，按预声明 `max_score_then_min_epoch_then_min_checkpoint_sha256` 选择 epoch8。四个 checkpoint 实算 SHA 与记录逐项一致；selected checkpoint SHA `fb469030d0628338a13fb6a327660a78bf3ad0529eb85d5ea082c34ff6dc607e`；selection SHA `9b73c6885266aa0e784665dcd3dc5a5d1bc5fa36d975d2dcb553ed2f459cd1b4`；provider/cache/test updates 均0，on-policy true，offline false，Fail-Only 双 reward 通道与冻结 predictor 身份一致。

主窗口已审核并授权原生 authorization。批准前文件 SHA `fc57c9fadceda5defb3dc4d168bd5be6894c0cac66c3283cb68cbc3c4d363d95`；批准后文件 SHA `5c7c5ee1d4cafa733a67928e779012958641adea4ffd1e41d88d7d5405cf9fec`，self manifest SHA `235c73d6eabea88ef70e4244006711a08056478b374e8460ebb70c6b5bb7269c`。09:23:07 单次启动门：free RAM `3753 MiB`（唯一失败项，要求>=4096），CPU `21.3%`、GPU `83%`、free VRAM `6408 MiB`、target writer0。因此状态为 `APPROVED / READY_WAIT_RESOURCE`，未创建 test log、未启动 test、未轮询或降低门值。下一次资源释放事件优先启动该 test。

### 14.5 CARL fresh 科学修复与加速门

CARL fresh 修复分支 `D:\w\carl` 已推进至 clean HEAD `ab4715a7`（父提交 `2a0e53e9`，再上游 `28c8bdbd`），仍未启动 train/validation/test。相对旧失败 lineage 的实质修复保持：train-only action-conditioned SCM、真实环境 validation、recovery denominator/TP 硬门、无 test 选参。为缩短关键路径，候选在 test 前固定为 epoch `8/16/24/32`，训练仍完整32 seeds；validation 从32候选×8降为4×8 episodes，不改变验证 seed/tick 或选择门。

每候选的8个 validation raw episodes、decisions、metrics 与 manifest 将持久化，selection/training manifest/checkpoint 绑定相对路径和文件 SHA；resume/final preflight 从 raw 独立重算官方指标、CAICS score、TP/FP/denominator/gate，绝对路径、`..`、缺失或篡改均 fail closed。主审新增的最后性能门是：不得在100个 test episode中重复100次全量读取4×8证据；必须在 test 前完整审一次并形成 frozen bundle SHA，结束后完整复验一次，中间只核验身份。该优化完成前 fresh train仍未授权。

### 14.6 09:32 运行快照

- LWM-RL：`ACTIVE/test` PID `34972`，CPU `22702.3s`，train32/32；仍无最终 test artifacts。
- WM-RL：`ACTIVE/train` PID `30284`，完成 `7/32`，test=false。
- Fail-Only：`RELEASED/prepare-test`，approved authorization 已存在，test=false、writer0。
- 资源：CPU `22.7%`，free RAM `4013 MiB`（仍低于4096），GPU `94%`，显存 used/free `1564/6385 MiB`。没有启动门转绿事件，未增加 writer。
