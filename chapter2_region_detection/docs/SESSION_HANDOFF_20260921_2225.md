# CC4 v3 Repeat-1 冲刺交接（2026-09-21 22:25 +08:00）

## 1. 最终目标与当前覆盖

最终交付仍是第三版论文 Table 1、Table 2、Table 3 和正式图的可重算、可审计结果。
当前用户覆盖是先产出每个方法/消融的完整 `repeat_1` 第一版结果：独立训练、validation、
固定 test seeds `4000..4099`、每 episode 500 ticks。除既有只读 DCA 五 repeats 外，不启动
repeat 2--5；单 repeat 只能标为 `PROVISIONAL`。

最新奖励规则：没有论文独立 reward 的 baseline 使用与 LWM-RL 相同的第三版 Full-Reward；
有论文自身 reward 的 baseline 保留自身 reward；所有方法最终仍由统一 evaluator 计算
Official Reward、Failure Penalty、Recovery Precision、Censored Recovery Time。当前取消不阻塞
repeat 1 四指标的额外 pilot、扩展调参、诊断消融和非交付实验，但门禁、test 隔离、原始事件、
SHA、备份和 provisional/eligibility 审计不得省略。

- 强制要求：`docs/USER_MANDATORY_REQUIREMENTS.md`
  SHA256 `5450bc5d692f4441497737f291121923693d15f6a27f92b5f55d0c24d91026a8`。
- 任务书：`CC4_V3_0917_FINAL_EXPERIMENT_TASKBOOK.md`
  SHA256 `6d862e9475c92a668358f69d9e09e9b32b7d23cd83679dcd4bd0d3847eae00a5`。

## 2. 当前活动 writer（实时线索，接管后必须重查）

### 2.1 LWM-RL true-PPO v3 repeat 1（最高优先级）

- 输出：`outputs/formal_v3/table2/lwm_rl_trueppo_v3/repeat_1`；同时是 Table 1 Ours，
  Table 3 Full-Reward 只能 alias 此结果。
- worktree：`D:\w\trueppo_integrated`，HEAD
  `87a52483fbda4c4c9f383d86adc38a260d81cbc2`，22:24 clean。
- 初始 Python 链：shim `6796` -> runtime `36728`；writer lock `ACTIVE/train`。
- 身份：`lwm_rl`、Full-Reward、policy seed `51001`、train seeds `1000..1031`、
  500 ticks、K=6、H=4、CUDA、provider calls 0、true on-policy、offline=false。
- 22:24 已 durable seed `1000`，curve/resume 各 1，正在 seed `1001`；validation/test 未开始。
- launcher：`outputs/launcher_scripts/lwm_rl_trueppo_v3_repeat1_calibration_fix_20260921.ps1`，
  SHA `d2e0c450a30b2caea52345eb6a9e67d586bd3b05d992e6e84cc1fd06b9f0d70e`。
- preflight SHA `8e097a9c6047f936323db291c62fef642caadc856398cb090e923fec74e53795`；
  contract SHA `14079850b7babf19891fc048608e4f9f16a03d7157407eb47eae3d271df53d4c`。
- repeat 内 calibration manifest SHA
  `8b16d4d4d903135cddd411f39f8824549e3dedef7372db675ffda1c924ac1d98`；normalizer SHA
  `e50df37c17cf7c49d2c3127800144a948a1a709c5a8396e773d2fb71f8765025`。
- 本轮修复的是 fitted calibration manifest 对 source manifest 的引用哈希语义；错误路径、SHA、
  缺半字段仍 fail closed。定向 2/2、stage 16/16、runner 38/38（3 skips）PASS。
- train 启动时 mandatory SHA 是旧值 `d141ceda...1ce95f42`；随后用户只新增 baseline 奖励和
  调度覆盖，未改变 LWM 算法、数据、指标或冻结依赖，因此记录时序但不停止/重跑。

### 2.2 UAMCTS Table 1 repeat 1

- 输出：`outputs/formal_v3/table1/uamcts_cc4/repeat_1`；worktree `D:\w\uamctsformal`，
  HEAD `f620fc39698e1c922657fb557db3dacdfa867579`，clean。
- 初始链 `12048 -> 33272 -> 32844`；policy51001、test4000..4099、500 ticks、CUDA、provider0。
- 22:24 为 seeds4000..4008，共 `9/100`；约 62 分钟/episode。
- 9 episodes 备份：
  `outputs/dependency_backups/uamcts_repeat1_partial_9eps_20260921_221501/backup_manifest.json`，
  SHA `860ef4e3aa954854513a5cb7b6a8fa6b1301b3e6655ea383053e944a726735c2`。
- 加速分支 edge benchmark 约 1.934x，但真实端到端 equivalence 仍被冻结 prototype OOD 门阻塞；
  禁止把旧 9 episodes 与新 commit 分片混合，也禁止当前切换/重启。继续 canonical writer。

### 2.3 RSMBRL Table 1 repeat 1

- 输出：`D:\w\rsmbrl_v32\chapter2_region_detection\outputs\formal_v3\table1\rsmbrl_cc4\repeat_1`；
  HEAD `c9e0f57137274669da05e8dcd3d54f2c1e12bbd5`，clean。
- 链 `4284 -> 24192 -> 36508`；policy51001、test4000..4099、500 ticks、CPU、beta=1、provider0。
- 22:24 为 `86/100`；完成后先审核 metrics/manifest/eligibility、冻结哈希并备份。

## 3. 已完成的有用里程碑与备份

- DCA-CC4：唯一正式完整行，5×100×500、五 repeats PASS；不可重跑或覆盖。
- RL-Only true-PPO v2 repeat1 train：32/32 seeds1000..1031 完成，test_started=false、provider0、
  on-policy=true。备份：
  `C:\aptdetect_experiment_backups\rlonly_repeat1_train32_partial_20260921T1405Z_20260921T140713.998046Z_a48dff1da9754088b075f91a5dc9a620.finalized\backup_manifest.json`，
  SHA `0a34c233baa147782737c33c04645e54ee5bdbd6f1dde17ad9cad4802908802e`。
  validation ready launcher已生成，但因主方案优先/RAM门未启动。
- LWM v3 calibration 备份：
  `outputs/formal_v3/dependencies/backups/lwm_rl_trueppo_v3_repeat1_calibration_fix_20260921T2215`。
- LWM v3 seed1000 备份：
  `outputs/formal_v3/dependencies/backups/lwm_rl_trueppo_v3_repeat1_seed1000_20260921T2222`。
- 旧 LWM true-PPO OOD partial（8 train seeds）备份：
  `outputs/dependency_backups/lwm_trueppo_repeat1_ood_partial_20260921T2055Z`；只能 FAILED/PARTIAL。

## 4. 旧资产与临时指标结论

当前没有任何可按第三版统一 evaluator 比较的 LWM-RL 临时四指标：所有保留 LWM/AWR 资产
均 `N_test=0` 或为旧 100-step 开发 probe。旧 B4 累计 official reward `-2400/-2525` 缺少
failure/recovery/incident 原始字段、有在线调用且协议不同，禁止填表。训练 objective 也不得冒充
Official Reward。

## 5. 当前 ready/未启动队列

- RL-Only validation：ready，等待 LWM 健康运行且 RAM≥4GiB；只允许 seeds2000..2007×500，
  完成后主窗口审核，绝不自动 test。
- PriorRL：`D:\w\priorrl` HEAD `3dc6228d42ae4e69f7c72e6653043ba211ad88b7` clean；
  v4 alpha-selection ready，等待 RSM 释放 CPU/RAM并更新 mandatory SHA pin。
- LLM-RL、WM-RL：Luna xhigh 正在从 commit87a52483建立独立 worktree和最新 preflight/launcher；
  不得修改活动 LWM worktree，且当前不启动第二个 GPU train。
- CARL、TERLA：Luna xhigh 正审核最新 worktree/launcher/reward契约，按资源门进入单一 CPU 队列。
- Table 3 Delay-Only、Fail-Only：predictor 依赖已有，但尚无 true-PPO repeat1 policy结果；
  Full-Reward 只 alias 当前 LWM，不重复跑。

## 6. Git 状态

- 主仓库 HEAD `fa52f8064527971cc9a2c0fe2288927f025ebb20`，dirty 内容包含用户/阶段文档、
  UAMCTS 加速审计及既有 TERLA 恢复目录；不得 clean/restore/reset。
- clean worktrees：trueppo_integrated `87a52483...`、trueppo_rlonly `28372df...`、
  uamctsformal `f620fc...`、rsmbrl_v32 `c9e0f571...`、priorrl `3dc6228d...`、
  carl `7bc76bc6...`。

## 7. 固定监控与活跃子任务

- 唯一 heartbeat：automation `cc4-15`，Luna high、每15分钟只读；已更新监控 LWM v3、
  UAMCTS、RSMBRL。正常保持静默，只在完成/异常/漂移/OOM/越权时通知，不启动下一阶段。
- Luna xhigh `baseline_repeat1_ready_queue`：审核 PriorRL/TERLA/CARL 最小 repeat1 队列。
- Luna xhigh `table2_llm_wm_repeat1_ready`：准备 LLM-RL/WM-RL 独立最新启动包。

## 8. 下一接管顺序

1. 只读复核三条活动 writer；不得高频轮询或创建重复 writer。
2. LWM train 达32/32后核验 seed连续、journal/curve/checkpoint SHA、provider0、无test并备份；
   再单独运行 validation2000..2007，人工审核后才 prepare-test/test4000..4099。
3. RSM 100/100完成即审核、重算、备份；释放资源后在 RAM≥4GiB、CPU<85% 条件下启动
   PriorRL alpha-selection或队列中最快且已完全READY的一个CPU baseline。
4. RL-Only validation只在不影响LWM且资源门满足时启动；随后仍需人工阶段审核。
5. LLM-RL、WM-RL、Delay-Only、Fail-Only各自独立初始化，只跑repeat1必要阶段；禁止共享policy。
6. UAMCTS保持当前canonical进程；在端到端等价与无损切换未通过前不采用加速分支。
7. 任一完整训练/validation/test milestone立即不可覆盖备份并核验manifest/逐文件SHA。
