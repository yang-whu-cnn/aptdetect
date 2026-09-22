# CC4 v3 repeat-1 冲刺交接（2026-09-21 23:52 +08:00）

## 1. 最终目标与冻结规则

- 第三版论文 Table 1/2/3 先完成每个方法/消融唯一完整 `repeat_1`，包含正式训练、validation、test seeds `4000..4099`、每 episode 500 ticks；统一标 `PROVISIONAL`。
- 不启动 repeat 2--5；DCA 既有五 repeats 只读。无独立奖励的 LWM-RL/RSMBRL/PriorRL 用 Full-Reward；有论文奖励的基线保留原奖励。Table3 Full-Reward 与 Table2 LWM-RL、Table1 Ours 是同一物理 run。
- 实验由 Luna xhigh 执行，主窗口设计/审核；唯一 `cc4-15` heartbeat 每15分钟只读核验。禁止删除/reset/clean/restore/覆盖，结果及时备份。
- 当前只做本地实验；不得 commit/push/sync，等待用户明确要求。
- 强制要求 SHA `5450bc5d692f4441497737f291121923693d15f6a27f92b5f55d0c24d91026a8`；任务书 SHA `6d862e9475c92a668358f69d9e09e9b32b7d23cd83679dcd4bd0d3847eae00a5`。

## 2. 当前两个活动 writer

### 2.1 LWM-RL true-PPO v3 repeat1 train

- Target：`outputs/formal_v3/table2/lwm_rl_trueppo_v3/repeat_1`。
- Worktree `D:\w\trueppo_integrated`，HEAD `87a52483fbda4c4c9f383d86adc38a260d81cbc2` clean。
- Launcher SHA `562811eaace6a9e2786ba7c83f72b0d9a1a694c2801dc7ca31f19464a1dee55a`；PID 链 `41176 -> 39832 -> 22220`。
- 23:51 已连续完成 22/32，最新 `checkpoint_0022.pt`；Full-Reward、policy51001、train1000..1031×500、K6/H4、CUDA、provider0、test=false。
- 32/32 后先审核连续 seed/curve/journal/checkpoint SHA 并备份，再单独 validation；不得自动 test。

### 2.2 PriorRL alpha-selection v5

- 旧 `powershell.exe` 启动失败的准确根因是环境无法识别 `Get-FileHash`，与算法/数据无关；没有产生正式 writer/target。改用已验证的 bundled `pwsh.exe`，不改 launcher。
- Worktree `D:\w\priorrl`，HEAD `3dc6228d42ae4e69f7c72e6653043ba211ad88b7` clean。
- Launcher `outputs/formal_v3/training/priorrl_ppo_cc4/alpha_selection_v5_20260921T2225Z_launcher.ps1`，SHA `2ddbbf0c3dbc42d0ca8335beb2c99eba03c4e32575206a7772f7eb033d344cab`。
- 23:50 成功启动，链 `33248 -> 43236 -> 33872`；target `outputs/formal_v3/training/priorrl_ppo_cc4/alpha_selection_v5_20260921T2225Z`。
- 23:51 已产生 `alpha_0.01/checkpoint.pt` 与 SHA sidecar；冻结 grid `[.01,.05,.1,.5]`、train1000..1031、validation2000..2007×500、tuning seed50999、Full-Reward、CPU、noWM/provider/test。
- 完成只审核 alpha report；不得自动进入 train-repeat/test。

## 3. RL-Only validation 已按用户优先级停止

- 原链 `44964 -> 23740 -> 25400 -> 41488`，只停止叶 writer `41488`，父链自然退出；LWM 未受影响。
- runner validation 是 4 checkpoints × 8 validation seeds = 32 个500-tick episodes，且最终才原子写 records。停止时 durable validation=`0/32`，无 validation/test/provider 文件；训练32/32的6项 SHA不漂移。
- stale lock 保留为证据，status ACTIVE/pid41488/stage validation；未来 runner 会 RECLAIM，不手工修复。
- 停止备份：`outputs/formal_v3/dependencies/backups/rlonly_repeat1_validation_STOPPED_BY_USER_PRIORITY_RESCHEDULE_20260921T154131678Z_d2f9640c7c394aef9bf78247557505bc`。
- 权威 `backup_manifest_v3.json` SHA `ebfc2166c31155f43cf74c36d58e391a13453279b3817f73b067d9bc05978d6d`，12 files/139236 bytes。
- 以后资源空闲时从 validation 完整重跑，不必重训。

## 4. 已完成/准备状态

- DCA：既有五 repeats PASS，永不重跑/覆盖。
- RSMBRL repeat1：100/100×500、eligibility PASS；corrected backup manifest SHA `c35fb2eb6b526cd0c0da8786cfe3f9652917326e32145ecc4fcc7c1a25828b36`。
- Table3 Fail-Only：worktree `D:\w\t3fail` HEAD `b094ae9ba13cf20add2c0353c9457b20fccc6395` clean，review ancestor PASS；launcher SHA `1c2659d3473d517bd0abd7b611ccd669525a35165921a79415a9bf7c0f9e80b5`，preflight PASS，未启动。
- Table3 Delay-Only、Table2 LLM-RL/WM-RL 均 ready，未启动。
- UAMCTS v3：worktree `D:\w\uamctsv2` HEAD `e21fad1827abcf54a0850914d9e8a1095cce1ba2` clean；synthetic equivalence PASS、merge gate PASS，但500-tick paired equivalence和single-shard资源实测仍BLOCKED；不得test。
- CARL 要求较高 RAM；TERLA 仍缺完整审计后的 validation/test 路径，held。

## 5. 当前进度口径与下一步

- 三表共有14个展示行；LWM/Ours/Full-Reward三行共享一个物理结果，因此共有12个唯一 repeat1 物理结果。
- 已完整：DCA、RSMBRL，共2/12。尚缺10个完整物理结果；其中 LWM 正在直接训练，PriorRL 的 alpha 前置正在运行，其余9个物理结果仍待完成。
- 当前 RAM 是主要瓶颈；LWM+Prior 并行后不再启动第三 writer。Prior alpha结束释放资源后，主窗口审核并在 Table3 Fail/Delay、Prior train、LWM validation 中按可用 RAM/GPU选择下一条，优先避免重跑。
- heartbeat 已更新，只把 LWM 和 Prior alpha 视为活动 writer；RL stale lock不算 writer。
