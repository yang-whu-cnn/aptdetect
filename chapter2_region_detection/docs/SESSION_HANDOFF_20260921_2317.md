# CC4 v3 Repeat-1 冲刺交接（2026-09-21 23:17 +08:00）

## 1. 最终目标与当前覆盖

最终交付仍是第三版论文 Table 1、Table 2、Table 3 和正式图的可重算、可审计结果。
当前用户覆盖为：每个方法/组件消融/奖励消融先只完成一个完整 `repeat_1`，包含独立训练、
validation、固定 test seeds `4000..4099` 和每 episode 500 ticks，作为第一版
`PROVISIONAL` 结果；除既有只读 DCA 五 repeats 外，不启动 repeat 2--5。

训练/规划奖励冻结如下：无独立 defender reward 的 LWM-RL、RSMBRL、PriorRL 使用第三版
Full-Reward；UAMCTS 使用第三版基础 reward 加原论文 potential shaping；CARL、TERLA 使用各自
论文 reward；DCA 的攻击路径 reward 仅用于推断。所有行仍由统一 evaluator 计算四指标。
取消不阻塞 repeat 1 的额外 pilot、扩展调参和诊断实验；门禁、test 隔离、原始事件、SHA、
备份和 provisional 审计不得省略。当前只做本地实验，不 commit/push/同步远端，等待用户要求。

- 强制要求 `docs/USER_MANDATORY_REQUIREMENTS.md` SHA256
  `5450bc5d692f4441497737f291121923693d15f6a27f92b5f55d0c24d91026a8`。
- 任务书 `CC4_V3_0917_FINAL_EXPERIMENT_TASKBOOK.md` SHA256
  `6d862e9475c92a668358f69d9e09e9b32b7d23cd83679dcd4bd0d3847eae00a5`。

## 2. 当前唯一正式 writer：LWM-RL true-PPO v3 repeat 1

- canonical 输出：`outputs/formal_v3/table2/lwm_rl_trueppo_v3/repeat_1`；同时是
  Table 1 Ours 和 Table 3 Full-Reward 的唯一 alias。
- worktree `D:\w\trueppo_integrated`，HEAD
  `87a52483fbda4c4c9f383d86adc38a260d81cbc2`，23:17 clean。
- 旧 Python `6796 -> 36728` 在约 22:26 无日志退出；外层旧窗口不是 writer。没有 Windows
  crash event 或 stderr，无法把原因归结为算法错误。
- 中断现场 12 个 payload 文件、9,069,116 bytes 已不可覆盖备份到
  `outputs/formal_v3/dependencies/backups/lwm_rl_trueppo_v3_repeat1_interrupted_20260921T144458Z`；
  backup manifest SHA `08519aac6b66579170729dfba3dab4f3861a2147eed914d4c240d00bfcb23165`，
  明确为 PARTIAL。
- 恢复 launcher：`outputs/launcher_scripts/lwm_rl_trueppo_v3_repeat1_resume_recover_20260921.ps1`，
  SHA `562811eaace6a9e2786ba7c83f72b0d9a1a694c2801dc7ca31f19464a1dee55a`；日志目录
  `outputs/launcher_logs/lwm_rl_trueppo_v3_repeat1_resume_recover_20260921T1449Z`。
- 当前链 `41176 -> 39832 -> 22220`；lock pid `22220`、`ACTIVE/train`。
- 23:17 已连续完成 seeds `1000..1009`，curve/journal/checkpoint 各 10，正在 seed `1010`；
  未重跑 1000/1001。Full-Reward、policy seed51001、32 train seeds、500 ticks、K6/H4、
  CUDA、provider0、true on-policy、test=false。
- 只允许 train；32/32 后先审核、备份，再单独 validation，绝不自动 test/repeat2。

## 3. 本阶段新完成的正式 repeat-1 结果：RSMBRL

- 输出：`D:\w\rsmbrl_v32\chapter2_region_detection\outputs\formal_v3\table1\rsmbrl_cc4\repeat_1`。
- HEAD `c9e0f57137274669da05e8dcd3d54f2c1e12bbd5` clean；100 个 episode seeds
  `4000..4099`，每个 `tick_count=500`，无重复；Python 已退出。
- `eligibility_report.passed=true`、无 errors/warnings，所有 manifest/validator artifact SHA 一致。
- repeat-1 指标：Official Reward `-6227.94`；Failure Penalty `5156.49`；Recovery Precision
  `0.2161925601750547`；Censored Recovery Time `330.90568539698324`；completed-only time
  `6.855633802816901`；unrecovered rate `0.952925575998674`。
- 关键 SHA：episodes `a77a4d3089ded7c8ea96ea59b4deb0b7de8dce6803a9b871ed952fc6d1904d16`；
  decisions `8cfd2450de6a1e17a804beda4c4a6c19cf2e36103ded1c6fe144d127ad2fa891`；
  metrics `249ec6fb26089e35d8a5f8a3632db727532ed4ce230497fcbe996a5b3d0d63d5`；
  manifest `475ad51b3c3c07b21cd7ceee0442a7c098690ceb1c5f763bc181caf43c62c3c7`。
- 正确备份：
  `outputs/formal_v3/dependencies/backups/rsmbrl_repeat1_pass_20260921T230914Z_v2`，
  payload 210 files / 299,022,411 bytes；以 `backup_manifest_v2.json` 为准，SHA
  `c35fb2eb6b526cd0c0da8786cfe3f9652917326e32145ecc4fcc7c1a25828b36`。
- 证据保留：首次 wildcard 复制只创建了空目录
  `.../rsmbrl_repeat1_pass_20260921T230914Z`；未删除。v2 目录首份 `backup_manifest.json`
  的 aggregate total_bytes 误为 0，但逐文件复制/hash 已正确；未覆盖该文件，新增的
  `backup_manifest_v2.json` 是权威修正版。
- 这是完整且 gate PASS 的单 repeat，只能标 PROVISIONAL，不能声称五-repeat 最终完成。

## 4. 已准备但受资源门约束的任务

### 4.1 PriorRL alpha-selection

- worktree `D:\w\priorrl` HEAD `3dc6228d42ae4e69f7c72e6653043ba211ad88b7` clean。
- launcher `outputs/formal_v3/training/priorrl_ppo_cc4/alpha_selection_v5_20260921T2225Z_launcher.ps1`，
  SHA `2ddbbf0c3dbc42d0ca8335beb2c99eba03c4e32575206a7772f7eb033d344cab`。
- 身份和 DryRun 均 PASS；target/log/PID/exit 均未产生，writer=0。
- 两次 action-time 检查分别只有约 3.50 GiB、3.08 GiB free RAM，低于冻结的 4 GiB 门，
  因此未启动。只允许 grid `.01,.05,.1,.5`、train1000..1031、validation2000..2007、
  500 ticks、seed50999、Full-Reward、无WM/provider/test。资源恢复后重试，不降低门槛。

### 4.2 RL-Only validation

- train 已完成32/32并有既有外部备份；worktree `D:\w\trueppo_rlonly` HEAD
  `28372df20ebf544771dae5b117e23cf7b57399e8` clean。
- 最新 validation-only launcher：
  `outputs/launcher_scripts/table2_rl_only_trueppo_v2_repeat1_validation_ready_20260921T2258Z.ps1`，
  SHA `5654d2329ea591804769bbd1ad7c5c53f7135f0ec0c692171581fd1631be2743`。
- 源训练 32 seeds、resume/checkpoint0032/normalizer/calibration/journal/train summary 与备份哈希
  一致；test=false/provider0/on-policy=true。23:00 free RAM 3.126 GiB，因此保持 READY_HELD。
- 只允许 validation seeds2000..2007×500；完成后主窗口审核，不能自动 test。

## 5. Table 2 / Table 3 准备状态

- LLM-RL：`D:\w\t2llmready` HEAD `669859819cda21119f2b1fb5906613b917c3d6b8` clean；
  launcher SHA `23545881384172ad6e8857405127e0c89484eb556072c77014f9238feeafe9f8`；
  29/29 tests、preflight PASS、未启动。
- WM-RL：`D:\w\t2wmready` HEAD `3009d7d9a7fa215f29d817b7a47a3b64666ec18c` clean；
  launcher SHA `3d165628077ba49cf79c332e1e880adf5a53d09a4cb245796ee03bf2f0fddc9b`；
  29/29 tests、preflight PASS、未启动。
- Delay-Only：`D:\w\t3delay` HEAD `a3c09bcdcb69bd2eda5dd40a5e7365a528bee705` clean；
  preflight PASS；launcher SHA `9047a8c5...`；84 tests PASS；未启动。
- Fail-Only：正确 checkpoint `8800a65d...`、manifest `8295b517...` 已绑定，旧错误
  `47c2644c...` 未复用；最初 preflight 因 provenance review commit 非当前 HEAD 祖先而 BLOCKED。
  Luna xhigh 正在 `D:\w\t3fail` 建立本分支两阶段 review/binding 祖先链；23:17 HEAD
  `b094ae9ba13cf20add2c0353c9457b20fccc6395` clean，尚待最终报告。
- Full-Reward 不建独立物理 run，只 alias LWM canonical。

## 6. UAMCTS 失败与恢复包

- 旧 canonical 完成 test seeds4000..4008 共9个；seed4009 因 frozen prototype distance
  `2.58678643 > radius 2.560749706662155` fail closed、exit1；无活动 Python。
- 9eps 备份 manifest SHA
  `860ef4e3aa954854513a5cb7b6a8fa6b1301b3e6655ea383053e944a726735c2`，旧输出/备份未修改。
- 新 worktree `D:\w\uamctsv2` HEAD `c3ecebdc73c8a0d3765d914e1a85509808b1ff36` clean；实现
  train-only equal-row-mean OOD fallback、edge cache 和2路 shard adapter。fallback prior SHA
  `59f387c335c57217edc2678ca1bcafba202e09cfbda1be27537ffe4bba9ce440`；radius 未改；
  prototype SHA `e4359df9b6703a767ebe0e52ad09d2b208b8ddd19bd11e3820a887a249c96d09`。
- v2 contract SHA `383b9734d354020ac14364ef40ef86220e752adaba580388edd17187e677ef9f`；
  preflight SHA `7c1efe8c2f217076fe93029bf0ceafbb35332f58c55041487e0e6ffb722b65a9`；
  31 tests PASS、1 capability skip；没有新test输出。
- 仍 BLOCKED：尚未证明真实 formal 端到端等价和正式 shard merge gate，原VRAM门也不可达。
  Luna xhigh 正仅用 synthetic/calibration seed3000 做500-tick legacy+fallback vs edge+fallback 等价、
  merge preflight和资源实测；未通过前禁止启动test。

## 7. Git 与监控

- 主仓库 HEAD `fa52f8064527971cc9a2c0fe2288927f025ebb20`，dirty 内容属于既有要求/交接/UAM审计/
  TERLA恢复资产；不得 clean/restore/reset。
- 23:17 clean worktrees：trueppo_integrated、trueppo_rlonly、rsmbrl_v32、priorrl、t2llmready、
  t2wmready、t3delay、t3fail、uamctsv2。禁止远端同步，等待用户明确要求。
- 唯一 heartbeat `cc4-15`：Luna high、每15分钟只读；prompt 已更新为只监控 LWM 当前 writer，
  并记录 RSM 完成、Prior/RL-Only 资源阻塞、UAM无writer。不得启动下一阶段。
- 活跃 Luna xhigh：`table3_reward_repeat1_ready`（修Fail provenance）、
  `uamcts_ood_accel_recovery_ready`（非test等价/merge审计）。PriorRL/RL-only 启动代理均已因RAM门
  fail closed并释放。

## 8. 下一接管顺序

1. 只读监控 LWM 至32/32；核验连续 seeds、curve/journal/checkpoint SHA、provider0、无test并备份。
2. free RAM 稳定≥4GiB 时，优先启动 PriorRL alpha v5；若仍不足，不降低门槛。其完成后审核，
   再train-repeat；test必须在train/validation人工审核后单独启动。
3. 同一资源门下运行 RL-Only validation；完成后备份和审核，再prepare-test/test。
4. 审核 Fail-Only修复报告；Delay/Fail、LLM/WM都不得与主LWM盲目竞争资源。LWM训练稳定且
   GPU/RAM实测允许时可启动第二条GPU训练，否则顺序执行，避免重跑。
5. UAMCTS必须先取得 calibration500端到端等价、formal merge gate 和可达资源门 PASS；随后才可
   在全新target双shard重跑100 test，绝不混用旧9eps。
6. RSM结果只读，使用 corrected backup manifest v2；不要重跑或覆盖。
7. CARL需要≥12GiB free RAM；TERLA仍缺审计后的validation/test入口，当前不启动。
