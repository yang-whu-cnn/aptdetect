# CC4 v3 阶段交接：2026-09-21 20:20（Asia/Shanghai）

> 本文按 `USER_MANDATORY_REQUIREMENTS.md` 第 11 条新建，不覆盖历史文件。PID、资源与
> episode/seed 进度均为时间点快照；接管后必须实时复核。dependency、partial、train-only
> 与 alpha-selection 均不是 paper result。

## 1. 最终目标与冻结协议

最终交付仍是第三版论文 Table 1、Table 2、Table 3 及正式图。正式口径保持 CC4 / D27 /
A4、五个独立 policy repeats、每 repeat 测试 seeds `4000..4099`、每 episode 500 ticks，
完整 raw/manifest/SHA/provenance 且 `eligibility_report.passed=true`。

- Table 1：UAMCTS、RSMBRL、CARL、DCA、PriorRL、TERLA、LWM-RL；
- Table 2：RL-Only、LLM-RL、WM-RL、LWM-RL，分别 fresh 初始化与训练；
- Table 3：Delay-Only、Fail-Only、Full-Reward；Full-Reward 只读 alias Table 2 LWM-RL；
- 优先保证 LLM prior + world model + true on-policy clipped PPO 的 LWM-RL 正确、有效；baseline
  只要求正确、公平、无泄漏、可审计，不扩大调参或挑优。

严格 paper-row 完成度仍是 `1/14 = 7.14%`：仅 DCA-CC4 的 `5×100×500` 已证明 PASS。

## 2. 本次重新读取的权威文件

- `docs/USER_MANDATORY_REQUIREMENTS.md` SHA256
  `04e1319127cb47c4893842c8179ef87bb52d023c33e16cb09d800c5d07c5ccb7`；
- `CC4_V3_0917_FINAL_EXPERIMENT_TASKBOOK.md` SHA256
  `6d862e9475c92a668358f69d9e09e9b32b7d23cd83679dcd4bd0d3847eae00a5`；
- `docs/CHATGPT_HANDOFF_FINAL_EXPERIMENTS.md` SHA256
  `9b8b146fc24c4ceefeb8b925439535bb3fc6b239828274403178cd557af2c75e`；
- `docs/FINAL_REWARD_REPLAY_MANIFEST.json` SHA256
  `b7aae1a9189afad9717be7ebfff97ef2016bb25634ae8e8c83c254987653f8ee`；
- `docs/FINAL_REWARD_MODEL_MANIFEST.json` SHA256
  `0a8a2c5c196dee74eda94cca24e4904cf93badf9008bf6ddcc27e6b854503931`；
- 上一交接 `docs/SESSION_HANDOFF_20260921_1705.md` SHA256
  `873a555b4bf20a6100d87f1a13a2860ecfae570e310873759165d78161b83c146`。

## 3. 本阶段已完成并验收的依赖

### 3.1 final GPT-5.6 Sol prior cache binding

- true-PPO commit：`f34f8bdc0daa1ad2ce781a288aef0506dfea6096`；
- sidecar：`outputs/lwm_rl_final_20260917/prior_cache_final/manifest.binding.json`；
- sidecar file SHA：`e470cd41441ae5f022e426f5f1ddb9bc58e131915fac4d2566d7fb6025e35fab`；
- source manifest SHA：`64f374b15c8cc88f463ac8d3e25eda451eed13694302abab20e1f5c61bbba40e`；
- entry-set SHA：`4fd2271f1745c27c620da12a34a173a34be0bc868eafcd2251eb29556bf47554`；
- 内容绑定 2000 train + 6 supplement、model/prompt/schema/provider、runtime provider calls 0、
  D27/train、test leak false；validator PASS；
- 不可覆盖备份：
  `outputs/dependency_backups/prior_cache_binding_20260921T1945Z`。

### 3.2 canonical true-PPO calibration 8×500

- true-PPO 当前 HEAD：`8ca8ccdbeb3158574675a7a1920d20325ccb98c7`，worktree clean；
- collector commit：`8ca8ccdb Add canonical true PPO D27 calibration collector`；
- 定向测试 6/6 PASS；seed 3000 独立重采 deterministic；
- canonical root：
  `outputs/formal_v3/dependencies/trueppo_calibration_3000_3007_500`；
- raw `calibration_states.jsonl`：4000 行，seeds 3000..3007，各 500 ticks，
  `blue_agent_0`、D27 finite；SHA
  `0543411324c483247e599ea95bfc3cbef2a0396490dec4b0f013a70a850b121e`，
  1,237,006 bytes；
- source manifest SHA：`d838a6aee92b9b8101feb0548320b55b52494329d9dda3e6d0e39ee2d5b94413`；
- normalizer file SHA：`5db617635c77ed0b08a5fd5d6c74425a301dee55d2e6929e734a77ccab4f1f4f`；
- fitted calibration manifest SHA：
  `fe0a7eb95b1fc5ff02775b85b05135f3be8555118880d68a1d0dd1db14c356d9`；
- provider 0、policy_updated false、test_leakage false；runner validators PASS；
- 备份：
  `outputs/formal_v3/dependencies/backups/trueppo_calibration_3000_3007_500_20260921_2005`。

## 4. 当前最高优先级 writer：Table 2 LWM-RL true-PPO repeat 1

- 输出：`outputs/formal_v3/table2/lwm_rl_trueppo/repeat_1`；旧 AWR
  `outputs/formal_v3/table2/lwm_rl/repeat_1..5` 完全保留且只算 DEPENDENCY-ONLY；
- worktree：`D:\w\trueppo`，HEAD
  `8ca8ccdbeb3158574675a7a1920d20325ccb98c7`，20:19 核验 clean；
- runner protocol：`cc4_v3_true_on_policy_clipped_ppo_v1`；variant `lwm_rl`；
  Full-Reward；policy seed 51001；train seeds 1000..1031；每 seed 500 ticks；K=6/H=4；
  CUDA；provider calls 0；`offline_ppo_used=false`；`on_policy_ppo_used=true`；
- preflight：当前 HEAD 下 exit 0 / PASS；`formal_result_eligible=true`；global env steps 16000；
- calibration stage：exit 0；repeat 内 normalizer/calibration manifest SHA 与上节相同；
- 运行链时间点身份：launcher/venv shim PID `21080` -> base Python PID `35572`；
  runner lock ACTIVE，stage train；统一 exec session `34282`；
- 20:18 实时状态：seed 1000 已 durable commit；`training_curve.jsonl` 1 行；正在 seed 1001；
  pending checkpoint `pending/seed_0002.pt`，pending SHA
  `73fefb782dc21e2913e11215bb0ccf519feba0f0a46114e42abd6cea180fec53`；
  `completed_train_seeds=[1000]`、provider calls 0、test_started false；
- 20:14 资源：GPU 约 91%，used 1.79 GiB / free 6.16 GiB；系统 RAM 约 2.7 GiB；
  因此禁止再加重 RAM writer，避免 OOM；
- 固定 launcher：
  `outputs/launcher_scripts/lwm_rl_trueppo_repeat1_formal_20260921.ps1`。
  本阶段主窗口修复了两个 Windows launcher 问题：PowerShell `$PID` 名冲突、native stderr
  warning 被 `ErrorActionPreference=Stop` 误杀；当前 launcher 使用唯一 `-LogTag`、保留旧日志、
  允许 runner 回收 dead ACTIVE/released lock，不删除任何证据。

注意：第一次 train 入口被 Gym stderr warning 误杀时已经产生 seed1000 pending checkpoint；runner
resume journal 在第二次启动时正确承接并 durable commit，未删除、未重采 calibration、未改协议。
原始失败日志仍在 `outputs/launcher_logs/lwm_rl_trueppo_repeat1/train.*`；当前有效运行日志前缀为
`train_resume1.*`。

## 5. 其他活动实验

### 5.1 UAMCTS Table 1 repeat 1

- 输出：`outputs/formal_v3/table1/uamcts_cc4/repeat_1`；
- worktree `D:\w\uamctsformal` HEAD `f620fc39698e1c922657fb557db3dacdfa867579`；
- 历史链 `12048 -> 33272 -> 32844`；policy51001、test4000..4099、500 ticks、CUDA、provider0；
- 20:19 为 7/100；仍只是 PROVISIONAL/PARTIAL，不能填表；继续后台，不停止、不混 commit。

### 5.2 RSMBRL Table 1 repeat 1

- 输出：`D:\w\rsmbrl_v32\chapter2_region_detection\outputs\formal_v3\table1\rsmbrl_cc4\repeat_1`；
- worktree HEAD `c9e0f57137274669da05e8dcd3d54f2c1e12bbd5`；
- 历史链 `4284 -> 24192 -> 36508`；policy51001、test4000..4099、500 ticks、CPU、beta1、provider0；
- 20:19 为 52/100；未完成 repeat，不能填表；RAM/CPU 门未允许 repeat2 并发。

## 6. PriorRL 状态、失败证据与已完成修复

- alpha_selection_v2 于 train 态出现 `PrototypeLookupMiss`：distance 2.5803386 > frozen radius
  2.56074971；无 test、无 OOM；该 run 是 FAILED/PARTIAL；
- 失败备份：
  `outputs/dependency_backups/priorrl_alpha_selection_v2_failed_20260921T194825Z`；
  backup manifest SHA
  `96e9e232648b1118d4349e3a29f80d4f10bb1e19d40f4b6583e062ea2a70c549`；
- 修复 commit：`D:\w\priorrl` HEAD
  `3dc6228d42ae4e69f7c72e6653043ba211ad88b7`，clean；70/70 tests PASS；
- OOD fallback 冻结为 train-only prototypes 的等行均值 A4 prior；不扩大 radius、不读 test/WM/
  provider；记录 state SHA/agent/distance/radius/fallback identity/mask，并绑定 checkpoint/report；
- v3 launcher：
  `outputs/formal_v3/training/priorrl_ppo_cc4/alpha_selection_v3_20260921T200820Z_launcher.ps1`，
  SHA `d5a24dd11e97c2afc03684da9b3592c3444b0369e0d21be83e0cf7e38e08586d`；
- 当前未启动：LWM writer 启动后 RAM 仅约 2.7 GiB，低于 PriorRL 4.0 GiB 安全门。不得强行
  并发；RAM 恢复后以 fresh `alpha_selection_v3` 启动，绝不 resume/覆盖 v2。

## 7. Git 与监控

- 主仓库 HEAD `fa52f8064527971cc9a2c0fe2288927f025ebb20`；dirty 包含用户/阶段 handoff、
  mandatory requirements、UAMCTS 审计和 `terla_backup_source_20260921_clone/`；不得清理或 restore；
- true-PPO worktree clean at `8ca8ccdb...`；PriorRL worktree clean at `3dc6228d...`；
- automation `cc4-15` 是唯一固定 Luna high、15 分钟只读 heartbeat；已更新纳入 UAMCTS、
  RSMBRL、LWM true-PPO train。它不启动/停止/修复任务；完成/异常才通知；
- 本阶段 Luna xhigh 子任务均已完成：prior binding、calibration collector/run、LWM launcher、
  PriorRL OOD repair。当前只剩长实验 writer 与 heartbeat。

## 8. 下一接管顺序与验收

1. 实时确认 PID35572/runner lock、RAM/GPU、`completed_train_seeds`、pending SHA、curve 行数；正常
   则不要高频轮询或触碰 writer。
2. LWM train 完成必须 32 seeds 连续 1000..1031、每 seed 500、curve/checkpoint/journal SHA
   一致、provider0、test_started false、lock RELEASED、train summary PASS；先建立不可覆盖备份。
3. 主窗口审核后才运行 LWM `validation`（2000..2007×500），随后 `prepare-test`；人工复核并
   签 test seal 后才能 test4000..4099，绝不自动越阶段。
4. RAM ≥4.0 GiB 且 CPU<85%、无 PriorRL writer 时，使用上节 v3 launcher fresh 启动 alpha
   selection；更新 heartbeat；完成后主窗口审核再安排 PriorRL 正式 repeats。
5. RSM repeat1 完成后审核 100 episodes/eligibility/hash 并备份，再按资源门启动 repeat2；不与
   低 RAM 的 LWM/PriorRL 盲目并发。
6. LWM repeat1 train/validation 健康且峰值资源已测后，生成 repeats2..5 的独立 policy/output
   启动包；GPU compute 已接近 90%，当前不并发第二个 LWM writer。RL-Only 可在 RAM 恢复时 CPU/
   GPU 余量运行，但必须独立 fresh policy。
7. Table3 Delay predictor 可作为 dependency；Fail-Only runner 当前期待 manifest SHA
   `47c2644c...` 与实际 `8295b517...` 不一致，保持 fail closed，禁止放宽门；Full-Reward 只 alias。
8. 任何完整 milestone 立即备份并验证文件数/bytes/manifest/逐文件 SHA；任何 partial 不得提升
   为 paper result。
