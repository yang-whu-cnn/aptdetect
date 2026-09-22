# CC4 v3 阶段交接：2026-09-20 22:30（Asia/Shanghai）

> 本文按 `USER_MANDATORY_REQUIREMENTS.md` 第 11 条新建，不覆盖旧交接。它记录本阶段
> 已完成工作、当前运行状态和接管顺序。PID、资源和进度是时间点快照，接管后必须实时复核。

## 1. 最终任务与正式口径

最终目标是在 `D:\paper\github-me\aptdetect` 完成第三版论文可正式使用、可重算、可审计的
CC4/D27/A4 结果：Table 1 七个方法、Table 2 四个独立变体、Table 3 三个奖励消融以及任务书
规定的正式图。只有完整的 5 repeats × 100 test episodes × 500 ticks、固定 test seeds
`4000..4099`、raw artifacts/manifest/provenance/hash 全部完整且
`eligibility_report.passed=true` 的结果可以填表。LWM-RL 的效果、正确性、消融和可复现性优先；
基线只要求正确、公平、无泄漏和可审计，不为提高效果额外调参或挑优重跑。

截至本文，严格 paper-table 完成度仍为 `1/14 = 7.14%`：只有 Table 1 DCA-CC4 完整 PASS。
train-only safe stop、预测器、候选依赖和 partial episode 都不能计作完成表格行。

## 2. 本阶段核对的权威文件

| 文件 | SHA256 |
|---|---|
| `chapter2_region_detection/docs/USER_MANDATORY_REQUIREMENTS.md` | `0a8f7ad4e22ce7cfaa6bc71789fb41a69a584b9684ad13681539d9f1e65d758a` |
| `CC4_V3_0917_FINAL_EXPERIMENT_TASKBOOK.md` | `6d862e9475c92a668358f69d9e09e9b32b7d23cd83679dcd4bd0d3847eae00a5` |
| `chapter2_region_detection/docs/CHATGPT_HANDOFF_FINAL_EXPERIMENTS.md` | `9b8b146fc24c4ceefeb8b925439535bb3fc6b239828274403178cd557af2c75e` |
| `chapter2_region_detection/docs/FINAL_REWARD_REPLAY_MANIFEST.json` | `b7aae1a9189afad9717be7ebfff97ef2016bb25634ae8e8c83c254987653f8ee` |
| `chapter2_region_detection/docs/FINAL_REWARD_MODEL_MANIFEST.json` | `0a8a2c5c196dee74eda94cca24e4904cf93badf9008bf6ddcc27e6b854503931` |

## 3. 本阶段完成的正式基础工作

### 3.1 LWM-RL repeat 3 完成与不可覆盖备份

- canonical：`chapter2_region_detection/outputs/formal_v3/table2/lwm_rl/repeat_3`；
- worktree commit：`f14472dd7f467efe0e74970e34ef1942d8f11213`，clean；
- 32 journal / 32 curve / 32 checkpoints，train seeds `1000..1031` 连续且无重复；
- 32 个 checkpoint SHA 重算全匹配；无 validation/test/episode/eligible manifest；
- `exit_code.txt=3`，`STOPPED/PARTIAL`，`expected_safe_stop=true`，
  `formal_result_eligible=false`，`stop_after=train`；
- 本次训练 provider calls=0，online environment steps=0；
- 官方不可覆盖备份：
  `C:\aptdetect_experiment_backups\repeat_3_20260920T141900.886063Z_3078c7128f284c65a54f81436324aa11.finalized`；
- backup verify PASS：40 files、16,077,546 bytes；backup manifest SHA
  `15a5ec6c7d2fa4e82e1cd631ad0519437752c7a6c391bcb9180a694661ccc7bc`。

repeat 1 仍为 exact safe stop 且已备份；repeat 2 的 32 个训练 seed/checkpoint 本身完整，但外层
`exit_code.txt` 被写成字面量 `System.String`，只按 anomalous dependency evidence 保存，不能冒充
exact safe stop 或 paper result。

### 3.2 Table 3 runner、canonical provenance 与预检

`D:\w\priorrl` 已完成经过主窗口审核的合并与 canonical LF provenance 重绑定：

- runner/provenance merge commit：`c6bd264637ffccdb8fcb46b9a9703a1e46a8668c`；
- candidate anchor commit：`f441f15b6cd9252aea71ad7873bb09bf4f577fdb`；
- final approval HEAD：`844e5f920dcac9447a8a4ad6e0eaca6e02aedf92`，clean；
- 保留 offline AWR v4、runner v5、resume/execution-domain binding、Delay/Fail 两类 provenance gate；
- 固定 Python 3.11 全套 `103/103` 测试 PASS；
- Delay-Only 和 Fail-Only CUDA formal dry-run preflight 均 PASS，`errors=[]`；
- canonical replay manifest raw SHA：
  `b7aae1a9189afad9717be7ebfff97ef2016bb25634ae8e8c83c254987653f8ee`；
- Delay candidate / final sidecar SHA：
  `934158f9637d455d9c937896ced57541b1cd9bbb3e441ec2e7e4b481afdfccfc` /
  `a1982305a3388ca22246d0406a71e7f0d17b59fe7e3e3cd170f55a83c4c88a4d`；
- Fail candidate / final sidecar SHA：
  `7f6396571b0587ef22a7af656f8cbe9bf01ca2a0b1745a08f4b959bc8a2e0949` /
  `592f535baa315c0d0437c723ade7f3453bcd636d4fdff19dc0aaef09ff7bd88c`。

旧 CRLF sidecars 未删除，已非覆盖移到
`D:\w\priorrl\chapter2_region_detection\outputs\formal_v3\table3_reward_models\provenance_legacy_crlf_20260920T2205`。
旧八资产备份仍保留；当前四个新 provenance JSON 正由 Luna xhigh 子任务创建新的 verified-dependency
备份。Table 3 policy 训练尚未启动。

### 3.3 UAMCTS dependency

候选目录：`outputs/rebuild_20260920/uamcts_assets_v2`。固定 Python 3.11 定向测试 `19/19` PASS，
preflight `eligible=true`，但它仍只是正式运行依赖，不是 paper result。不可覆盖备份：

`C:\aptdetect_experiment_backups\uamcts_verified_dependency_20260920T141408.016478Z_4025ddef2a474dd7b585919c4ab92b57.finalized`

验收：11 files、335,011 bytes，manifest SHA
`53296d3941b540c509bb53ec134569bbdf4fcf90924aa52601fcd8f4c5029220`，逐文件 SHA PASS，
`PARTIAL`、`formal_result_eligible=false`、`paper_table_eligible=false`。

### 3.4 PriorRL v2 candidate

候选目录：`outputs/rebuild_20260920/priorrl_v2_candidate`。固定 Python 3.11 retriever/静态检查、
真实 CC4 100-tick probe 和定向测试已经通过；当前 artifact SHA
`8a4062a6813d8121923ecabefad846495aca84f6295217906375c790367d7b9d`，train coverage
`41079/41079`，validation `10515/10515`，2005 centers，radius `2.5265297329525533`。

但主窗口复核后结论仍是 `PASS_CANDIDATE / FORMAL BLOCKED`：任务书要求 6 条精确 train-only D27
OFOX entry，现候选只有 5 条 train-replay teacher center，第 6 条是 non-train fail-closed probe；
artifact 根的 train source/threshold SHA 也尚未与 candidate sidecars 自洽。不得放宽 validator、
不得覆盖旧 canonical、不得启动 PriorRL 正式 PPO/test。候选正在制作 blocked verified-dependency
备份；后续必须新建自洽候选和 candidate-aware fail-closed validator，再重新预检。

## 4. 当前唯一正式训练：LWM-RL repeat 4

- 启动时间：2026-09-20 22:23:04（Asia/Shanghai）；
- 统一终端 session id：`76879`，没有为运行核验另开窗口；
- launcher：`outputs/launcher_scripts/lwm_repeat4_cc4.ps1`，SHA
  `b5229086aaa6db8ccb4c630aede53ace814cd3657e454338eb019da5990efd41`；
- runner SHA：`444fa0b751e20f277204260adca2967df1a9df18dd5fa491afc8498bd9b78722`；
- worktree HEAD：`f14472dd7f467efe0e74970e34ef1942d8f11213`，clean；
- canonical output：`outputs/formal_v3/table2/lwm_rl/repeat_4`；
- 身份：Table 2 / `lwm_rl` / repeat 4 / policy seed 51004 / CUDA / 32 train seeds
  `1000..1031` / 500 ticks / `stop_after=train` / 无 resume；
- 唯一 writer 父子链：PowerShell `33776` → venv Python `32680` → base Python `28632`；旧
  PowerShell `31564` 无 Python 子 writer，不是重复实验；
- 独立启动验收时进度 2/2/2，seeds 1000、1001；checkpoint 2 SHA
  `d9d8c254823f4af6481458f552060fba1e2eb6a6bcc937cf17fd376effbd65f9`；
- provider calls=0、online environment steps=0，无 validation/test/episode/eligible 文件；
- 启动验收资源：可用 RAM 约 4.26 GiB，GPU 40%，显存 1460/8188 MiB。

资源门结论：系统内存不足以安全并行 TERLA/CARL/RSMBRL 或另一项 GPU policy 训练。因此当前
只保留一个 GPU writer；三个 Luna xhigh 子任务并行做只读审核、launcher 准备和不可覆盖备份。
这符合“安全并发”，不得为表面利用率盲目超售内存造成 OOM/重跑。

唯一 heartbeat 已从 repeat 3 原地更新为 `CC4 LWM repeat4 核验`，仍使用 automation id
`cc4-lwm-repeat3`，每 10 分钟附着当前主任务只读核验；正常无变化保持静默，完成/异常才通知，
绝不自动启动 repeat 5。

## 5. Git 快照

| 工作区 | HEAD | 状态 |
|---|---|---|
| 主仓库 `D:\paper\github-me\aptdetect` | `00de7088f5aef76e0b85c46886dd2db691a8a628` | 本文创建前 clean；本文为主窗口新增 tracked 文档 |
| LWM `C:\Users\25453\.codex\worktrees\3cf0\aptdetect` | `f14472dd7f467efe0e74970e34ef1942d8f11213` | clean |
| Table3/PriorRL `D:\w\priorrl` | `844e5f920dcac9447a8a4ad6e0eaca6e02aedf92` | clean |
| TERLA `C:\Users\25453\.codex\worktrees\a553\aptdetect` | `2eb9ba7502182c3971cf880f46b14fbe76e60f37` | clean |
| CARL `D:\w\carl` | `7bc76bc6e23ed7acb62ac7c3ec1c291b8dc8b90e` | clean |
| RSMBRL `D:\w\rsmbrl` | `da9b096efea2af621732a3a4d86239d875e5fbba` | clean |

## 6. 其余正式进度与禁区

- DCA：5×100×500、eligibility PASS，唯一完成表格行。
- LWM repeat 1、3：exact train-only safe stop 并已备份；repeat 2 训练内容完整但外层退出证据异常；
  repeat 4 正在运行；repeat 5 launcher 正由 Luna xhigh 准备，未启动。
- TERLA：policy seeds 51001..51003 完成并备份，51004/51005 未运行；当前 RAM 门阻塞。
- RSMBRL：五 repeats 各只有 seed 4000 的 durable partial，可从 4001 单调 resume，但当前 RAM 门阻塞。
- CARL：代码/预检已通过，但规定可用 RAM ≥12 GiB；当前禁止启动。
- Table 3：runner、provenance、preflight 就绪；Delay/Fail/Full 三行均无正式结果。
- Table 2 RL-Only、LLM-RL、WM-RL 尚无正式行；不得用 LWM checkpoint 替代独立训练。
- 所有正式图仍是 `0/5`，必须等消费行 eligibility PASS 后生成。

## 7. 活跃 Luna 子任务

1. `lwm_repeat4_launch_audit`：已完成 repeat4 独立启动 PASS，现只准备 repeat5 launcher；
   不启动、不创建 repeat5 输出。
2. `table3_provenance_backup_v2`：创建新 canonical Table3 provenance dependency 的不可覆盖备份。
3. `priorrl_validator_design`：已给出 PriorRL formal BLOCKED 结论，现备份 blocked candidate。
4. heartbeat `cc4-lwm-repeat3`：Luna high、10 分钟、只读、正常静默。

所有子任务禁止删除、reset、clean、restore、覆盖、推送或自行启动正式实验。

## 8. 接管后的精确顺序

1. 先完整读强制要求、任务书、两个 final manifest 和本文；实时复核 Git、PID、输出和资源。
2. 等 repeat4 exact safe stop；要求 exit 3、32/32/32、seed/SHA 连续一致、STOPPED/PARTIAL、
   provider/online steps 0、无越权阶段。主窗口审核后立即建立并 verify 不可覆盖备份。
3. 若资源与身份门通过，审核 prepared repeat5 launcher 后只启动 repeat5 train-only；同时更新现有
   heartbeat，不创建第二个监控。
4. repeat5 训练等待期间，审核 Table3 新 provenance 备份；准备 Fail-Only launcher；不得与 LWM
   争用当前低内存 GPU 环境。
5. LWM 五个训练 checkpoint 审核完成后，按任务书分阶段执行 validation/prepare_test/test，固定
   100×500 test seeds，禁止窥视调参；随后冻结 Table2 LWM-RL 和 Table1 LWM-RL 结果。
6. GPU 释放后启动 Table3 Delay-Only repeat1；其后人工审核、备份，再安排 Fail-Only；
   Full-Reward 只读复用 Table2 LWM-RL，禁止重复训练。
7. 系统内存恢复到各方法硬门后才恢复 TERLA 51004、RSMBRL/CARL；优先选择能形成完整行且
   不影响 LWM/Table3 的任务。
8. PriorRL 必须先修复 6/6 train-only 和 SHA 自洽性，完成 candidate-aware validator 与测试；
   旧 canonical 保持不动，正式 preflight PASS 前禁止训练。
9. 每项有用产出先备份并核验文件数、bytes、manifest、逐 SHA，再进入下一阶段；任何 partial
   或 blocked 资产都不得提升为 paper result。

## 9. 2026-09-20 23:35 增量状态（以后续状态为准）

### 9.1 LWM-RL repeat 4 已完成并备份

- 统一终端 session `76879` 已结束；launcher 外层退出正常，实验 `exit_code.txt=3`；
- `32/32/32` journal / curve / checkpoints，train seeds `1000..1031` 连续且唯一；
- 32 个 checkpoint SHA256 重算全部匹配；provider calls=0、online environment steps=0；
- 无 validation/test/episode/eligibility/正式 manifest；
- `STOPPED/PARTIAL`、`expected_safe_stop=true`、`formal_result_eligible=false`；
- canonical 共 40 files、16,077,086 bytes；
- 不可覆盖备份：
  `C:\aptdetect_experiment_backups\repeat_4_20260920T152923.698103Z_817234b293a840f3bdf8ecc3792ff101.finalized`；
- backup verify PASS：40 files、16,077,086 bytes；backup manifest SHA256
  `8a387f09ebbc96bfbc93b38af95ed18cb9c0c19104513a69b8d8b3f9ab18b29`
  （比较时大小写不敏感）。

### 9.2 f144 resume 缺陷与正式处置

主窗口复核确认：`f14472dd...` 的 callback 构造器会 glob 全部 32 个训练 checkpoint；train 阶段
已完成时 stage runner 跳过 `train()`，因此恢复 validation 会错误评估 32 个 checkpoint，而不是
train payload 冻结的 `0008/0016/0024/0032`。旧 repeat 1–4 的 commit/domain 又禁止用新代码直接
续接；不得修改旧 resume state、放宽 domain、使用 wrapper/monkeypatch 或运行旧 validation launcher。

因此：旧 repeat 1–4 继续作为已备份的 train-only dependency evidence，不能直接成为论文结果；
prepared f144 repeat5 和三种 Table2 消融 launcher 也全部冻结，不再启动。为了最快获得可信正式结果，
在修复通过独立审查后，先用新 commit fresh 运行尚不存在的 repeat 5，再由主窗口保留并归档旧
canonical 后 fresh 重跑 repeat 1–4。不得覆盖或删除旧产出。

修复候选位于 `D:\w\lwmres` commit
`1498e4a99b5932f955b9e4dab25c136bff86a82c`，新增显式 candidate rehydration、逐路径/SHA/domain
门和 validator provenance gate；固定 CC4 Python 3.11 定向测试 `26/26 PASS`。当前仍等待独立
Luna xhigh 只读审查，审查通过前不得 cherry-pick 或启动新正式 run。

### 9.3 资源与其他阻塞更新

- repeat4 结束后无 Python writer；可用 RAM 约 4.94 GiB，GPU 约 26%、显存 1304/8188 MiB；
- TERLA repeat4 的 launcher 已准备，但冻结硬门要求可用 RAM >=12 GiB，当前不启动；
- RSMBRL 五个 seed4000 partial 虽逐 SHA 可恢复，但任务书要求 beta 由 validation 选择，而旧 bundle
  声明 `paper_fixed_no_validation_tuning`；旧 partial 不得从 seed4001 继续成为正式结果。已派发
  validation-only beta selection 与 fresh-test fail-closed 修复；
- backup tool 的 verified-dependency 扩展候选 commit 为
  `e52f1deee4392774df0ff0c68b689c67d796cb92`，固定 Python 3.11 测试 `33/33 PASS`（1 skip），
  仍等待独立安全审查，审查通过前不合入、不对真实外部资产执行 dependency-create；
- 原 heartbeat `cc4-lwm-repeat3` 已在 repeat4 完成和备份后暂停；启动下一项正式实验时只更新并
  恢复这一个 heartbeat，不创建第二个监控。

### 9.4 当前下一步顺序

1. 等待并审核 `1498e4a...` 独立审查；通过后 cherry-pick 到已释放且 clean 的 3cf0 worktree；
2. 用新 HEAD 重新生成、预检并主窗口审核 repeat5 launcher，fresh train-only 启动；
3. 同时等待并审核 `e52f1dee...`，通过后备份 Table3 八资产及 PriorRL blocked candidate；
4. 审核 RSMBRL beta-validation 实现；先只跑 validation 并冻结 beta，随后 fresh test，旧 partial 禁用；
5. 所有新有用产出先不可覆盖备份并 verify，再进入下一阶段。
