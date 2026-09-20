# CC4 v3 阶段交接：2026-09-20 15:40（Asia/Shanghai）

> 本文按 `USER_MANDATORY_REQUIREMENTS.md` 第 11 条生成，用于上下文压缩、应用重启或
> 新主窗口接管。本文记录的是时间点快照；PID、进度、资源和 Git dirty 状态必须实时复核。
> 本文不能替代总任务书、冻结 manifest 或 eligibility gate。

## 1. 权威入口与已核对身份

本阶段主窗口已完整重读以下文件：

| 文件 | SHA256 |
|---|---|
| `docs/USER_MANDATORY_REQUIREMENTS.md` | `6beb6097f0b0fd1ec5bef9d51af2dcb19ae121306f7af5da54f6a1dcbba3a8ca` |
| `CC4_V3_0917_FINAL_EXPERIMENT_TASKBOOK.md` | `6d862e9475c92a668358f69d9e09e9b32b7d23cd83679dcd4bd0d3847eae00a5` |
| `docs/CHATGPT_HANDOFF_FINAL_EXPERIMENTS.md` | `9b8b146fc24c4ceefeb8b925439535bb3fc6b239828274403178cd557af2c75e` |
| `docs/FINAL_REWARD_REPLAY_MANIFEST.json` | `b7aae1a9189afad9717be7ebfff97ef2016bb25634ae8e8c83c254987653f8ee` |
| `docs/FINAL_REWARD_MODEL_MANIFEST.json` | `0a8a2c5c196dee74eda94cca24e4904cf93badf9008bf6ddcc27e6b854503931` |

`USER_MANDATORY_REQUIREMENTS.md` 现含编号 `0..11` 共 12 条强制要求。新增第 11 条要求在
上下文压缩前生成带时间戳且不覆盖旧交接的 Markdown，并在压缩后先读交接再实时复核。

## 2. 最终任务

在 `D:\paper\github-me\aptdetect` 完成第三版论文可正式使用、可重算、可审计的
CC4/D27/A4 实验，而不是 smoke、pilot、旧 reward 或单 seed 数字：

- Table 1：UAMCTS、RSMBRL、CARL、DCA、PriorRL、TERLA、LWM-RL；
- Table 2：RL-Only、LLM-RL、WM-RL、LWM-RL，四者独立训练；
- Table 3：Delay-Only、Fail-Only、Full-Reward，Full-Reward 只读复用 Table 2 LWM-RL；
- 生成任务书要求的正式图，只能消费 eligibility PASS 的完整结果；
- 默认正式口径为 5 repeats × 100 test episodes × 500 ticks，test seeds `4000..4099`；
- 优先保证 LWM-RL 的正确性、效果、完整消融和可复现性；基线只需正确、公平、无泄漏、
  可审计，不额外调参、扩大预算或挑选更优重跑。

截至本快照，严格 paper-table 完成度仍为 14 个展示行中的 1 行：DCA 已完成并 PASS。
其他训练依赖和 partial 产物不能冒充 paper-eligible 表格行。

## 3. 本阶段已完成工作

### 3.1 强制要求与恢复入口

- 新建并完整校验 `docs/USER_MANDATORY_REQUIREMENTS.md`；无行尾空白，12 个编号完整。
- 新增强制要求 11，并建立本文作为本阶段首次带时间戳交接。
- 10 分钟只读自动核验 `cc4-2` 已更新为 Luna high，当前只监控 TERLA、LWM 门禁/启动，
  正常无变化保持安静；不得启动、停止、修复、复制或清理实验。

### 3.2 已恢复并核验的公共资产

- train replay：43,297 transitions，SHA
  `ba608ed7bd63d6a6c28a259738ca35e6122bc2f0a6cf7f427df1f2b3252c8cee`；
- validation replay：10,796 transitions，SHA
  `7e23294678167078704c9603932b014d893fbd35e06b5a145259aef16c23588d`；
- absolute WM：`3b86593aa8adda3e0e173bfb700c543bd88641d23cf3e73ddc3992284c8f900f`；
- Full-Reward predictor：
  `f333330510b5de8e78fb9e2dc4287dd87fee29695fe38cbec5e60700ee615624`；
- prior cache：2,006 train-only entries，0 runtime provider calls；
- 重建 prototype：41,079/41,079 train unique、10,515/10,515 validation、6/6 supplement；
- UAMCTS progress ensemble：5 members，validation RMSE `0.091226`，constant `0.128235`。

### 3.3 DCA

- canonical 路径：`outputs/formal_v3/table1/dca_cc4`；
- 五个 repeats 均完成 100 × 500 test episodes，并通过正式 validator/eligibility；
- DCA 是目前唯一可进入第三版表格的完整行。

### 3.4 TERLA 51001

- 代码工作树：`C:\Users\25453\.codex\worktrees\a553\aptdetect`；
- HEAD：`2eb9ba7502182c3971cf880f46b14fbe76e60f37`，本快照 clean；
- 成功恢复并完成 policy seed 51001：32 train seeds、validation candidates 8/16/24/32、
  final checkpoint/optimizer/training manifest，未用 test seeds，退出码 0；
- canonical 恢复输出：
  `outputs/rebuild_20260920/terla_a4_resume_51001_attempt2`；
- 该产物是训练依赖，不是完成的 Table 1 测试行。

### 3.5 CARL 运行时修复

- 工作树：`D:\w\carl`；
- 新提交：`7bc76bc6e23ed7acb62ac7c3ec1c291b8dc8b90e`；
- 仅修复 `formal_training.py` 对 binary Git diff 再次 `.encode()` 的错误，并新增回归测试；
- 48/48 定向测试 PASS，真实只读 dry-run PASS，提交后 clean；
- train/validation/WM SHA、split 隔离、H=4 `ADAPTED_TRUNCATED`、8 synthetic/real 和
  test/provider 禁止项均通过；
- 未启动 CARL 正式训练。资源门检查时可用内存仅 5.12 GiB，低于 12 GiB 安全阈值，
  因此 fail closed，未创建输出、未消耗 seed。

### 3.6 备份核验

- TERLA 51001 旧格式不可覆盖备份：
  `C:\aptdetect_experiment_backups\20260920T071646Z__verified_dependency__terla_a4_resume_51001_attempt2__9d357539.finalized`；
- schema 为 `aptdetect_verified_dependency_backup_v1`，因此新版通用备份工具会报告“不支持的
  manifest schema”，这不是 payload 损坏；
- 已按创建脚本原算法重算：manifest sidecar PASS、10 个 payload 文件逐文件 SHA/长度 PASS、
  35,195,618 bytes PASS、snapshot digest
  `798a67b5301b2dd4bc14b58685cbb28af3bee69fa09df26aefa061161ec1973c` PASS；
- 禁止删除或覆盖该备份。后续可在 TERLA 队列停止后再创建新版 schema 备份，但不能在活动
  输出根变化时强制快照。
- 冻结 normalizer canonical 备份：
  `C:\aptdetect_experiment_backups\20260920T074824Z__verified_dependency__calibration__e0baa56f.finalized`；
  2 files、11,726 bytes，snapshot
  `6de5048af5860b940ce98ecaa75b5522baea3b485748d57840342a54d6df94c1`，manifest
  `77cb088667427f5f9c948274c5801e7c9fbeeef6caa7742e6e3d5fe8935b0a72`，逐文件复核 PASS。
- TERLA 51001+51002 联合快照：
  `C:\aptdetect_experiment_backups\20260920T075243Z__verified_dependency__terla_a4_resume_51001_attempt2__40a78c1a.finalized`；
  19 files、57,864,699 bytes，snapshot
  `d2fb0ef789c07602e2a839b3f0712b6f7aac67e279a9d86af84d438e98014535`。
- LWM 错误路径 preflight-only 证据备份：
  `C:\aptdetect_experiment_backups\20260920T075152Z__verified_dependency__repeat_1__4b9abbc2.finalized`；
  3 files、34,560 bytes，snapshot
  `2f4ca6a90043607b1023bf42b14512d258366ed71d272bf3b503fc3664300558`。原证据已由主窗口
  非覆盖移动到 `outputs/rebuild_20260920/lwm_repeat1_preflight_blocked_20260920T154905`。

## 4. 本快照正在运行或执行中的任务

### 4.1 TERLA policy 51003

- 51002 已于约 15:48 完成：32 train seeds、4 个 validation milestones、final checkpoint、
  optimizer、training manifest，test unused；checkpoint SHA
  `b32ff35c11b78a6ca8257b4afce7a64ee62cdad6593a1b4bf76f1fdca95c54f5`；
- 51002 已包含在上节 51001+51002 联合备份中；
- 51003 已 fresh 启动，唯一 wrapper/child 起始 PID：`29196 / 30752`；实际 PID 接管后必须复核；
- 输出根：`outputs/rebuild_20260920/terla_a4_resume_51001_attempt2`；
- 早期启动核验已见 CPU 增长、fresh `policy_51003.decisions.jsonl` 和仅 Gym/lz4 warning；
- 无重复 TERLA writer、无 test `4000..4099`、无 provider 调用；
- 51003 完成后必须审核并备份，再严格串行启动 51004；不得并行 TERLA policies。

### 4.2 LWM-RL repeat 1 train-only

- prototype 8 文件已复制到 canonical 路径且逐文件 SHA 一致；41,079/41,079 train、
  10,515/10,515 validation、6/6 supplement、test leak 0；
- normalizer+sidecar 已确定性重建：PT SHA
  `fefa9042586c6858f4e56a3728c5c2a5eea310e2916553a6b0ea7504c799ba4c`，sidecar SHA
  `c0db1c4cef39abd9db28bcb6c6e18ebc365cbaf3b3968f8b4e29bf3150cc0d49`；
- 四个信任锚/测试文件已由主窗口审查并提交，工作树 HEAD
  `f14472dd7f467efe0e74970e34ef1942d8f11213`，clean；
- 固定 `.venv_cc4` 中 46 项精确测试 PASS；提交后 LWM dry preflight exit 0/PASS；
- 首次 execute 因错误传入主仓库绝对 replay 路径被 canonical path gate 安全拒绝，exit 2，
  零训练 seed；证据已备份并归档，未删除；
- 现已使用工作树 canonical replay 路径 fresh 启动唯一 repeat 1 train-only，可见 launcher/python
  起始 PID `30060 / 28744`；输出为 `outputs/formal_v3/table2/lwm_rl/repeat_1`；
- 只允许 `preflight + train + stop-after=train`，完成应退出 3、STOPPED/PARTIAL，禁止
  validation/test/episode/eligible manifest；实际 PID 与当前 seed 必须接管后复核。

### 4.3 Table 3 reward predictors

- Luna xhigh 子任务已完成输入/输出资源门审查，但为 LWM GPU 优先级主动暂停，尚未运行测试
  或启动 Delay-Only、Fail-Only；目标目录仍不存在；
- 代码工作树：`D:\w\priorrl`，基线 HEAD
  `f89b43f16bf57db281d626f7abd9dadef4fb9fb5`；
- 输出目标：`outputs/rebuild_20260920/table3_reward_models/{delay_only,fail_only}`；
- Delay-Only 必须是标准化标签 MSE；Fail-Only 只允许未加权 SmoothL1/Huber beta=1.0；
  固定 50 epochs、无 validation checkpoint selection、无权重/过采样/重跑挑优；
- 两个模型严格串行，GPU 优先；完成后主窗口需审核 quality gate、SHA、manifest 并备份，
  未审核前不得启动 Table 3 PPO。

## 5. Git 状态快照

- 主仓库 HEAD `f89b43f16bf57db281d626f7abd9dadef4fb9fb5`；存在四个预期 untracked：
  `docs/EXPERIMENT_BACKUP_PROTOCOL.md`、`docs/USER_MANDATORY_REQUIREMENTS.md`、
  `formal_experiments/evaluation/backup_outputs.py`、`tests/test_backup_outputs.py`。
- LWM 工作树 HEAD `f14472dd7f467efe0e74970e34ef1942d8f11213`，四个已审核信任锚修改已提交，clean。
- TERLA 工作树 HEAD `2eb9ba7502182c3971cf880f46b14fbe76e60f37`，本快照 clean。
- CARL 工作树 HEAD `7bc76bc6e23ed7acb62ac7c3ec1c291b8dc8b90e`，clean。
- PriorRL/Table3 工作树 HEAD `f89b43f16bf57db281d626f7abd9dadef4fb9fb5`，任务启动前 clean。
- RSMBRL 工作树 `D:\w\rsmbrl` HEAD
  `da9b096efea2af621732a3a4d86239d875e5fbba`，本快照 clean；五 repeats 仅各保留
  episode seed 4000 的 durable partial，当前无 RSMBRL 进程。

## 6. 当前阻塞与禁止误用

1. LWM-RL repeat 1 已通过依赖门并启动；任何异常不得自动重跑，先保留并审核 journal、curve、
   checkpoint 和 launcher 证据。
2. CARL 在可用内存未达到 12 GiB 时不得与 TERLA 并行启动；不得降低门槛制造 OOM 重跑。
3. RSMBRL 五个 repeats 先前各只有 1/100 episode；不可称为完成，也不要在 TERLA 占用大量
   内存时五路并发恢复。
4. UAMCTS 仍需 prototype/entropy/normalizer 完整身份链后才能正式运行。
5. 旧 v2、smoke、pilot、provisional、单 seed、短 ticks 和旧 reward 结果均禁止填表。
6. CARL H=4 是公开披露的 `ADAPTED_TRUNCATED` 训练边界；不得声称论文原 H=256 已验证。
7. `formal_v3/training/terla_a4` 的旧失败/partial 与当前 `attempt2` 不是同一 canonical 运行，
   禁止混合或覆盖。

## 7. 接管后的精确顺序与验收

1. 先读强制要求、任务书、本文和两个 final manifest；再实时核验 PID、Git 和输出 mtime。
2. 10 分钟核验 LWM repeat 1：确认单 writer、GPU增长、seed `1000..1031` 连续、journal/curve/
   checkpoints 同数、provider/online steps 0、无 validation/test；完成后要求预期退出码 3。
3. LWM repeat 1 完成后先独立审核和不可覆盖备份，再启动 repeat 2；不得自动越过审核。
4. 审核 Table 3 两个 predictor 的 manifest、checkpoint SHA、H1/H4 gate；每个有用结果建立
   不可覆盖、逐文件 SHA 备份；任一 gate 失败不调参、不覆盖重跑。
5. TERLA 51003 完成后核验 32 train seeds、8/16/24/32 candidates、final checkpoint、optimizer、
   training manifest、test unused、exit 0；先备份，再启动 51004。
6. TERLA 释放内存且 available RAM ≥12 GiB 后，按 4 CPU 线程启动 CARL repeat 1；仅一个 writer，
   先审核 repeat 1，再决定 2–5。
7. LWM GPU 训练优先；RSMBRL、PriorRL、UAMCTS 的恢复/训练只能在资源和身份门允许时并行。
8. 每个训练完成后必须先主窗口审查、备份，再启动 test；每个方法五个训练 checkpoint 完整后
   才运行共享 100 × 500 test suite。

## 8. 活跃代理与监控

- `/root/terla_remaining_queue`：Luna xhigh，当前严格串行执行 TERLA 51003；10 分钟核验一次；
- `/root/lwm_dependency_gate_repair`：Luna xhigh，本轮已转为 LWM repeat 1 唯一启动执行代理；
- `/root/remaining_gate_status`：Table 3 predictor 任务已暂停，让出 GPU/主存；
- 自动化 `cc4-2`：Luna high，每 10 分钟只读核验 TERLA 51003 与 LWM repeat 1；正常静默，
  完成/异常才通知；
- CARL 门检代理已完成并退出；因内存门未启动正式训练。

任何代理都不得删除、reset、clean、restore、覆盖或自行降低门禁。完成或异常只向主窗口报告，
由主窗口审查、备份并安排下一步。

## 9. 2026-09-20 17:21 接管续写

本节覆盖上文中已经过时的运行中状态；上文仍保留作为审计历史。

### 9.1 LWM-RL

- Table 2 `lwm_rl/repeat_1` 已按预期完成 train-only safe stop：exit code `3`，
  `STOPPED/PARTIAL`，`expected_safe_stop=true`，`formal_result_eligible=false`；
  journal/curve/checkpoints 为 `32/32/32`，train seeds `1000..1031` 连续无重复，
  逐 checkpoint SHA 一致，provider calls 和 online environment steps 均为 `0`，没有
  validation/test/episode/eligible manifest。
- repeat 1 已建立新版不可覆盖备份并再次 verify PASS：
  `C:\aptdetect_experiment_backups\repeat_1_20260920T090537.829235Z_508564ad297c41b18c277eeb8158c158.finalized`；
  40 files、16,076,431 bytes，backup manifest SHA
  `22a7331bb5652745f09259616989855820385ebdf140890a3e19d2e220d8f4d6`。
- repeat 2 已 fresh 启动：工作树 HEAD
  `f14472dd7f467efe0e74970e34ef1942d8f11213` 且 clean，wrapper/launcher/writer 起始
  PID `32496/23988/17708`；输出
  `outputs/formal_v3/table2/lwm_rl/repeat_2`；身份为 repeat `2`、policy seed `51002`、
  32 train seeds、500 ticks、`stop-after=train`、无 `--resume`。
- repeat 2 首检至少推进到 `2/2/2`，首 checkpoint SHA
  `a0c51a85b69b2522509a2be7551520d0687f6b86c5a7f3f75324f3632f243604` 重算匹配；
  provider/online steps 为 `0`，没有越权阶段。后续只由 10 分钟 heartbeat 核验。

### 9.2 TERLA-A4

- policy seeds `51001..51003` 均已完成 32 train seeds、4 个合法 validation milestones、
  500 ticks，未使用 test seeds；51003 已包含在不可覆盖备份
  `C:\aptdetect_experiment_backups\20260920T084251Z__verified_dependency__terla_a4_resume_51001_attempt2__9c02e748.finalized`。
- 51004 的第一次可见 launcher 在 Python spawn 前退出；未生成任何 51004 文件或目录，
  TERLA writer 为 `0`，因此这不是一次实验运行，也不需要清理或恢复。
- LWM repeat 2 启动后可用主存约 3.64 GiB，低于并行安全门；51004 保持暂停。不得把
  launcher 失败当作训练失败或自动重跑，资源恢复后由主窗口重新授权。

### 9.3 Table 3 与世界模型恢复

- Delay-Only predictor checkpoint/manifest SHA：
  `192efbe6512fbd83a746d86d759ebe5e68b1437ee78e570747a119d74b0e683f` /
  `fb9fdd64a6f576d0ffb444daacce07e891ba5505da66381b57acb52631c71eaf`；
  Fail-Only：
  `8800a65d4cd4f540ca6d5fa5bcc0f7e87a913403c4cded5beed931e91bb0b3b2` /
  `8295b5177e900f032f7fed142b07cb7773607911d77d73be01b414aebf61f5cc`。
  两者 quality gate PASS，预测器及日志已分别建立不可覆盖备份；它们仍只是训练依赖。
- replay manifest 的 `incident_response.py` 锚点已证明是同一 Git blob 的 Windows 换行字节
  问题；主仓库精确 raw SHA
  `f717ecf5eb118255858749dc095f385edb14c5229d77d47b315a0fdea80621ad` 已恢复到
  `D:\w\priorrl`，无需重跑 replay。Table 3 两阶段 provenance 正在生成和审核。
- `a4_5b_report.json` 已从相同 replay/命令确定性恢复；canonical SHA
  `9096ea8deffedd4b9b430d2a1cc9b290c47ea053e3133804c5ebaba29078c193`，absolute/delta
  checkpoint SHA 也与 final manifest 一致。final WM 目录和恢复证据两份备份均已人工复核
  文件数、字节数、逐文件 SHA 和 manifest sidecar，全部 PASS。

### 9.4 UAMCTS、备份工具与 Git

- UAMCTS 信任链 fail-closed 补丁已收窄为仅影响 `uamcts_cc4`，固定 Python 3.11 环境
  `38/38` 定向测试 PASS，提交 `9b953f6e`；其他方法保持原 runner/resume/manifest 语义。
- UAMCTS 仍为预期 NO-GO：progress、prior-entropy、normalizer 三类 sidecar 尚未完整，且旧
  entropy 与当前 prototype 绑定不一致；禁止启动正式 episode。
- 备份工具已增加对 train-only `orchestrator_report.json + exit_code 3` exact safe-stop 的
  fail-closed 支持，`15/15` 测试 PASS；只有 exact STOPPED/PARTIAL/non-eligible 合同可备份。
- 主仓库当前 HEAD 为 `9b953f6e`；要求文档、备份协议、备份工具和本交接文档将在主窗口
  审核后单独提交，不得与实验输出混合。

### 9.5 单窗口监控

- 自动化 `cc4-2` 已从每次创建独立任务的 cron 改为附着当前主任务
  `01a0b807-955b-7ce1-a36b-79594f6a1674` 的单一 10 分钟 heartbeat；正常状态保持安静。
- 已归档此前 32 个 `CC4 启动与运行核验` / `CC4 启动门禁核验子代理` 窗口；归档不删除
  历史。后续不得再为每次轮询创建新窗口。
