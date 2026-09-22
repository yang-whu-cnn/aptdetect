# CC4 v3 阶段交接：2026-09-21 17:05（Asia/Shanghai）

> 本文按 `USER_MANDATORY_REQUIREMENTS.md` 第 11 条新建，不覆盖历史交接。PID、资源和
> 进度仅是时间点快照；接管后必须重新核验。本文不把 dependency、partial、单 episode 或
> train-only 资产提升为 paper result。

## 1. 最终目标与正式口径

最终交付是第三版论文 Table 1、Table 2、Table 3 及正式图。默认正式协议保持为 CC4 / D27 /
A4、五个独立 policy repeats、每 repeat 测试 seeds `4000..4099`、每 episode 500 ticks，且需完整
raw artifacts、manifest、冻结身份、可重算 SHA 和 `eligibility_report.passed=true`。

- Table 1：UAMCTS、RSMBRL、CARL、DCA、PriorRL、TERLA、LWM-RL；
- Table 2：RL-Only、LLM-RL、WM-RL、LWM-RL，分别初始化和训练；
- Table 3：Delay-Only、Fail-Only、Full-Reward，其中 Full-Reward 只读复用 Table 2 LWM-RL；
- 自有 LLM prior + world model + PPO 的 LWM-RL 效果与完整消融优先；baseline 只做冻结的正确、
  公平、无泄漏、可审计实现，不为提高 baseline 效果扩大调参或挑优重跑。

严格 paper-table 完成度仍为 `1/14 = 7.14%`：仅 DCA-CC4 的 `5×100×500` 已证明 PASS。

## 2. 本次重新读取的权威文件

| 文件 | SHA256 |
|---|---|
| `chapter2_region_detection/docs/USER_MANDATORY_REQUIREMENTS.md` | `04e1319127cb47c4893842c8179ef87bb52d023c33e16cb09d800c5d07c5ccb7` |
| `CC4_V3_0917_FINAL_EXPERIMENT_TASKBOOK.md` | `6d862e9475c92a668358f69d9e09e9b32b7d23cd83679dcd4bd0d3847eae00a5` |
| `chapter2_region_detection/docs/CHATGPT_HANDOFF_FINAL_EXPERIMENTS.md` | `9b8b146fc24c4ceefeb8b925439535bb3fc6b239828274403178cd557af2c75e` |
| `chapter2_region_detection/docs/FINAL_REWARD_REPLAY_MANIFEST.json` | `b7aae1a9189afad9717be7ebfff97ef2016bb25634ae8e8c83c254987653f8ee` |
| `chapter2_region_detection/docs/FINAL_REWARD_MODEL_MANIFEST.json` | `0a8a2c5c196dee74eda94cca24e4904cf93badf9008bf6ddcc27e6b854503931` |

冻结 replay：train 32 seeds / 43,297 transitions，validation 8 seeds / 10,796 transitions。
冻结 absolute WM SHA `3b86593aa8adda3e0e173bfb700c543bd88641d23cf3e73ddc3992284c8f900f`，
Full-Reward predictor SHA `f333330510b5de8e78fb9e2dc4287dd87fee29695fe38cbec5e60700ee615624`。

## 3. 当前正式运行与资产

### 3.1 UAMCTS Table 1 repeat 1

- worktree `D:\w\uamctsformal`，HEAD `f620fc39698e1c922657fb557db3dacdfa867579`，17:02 核验 clean；
- canonical output `chapter2_region_detection/outputs/formal_v3/table1/uamcts_cc4/repeat_1`；
- launcher/Python 链历史身份 `12048 -> 33272 -> 32844`，当前 `32844` 仍存活；
- repeat 1 / policy seed 51001 / test seeds 4000..4099 / 500 ticks / CUDA / provider calls 0；
- 截至 17:02 共完成 4 个 durable episode（4000..4003），最近写入
  `2026-09-21T08:51:29Z`；单 episode 约 55--68 分钟；
- GPU 快照约 36%，1646/8188 MiB；Python `32844` working set 约 587 MiB；
- 尚未完成 repeat，不得称为论文结果；不得由接管者停止、重启或混入新 commit 产物。

性能审计发现当前 `planner.py` 每个 64-simulation MCTS 决策串行执行；更关键的是同一已展开
`node/action` 被再次访问时仍重复调用 WM ensemble、uncertainty 和 reward 推理，尽管 child 已固定。
这是可以在不改变搜索预算、H=4、RNG、树更新和测试协议的前提下做缓存等价优化的主要候选。
任何优化必须先在非 test seed 上证明 serial/optimized action、visits、Q、raw episode/metric 等价，
再用新 commit 建立 fresh 正式 lineage；不能直接把当前四个 episode 与新 commit 混合。

### 3.2 RSMBRL

- worktree `D:\w\rsmbrl_v32`，HEAD `c9e0f57137274669da05e8dcd3d54f2c1e12bbd5`，clean；
- beta validation 已完成 56/56（7 beta × 8 validation seeds × 500），选择 `beta=1.0`；
- `beta_selection_report.json` SHA `49361599e17726abf4a39d4338f9645f7b82e3f90307d4a8464781409041ce5a`，
  built-in validation `valid=true`、`eligible_to_freeze=true`、无 test seed；
- 不可覆盖备份已完成：
  `C:\aptdetect_experiment_backups\rsmbrl_beta_selection_20260921_085308Z.finalized`；
- Luna xhigh 子任务正在执行备份复核与 formal repeat 1 启动门；17:03 出现新的 Python 父/子进程
  `18016/19376`，但本文写入时尚未获得该子任务的命令身份与首个输出确认，因此只能记为
  “可能正在启动/核验”，不得据此宣称 formal writer 已验收。

### 3.3 PriorRL

- worktree `D:\w\priorrl`，HEAD `025b5bb1852c6e58bb380e80afa75e3eeb1f072b`；
- 当前由 Luna xhigh 修改 `formal_training.py` 和 `test_priorrl_formal_training.py`，尚未提交；
- 目标是 checkpoint schema v2 的严格 resume 身份（code/config/prototype/coverage/provenance、
  seed prefix、RNG/optimizer、KL 方向），旧 19-seed partial 已单独备份且禁止复用；
- fresh 目标 `outputs/formal_v3/training/priorrl_ppo_cc4/alpha_selection_v2` 当前不存在；
- 本轮只做代码与测试，禁止自行启动；主窗口审核 commit 后另派 Luna xhigh 正式运行。

### 3.4 TERLA / CARL / True-PPO

- TERLA worktree `C:\Users\25453\.codex\worktrees\a553\aptdetect`，HEAD
  `5840f209bc357b38d53f9bf13e748a27671b546c`；Luna xhigh 正修复 transactional pending、
  frozen-file identity、aggregate/state tamper、500-tick callback、writer lock 等 P0/P1 问题；未启动实验。
- CARL worktree `D:\w\carl`，HEAD `7bc76bc6e23ed7acb62ac7c3ec1c291b8dc8b90e`，clean；冻结 replay/WM
  staging 完成，formal target 尚未创建，受现有 12 GiB free-RAM 硬门阻塞。
- True-PPO/LWM worktree `D:\w\trueppo`，HEAD `bd62e64e3318545f0de003b080c6a61c9c06f5c0`，clean；
  代码和定向测试已完成但未独立审查/正式运行。

## 4. 为什么耗时及允许的加速

耗时主要来自正式协议的 14 个逻辑表格行以及 planner/PPO 的五次独立身份，不来自聚合 CSV。
其中 UAMCTS 当前每 tick 进行 64 simulations × H4 世界模型搜索，按当前速度单 repeat 约 100 小时，
是最大关键路径。任务书明确允许 planner 方法按 repeat/episode 并行，并要求 inference mode、批量 WM
rollout 和预加载 cache。

立即取消/禁止的非必要工作：额外 pilot/smoke、baseline 扩大调参网格、DCA 重跑、Table 3
Full-Reward 重跑、test seed 调参、重复绘图/聚合。不得取消的是五 repeats、100 test episodes、
500 ticks、独立 Table 2 训练和 eligibility gate。

安全优化设计：

1. 为 UAMCTS 缓存同一 node/action 的冻结 transition ensemble uncertainty、child 和 base reward，
   避免重复 WM/reward inference；复用 node progress score；保持首次访问 RNG 顺序不变。
2. 设计 disjoint episode-seed shards，每 shard 独立目录和 writer；merge 只接受同 commit/config/
   artifact/policy identity，验证恰好 100 个唯一 seeds、无 gap/overlap、逐文件 SHA 后再聚合。
3. 先在 train/calibration/dev seed 做 serial/optimized paired equivalence 与 wall-clock benchmark；任何 action、
   visit、Q、decision schema 或指标漂移都 fail closed。正式 test 不用于优化选择。
4. 资源允许时并行不同 repeats/shards；单个 episode 内环境仍按 tick 顺序。CPU 任务设置线程/affinity，
   避免与 GPU writer 争抢仅约 4.19 GiB 的当前可用 RAM。
5. 不停止当前 UAMCTS；先取得等价性和加速倍数，再由主窗口比较“继续旧 lineage”与“备份后 fresh
   新 lineage”的剩余工时。子代理无权作停止决定。

## 5. Git 与资源快照

- 主仓库 HEAD `fa52f8064527971cc9a2c0fe2288927f025ebb20`；dirty：用户已有
  `SESSION_HANDOFF_20260920_2230.md`、`USER_MANDATORY_REQUIREMENTS.md`，以及未跟踪
  `terla_backup_source_20260921_clone/`；不得清理、restore 或覆盖。
- UAMCTS、RSMBRL、CARL、True-PPO worktree clean；PriorRL/TERLA 的 dirty 文件属于当前 Luna
  实现任务，不得由其他代理碰触。
- 17:03 可用 RAM 约 4.19 GiB；RTX 4070 Laptop 约 36% utilization、1646/8188 MiB。

## 6. 活跃任务和监控

- `/root/rsmbrl_formal_repeat1_v42`：Luna xhigh，验证备份后启动 formal repeat 1；完成启动验收即结束，
  不长期监控、不自动 repeat 2。
- `/root/priorrl_alpha_resume_v39`：Luna xhigh，完成 v2 resume 修复、测试和 commit；不启动实验。
- `/root/terla_trainonly_stage_v38`：Luna xhigh，完成既定 P0/P1 修复与 commit；不启动实验。
- automation `cc4-15`：唯一 Luna high 风格、每 15 分钟只读 heartbeat；正常静默，完成/异常才通知；
  不占协作 agent 槽位，不启动/停止/修复实验。

## 7. 下一接管顺序与验收

1. 实时复核三个 Luna 子任务；任何已完成任务先由主窗口审查 diff、测试、commit、tracked clean，
   再释放槽位。
2. 第一空闲槽立即派 Luna xhigh `formal_sharded_acceleration`：只实现/证明 UAMCTS 等价缓存和
   formal seed-shard/merge 门；不得触碰当前 writer、test output 或启动正式运行。
3. RSMBRL repeat 1 必须先确认唯一 writer、HEAD/config/beta/artifact SHA、policy seed 51001、
   seeds4000..4099、500 ticks、CPU、首个输出增长；完成后要求 100 episodes、PASS eligibility、
   指标重算与测试前后冻结哈希一致，再备份并安排 repeats 2--5。
4. PriorRL/TERLA commit 经主窗口复核通过后，另派 Luna xhigh 启动；不把代码完成或 train-only
   completion 计为论文行。
5. 主方法优先：独立审查 `bd62e64...` 后尽快启动 LWM/Table 2/3 的正式 fresh lineage；GPU 和显存
   优先给自有方案，baseline 只使用不造成主方法 OOM/重跑的剩余资源。
6. 每个完整 milestone 建立不可覆盖备份，verify file count/bytes/manifest/逐 SHA；任何异常 fail closed。
