# Table 1 对比基线快速合规产出计划

日期：2026-09-21
最终目标：尽快完成第三版论文 Table 1 的 7 行 `5 repeats × 100 episodes × 500 ticks`
正式结果，同时把 GPU、调试和调优资源优先留给自有 LWM-RL。对比方法只要求正确、公平、
无明显实现错误和可审计，不为提高其效果反复调参或挑优重跑。

## 1. 当前状态

- `DCA-CC4 (adapted)`：5/5 repeats、每 repeat 100×500、eligibility PASS；已完成，冻结，
  禁止覆盖或重跑。
- `UAMCTS-CC4 (adapted)`：算法代码、H4 WM、progress ensemble、offline prior、三源 uncertainty、
  calibration normalizer 和 formal evaluator 已存在；14 个重建资产及 sidecar 完整，尚欠当前
  commit 的直接 preflight 和 5 个正式 repeats。它不需要训练 policy，是最快可新增完整表格行。
- `TERLA-A4`：结构复现和 formal training 已存在；当前 canonical 只有 policy seed 51001 的
  train-only checkpoint/decision 资产，尚无完整五策略与正式 test 行。
- `RSMBRL-CC4`：categorical uncertainty-guided CEM 与 formal evaluator 已存在；需先以
  validation 冻结 beta，再执行五个正式 repeats；规划计算量较大。
- `PriorRL-PPO-CC4`：PPO-KL、prototype prior、alpha validation、五 repeat 训练代码已存在；
  必须先完成 frozen prototype/provenance 和 alpha-selection 的当前 commit 预检，不能使用旧
  blocked candidate 直接测试。
- `CARL-CC4 (adapted)`：SCM、8× synthetic rollout、PPO、formal training/evaluator 已存在；
  需完成五个训练 checkpoint 后正式测试，资源成本高于 UAMCTS/TERLA。

严格论文完成度因此仍为 Table 1 的 `1/7`，全三表显示行 `1/14=7.14%`。代码存在、pilot、
train-only 或 dependency PASS 不能计作表格行完成。

## 2. 执行优先级

1. 立即完成 UAMCTS asset staging + preflight；PASS 后启动 formal repeat1，并根据首个 episode
   实测 RAM/VRAM/速度决定 1-way 或 2-way repeats。不中断自有方法实现。
2. 并行审计 TERLA seed51001 是否可在当前代码身份恢复/续接；若不兼容则 fresh 运行五个 seeds，
   不为避免重跑而降低门禁。
3. UAMCTS 运行时完成 PriorRL provenance/alpha 预检；通过后训练五 repeats。
4. RSMBRL 先 validation-only 冻结 beta，再并行 planner repeats。
5. CARL 最后运行；只做论文必要配置，不做扩大网格。

## 3. UAMCTS 即时工作包

### 3.1 最终目标关系

产出 Table 1 `UAMCTS-CC4 (adapted)` 行。方法必须保持论文核心：learned WM 上 MCTS、offline
LLM prior、progress potential shaping、三源 uncertainty、hybrid UCB、root visit count 选动作，
每次只执行首动作并重规划。CC4 适配固定 D27/A4/H4、Full-Reward、simulation budget 64。

### 3.2 冻结代码和输入

- clean worktree：`D:\w\uamctsformal`；
- HEAD：`f4878be466a672f727cc72eee197d04bbd53c96e`；
- Python：`D:\paper\github-me\aptdetect\chapter2_region_detection\.venv_cc4\Scripts\python.exe`；
- rebuilt asset root：
  `D:\paper\github-me\aptdetect\chapter2_region_detection\outputs\rebuild_20260921\uamcts_assets_v3_20260921T024147Z_3170`；
- asset manifest：`uamcts_assets_v3_sidecar_manifest_v1.json`，14 files、2,954,243 bytes；
- frozen source hashes：WM `3b86593a...900f`、reward `f3333305...624`、prototype
  `e4359df9...d09`、progress `0fb1fd62...cc2`、prior entropy `9503b8b0...24c0`、
  normalizer `6410fb92...817`；sidecar hashes以 asset manifest 为准；
- train/validation replay SHA：`ba608ed7...8cee` / `7e232946...588d`。

### 3.3 非覆盖 staging

formal runner 从自身 clean worktree 的 canonical relative paths 加载依赖。执行代理只能在目标
文件全部不存在时，复制下列 immutable source 到新目录；任何目标已存在即停止上报，不覆盖：

- `sources/world_model_absolute.pt` ->
  `outputs/world_model_final_20260917/a4_5b/world_model_absolute.pt`；
- `sources/response_reward_predictor.pt` ->
  `outputs/world_model_final_20260917/a4_5c/response_reward_predictor.pt`；
- `sources/frozen_prototypes.json`、coverage、provenance ->
  `outputs/priorrl_cc4/prototypes/`；
- `progress/*` -> `outputs/uamcts_cc4/progress/`；
- `calibration/*` -> `outputs/uamcts_cc4/calibration/`；
- main canonical final replay `train.jsonl`/`validation.jsonl` ->
  `outputs/formal_replay_final_20260917/`。

复制后必须逐文件重算 SHA，并与 rebuild manifest/final replay manifest 完全一致。源目录保持只读，
不移动、不删除。不得使用 junction/symlink/reparse，避免 formal safety gate 拒绝。

### 3.4 preflight

资源硬门：可用 RAM >=4.5 GiB、worktree clean、无同目标 writer。只运行：

```powershell
& 'D:\paper\github-me\aptdetect\chapter2_region_detection\.venv_cc4\Scripts\python.exe' `
  -m baselines.uamcts_cc4.preflight `
  --report outputs/uamcts_cc4/preflight_f4878be4.json
```

工作目录必须是 `D:\w\uamctsformal\chapter2_region_detection`。preflight 必须：eligible=true、
errors=[]、provider_calls=0、test seeds未使用、全部 replay/model/progress/prior/normalizer/sidecar SHA
匹配。preflight 不启动 episode，不创建 formal result。

### 3.5 formal repeat 设计

只有主窗口复核 preflight 后，另派 Luna xhigh 启动 formal repeat。第一项固定为：

- method=`uamcts_cc4`；repeat=1；policy seed=51001；test seeds 4000..4099；ticks=500；
  device=`cuda`；simulation budget=64；
- canonical output：
  `D:\paper\github-me\aptdetect\chapter2_region_detection\outputs\formal_v3\table1\uamcts_cc4\repeat_1`；
- output 必须事前完全不存在；runner 的 per-episode resume 只在相同 code/config/artifact/seed/hash
  identity 下允许；
- 单 episode 顺序执行；不同 repeat 仅在首个 episode 资源审查后安全并行；
- 固定 Luna-high 监控窗口每15分钟只读检查唯一 writer、episode parts增长、SHA/identity、异常、
  完成；正常静默。

repeat 完成验收：100 个 seeds `4000..4099` 各一行、500 ticks 全满、decisions/raw metrics 完整、
provider calls=0、测试前后冻结参数 hash一致、manifest/validation/eligibility全部 PASS。之后立即建立
不可覆盖备份并逐 SHA verify，再启动/继续其余 repeats。任何失败保留 episode resume 证据，禁止
删除或从头盲目重跑。

## 4. 其余基线的实现与验收边界

### TERLA

两层 HGT、global sum pooling、单 shared policy/五 agent 独立 context、A4、论文 cyber reward；
policy 不得读取 hidden truth。先审计 seed51001 的 commit/schema/reward/32 seeds/checkpoint SHA；
兼容才恢复，否则 fresh。五 policy seeds训练完成后，用统一 100×500 evaluator。

### PriorRL

D27 -> PPO A4，loss 为环境 PPO objective + `alpha_KL*KL(pi||p_LLM)`；禁止 WM。alpha 只在
train/validation 小网格冻结，test 不参与；prototype miss/超 radius/provider call 均 fail closed。

### RSMBRL

共享 frozen H4 WM/Full predictor，categorical CEM、uncertainty penalty、执行首动作。beta 只用
validation 选择并冻结，test 期间 normalizer 不更新；不把 beta=0 diagnostic 作为正式行。

### CARL

必须是 Dyna/CAICS：真实+8条 synthetic rollout 更新 PPO，不得实现成决策时 planner；明确
ADAPTED_TRUNCATED H4，SCM/reward mapping只用于训练，不泄漏到 policy observation。

所有基线共同验收：clean commit、论文/配置/provenance SHA、5 repeats、100×500、固定 test seeds、
raw episodes/decisions、四指标可重算、test冻结 hash、eligibility PASS、不可覆盖备份。不得为提升
对比方法效果额外调参、加预算或挑选更优重跑。
