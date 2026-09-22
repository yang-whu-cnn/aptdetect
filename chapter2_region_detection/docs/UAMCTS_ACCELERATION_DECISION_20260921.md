# UAMCTS 正式实验提速决策（2026-09-21）

## 1. 最终目标与不可变协议

本决策服务于第三版论文 Table 1 的 `UAMCTS-CC4 (adapted)` 正式结果。最终结果仍必须满足：

- 5 个独立 repeats，policy seeds `51001..51005`；
- 每 repeat 使用 test seeds `4000..4099`，每 episode 严格 500 ticks；
- 64 simulations、H=4、冻结 WM/reward/progress/prior/normalizer；
- provider calls 为 0，测试期无在线模型更新；
- 完整 raw artifacts、逐文件 SHA、manifest、独立指标复算和
  `eligibility_report.passed=true`。

不得把正式协议永久缩减为 3 repeats。允许前三个 repeats 完成后生成明确标记为
`PROVISIONAL/PARTIAL` 的早期观察，但不得进入最终论文表格；repeats 4--5 必须继续完成。

## 2. 当前实现为什么慢

旧正式实现对五个 Blue agent 的每个决策点执行 64 次 MCTS simulation，每次最多进行 H=4
的冻结世界模型 rollout。旧 `planner.py` 在同一 tree node/action edge 被重复访问时，会重复执行
WM ensemble、uncertainty、reward 和 progress 推理。模型调用是小批次、频繁 Python 调度和
CPU/GPU 同步，因此 RTX 4070 只有约三至四成利用率，瓶颈不是显存容量，而是串行小调用、
重复推理和单 writer 的决策循环。

非测试合成基准已观测到：WM/reward 调用由 `256/256` 降到 `15/15`、progress 调用由 `522`
降到 `16`，planner microbenchmark 为约 `4.84x`；该数字不是正式真实运行倍率，必须用真实冻结
资产和 train/dev 输入再次证明。

## 3. 结果复用边界

- 当前旧 commit `f620fc39698e1c922657fb557db3dacdfa867579` 已完成的 episode 只能在同一
  commit/config/artifact identity 下继续组成旧 lineage。
- 加速代码 commit 改变后，即使数学输出被证明等价，也不能把旧 episode 与新 episode 合并，
  因为正式 preflight/manifest 绑定代码 commit。切换时必须先建立不可覆盖备份，再从 seed 4000
  fresh 创建新 lineage。
- 可以且应该复用冻结 WM、reward predictor、progress ensemble、offline prior、normalizer 和
  validation selection；这些依赖不需要重训，但必须逐 SHA 匹配。
- 旧论文、smoke、pilot、train-only、其他 commit 或其他协议的 UAMCTS 数字不能直接填入表格。

## 4. 提速实施顺序

### A. 等价缓存门

用真实冻结资产和 train/dev seed 做 legacy/cached paired benchmark。必须比较 selected action、
root visits、Q、planner RNG end-state、call counts、异常路径和 wall-clock。任何语义漂移都阻止切换。

### B. 低风险运行时优化

仅接受 `inference_mode/no_grad`、immutable tensor/device conversion 缓存、无效 copy 消除、预加载
冻结资产和不改变 RNG/运算顺序的批处理。禁止减少 simulations/H、混合精度近似、扩大 prior radius、
增加 fallback 或在线 provider。

### C. episode shard 并行

每个 shard 使用独立 fresh 输出目录和预声明、互斥的 test seed 子集。merge 必须验证同一
commit/config/policy/artifact identity、恰好 100 个唯一连续 seeds、每 episode 500 ticks、
episode/decision 成对、逐文件 SHA、provider0 和无重复 writer，然后重新聚合指标并执行正式 validator。

### D. 资源调度

当前 RAM 约 4--5 GiB 空闲时不增加第二个重量级 formal writer。资源释放后先做 2-way，并依据实测
RAM/VRAM/吞吐决定是否升到 3-way；不得因并发造成 OOM、无日志退出或重跑。GPU 优先留给主方法
LWM-RL，其次才是 UAMCTS shard。

## 5. 切换与验收

只有以下条件全部成立，主窗口才可停止旧 writer并切换：

1. 真实冻结资产等价门 PASS，稳定加速至少 `1.5x`；
2. shard writer/merge 的正向和拒绝测试全部 PASS；
3. 新 commit、worktree、冻结资产和启动命令完成独立审核；
4. 当前旧 partial 先安全停止并建立不可覆盖 SHA 备份；
5. 新 repeat 目录不存在且唯一 writer/resource 门通过；
6. 固定 heartbeat 已更新为新 lineage 的 commit、PID、目录和身份。

前三个 repeats 可先并行产出临时统计；最终 Table 1 只接纳五个 repeats 全部 PASS 后的聚合结果。
