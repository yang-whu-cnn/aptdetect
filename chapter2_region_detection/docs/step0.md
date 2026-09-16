# Step 0 — Experimental Branch Isolation 审核记录

> 项目：UG-CEM-APT  
> 分支：`ug-cem-apt`  
> 阶段：Step 0 — experimental branch  
> 审核日期：2026-09-16  
> 最终状态：**PASS / 已完成**

---

## 1. 本阶段目标

Step 0 的唯一目标是把 UG-CEM / CEM 对比复现工作从稳定实现中隔离出来，避免实验性改动污染稳定备份分支。

正式分支约定：

- 稳定备份：`me`
- 实验实现：`ug-cem-apt`

本阶段不实现 CEM、uncertainty、world model、CC4 adapter、PPO 或 LLM。

## 2. 分支核对

当前 Git 历史核对结果：

- `me` 当前基准 commit：`b79cbeeedcda4ac4677b1cf15526a91a7906aede`
- `ug-cem-apt` 与 `me` 的 merge base 为同一 commit；
- 审核时确认 `ug-cem-apt` 相对 `me` 为单向 ahead、behind 0；后续文档提交会继续增加 ahead commit 数，因此不把具体 ahead 数作为冻结验收值；
- 所有 UG-CEM / Gate A 新增实现均保留在实验分支。

这说明实验分支是在稳定备份之上单向前进，没有要求把实验性代码反向写入 `me`。

## 3. 隔离原则

后续正式复现遵守：

- 不在 `me` 上直接开发 UG-CEM baseline；
- 新 baseline 放在 `chapter2_region_detection/baselines/`；
- 公平比较共享接口放在 `chapter2_region_detection/shared/`；
- Gate A 的正式数据、验证和训练代码放在 `chapter2_region_detection/formal_experiments/`；
- 单元测试放在 `chapter2_region_detection/tests/`；
- 阶段审核记录统一放在 `chapter2_region_detection/docs/`。

## 4. 验收

- 实验分支存在：PASS；
- 稳定分支与实验分支职责分离：PASS；
- 实验分支未落后于 `me`：PASS；
- 后续实现目录具备独立承载位置：PASS。

## 5. 最终审核结论

```text
Step 0 — Experimental Branch Isolation

Branch isolation : PASS
Stable backup    : PASS
Experiment scope : PASS

FINAL STATUS: PASS
```

Step 0 正式关闭。