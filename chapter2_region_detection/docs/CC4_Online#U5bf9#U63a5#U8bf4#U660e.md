# CC4 Online 对接说明（最小接口）

本工程的 PPO 属于**在线强化学习**：需要“动作 → 环境反馈”的因果闭环。因此若要产出论文可用结果，最终建议在 CC4 平台使用 `mode: online` 进行交互训练/评估。

---

## 1. 需要提供的最小能力（两类API）

### 1) reset(region_id)
- 功能：开始/重置某区域的仿真 episode
- 返回：初始观测 `obs`

### 2) step(region_id, action)
- 功能：执行动作并推进一个决策步长
- 返回：下一观测 `obs2`、是否结束 `done`、以及可选 `info`

> **奖励 reward 可以不由平台返回**：本工程会在本地按论文公式基于观测与动作代价计算。

---

## 2. 统一观测结构（只需把CC4字段映射到这套结构）

```python
obs = {
  "region_id": rid,
  "t": step_idx,
  "events": [
     {"ts": t, "type": "auth_fail", "severity": 0.7, "src": "...", "dst": "..."},
     ...
  ],
  "risk_proxy": 0.0~1.0
}
```

- `events` 可为空列表
- `risk_proxy` 若平台不给，可用本地统计规则估计（后续可再换成更可靠的风险指标/告警概率）

---

## 3. 动作映射（本工程 → 平台）

本工程高层离散动作在 `src/action_space.py`：

- monitor
- light_evidence
- heavy_evidence
- local_mitigate
- strong_mitigate

你需要在 `src/cc4_client.py` 的 `OnlineCC4Client.step(action_id)` 中把 `action_id` 映射为 CC4 平台可执行的具体操作（例如：提高采样/触发补充日志/发起局部处置/隔离策略等）。

---

## 4. 需要补全的代码位置

文件：`src/cc4_client.py`

- `OnlineCC4Client.__init__`
- `OnlineCC4Client.reset`
- `OnlineCC4Client.step`

建议返回字段保持与 mock/replay 一致，以便其余模块无需改动。

