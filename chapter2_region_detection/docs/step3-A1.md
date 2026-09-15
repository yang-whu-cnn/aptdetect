# Step 3-A1 — Gate A 动作契约冻结审核记录

> 项目：UG-CEM-APT  
> 分支：`ug-cem-apt`  
> 阶段：Gate A / A1 — Shared Action Contract  
> 审核日期：2026-09-15  
> 最终状态：**PASS / 已完成**

---

## 1. 本阶段目标

A1 的目标不是接入环境，而是先冻结第二章正式公平比较使用的高层动作契约。

本阶段确定：

```text
A = 5

0 = no_op
1 = analyse
2 = control_traffic
3 = remove
4 = restore
```

论文语义为：

```text
no_op
= no operation / monitor

analyse
= intrusion investigation

control_traffic
= traffic blocking

remove
= user-level compromise removal

restore
= host reimaging
```

动作 ID 仅作为离散类别索引，**不得把 ID 大小解释为连续动作强度**。

---

## 2. 实现策略

为保留原项目代码和历史实验的可回滚性，本阶段没有修改旧：

```text
chapter2_region_detection/src/action_space.py
```

而是新增独立共享层：

```text
chapter2_region_detection/shared/__init__.py
chapter2_region_detection/shared/action_contract.py
chapter2_region_detection/tests/test_gate_a_action_contract.py
```

该共享层后续应同时供：

```text
LWM-RL (Ours)
UG-CEM-APT
以及后续公平比较 baseline
```

使用，不属于 UG-CEM 私有动作空间。

---

## 3. Git 范围审核

相对 Step 3 完成后的基线，本阶段只新增 3 个文件：

```text
chapter2_region_detection/shared/__init__.py
chapter2_region_detection/shared/action_contract.py
chapter2_region_detection/tests/test_gate_a_action_contract.py
```

审核确认：

- 未修改 `chapter2_region_detection/src/`；
- 未修改 UG 官方源码；
- 未提前修改 world model；
- 未提前修改 local-online；
- 未提前修改 CybORG evaluator；
- 未加入 reward / PPO / CEM / uncertainty 逻辑。

因此阶段边界符合 A1 要求。

---

## 4. 动作 ID 审核

当前 `ACTION_CONTRACTS` 固定为：

| ID | 动作 | 论文语义 |
|---:|---|---|
| 0 | `no_op` | no operation / monitor |
| 1 | `analyse` | intrusion investigation |
| 2 | `control_traffic` | traffic blocking |
| 3 | `remove` | user-level compromise removal |
| 4 | `restore` | host reimaging |

审核结果：**PASS**。

其中 `0 = no_op` 保留“默认不主动处置”的工程语义，其余动作 ID 仅用于稳定索引。

---

## 5. 当前 CybORG 类型字段

A1 当前记录：

```text
no_op           -> Sleep
analyse         -> Analyse
control_traffic -> BlockTraffic
remove          -> Remove
restore         -> Restore
```

这在 A1 中只作为“计划映射动作类型/动作族”的契约记录。

**A1 尚未完成真实 CybORG action label、host 参数、subnet 参数和 action mask 验证。**

尤其：

```text
control_traffic
```

对应的真实 CC4 底层动作名称和参数形式，必须在后续 **A3 CybORG adapter** 中通过实际 action space / label 做验证。

因此 A1 PASS 不等价于 Gate A 已全部通过。

---

## 6. Duration 记录

当前契约记录：

```text
no_op           1
analyse         2
control_traffic 1
remove          3
restore         5
```

A1 仅负责保存该动作契约信息。

是否把 duration 作为最终 predicted return / reward 的代价项，仍属于后续 reward 设计问题，不在 A1 中决定。

---

## 7. Cost 处理

A1 没有在 `ActionContract` 中加入正式 action cost。

这是有意设计，因为 Gate B 尚未冻结最终 reward。

因此当前阶段不继续沿用旧：

```text
0.05 / 0.20 / 0.45 / 0.75 / 1.10
```

作为新动作空间的正式动作成本。

后续如果最终 reward 需要 action cost，应在 Gate B 中统一冻结，并保证 Ours 与 baseline 完全一致。

---

## 8. API 审核

当前共享动作契约提供：

```text
ACTION_CONTRACTS
N_ACTIONS
ID2ACTION
NAME2ID
get_action()
get_action_id()
get_cyborg_action_type()
get_action_duration()
```

审核确认：

- ID -> ActionContract 查询明确；
- name -> ID 查询明确；
- 非法 ID 抛 `ValueError`；
- 非法动作名抛 `ValueError`；
- 不依赖旧 `src/action_space.py`；
- 不把 action ID 用于连续数值运算。

---

## 9. 单元测试

用户本地运行：

```bash
python tests/test_gate_a_action_contract.py
```

结果：

```text
Ran 9 tests
OK
```

测试覆盖：

1. 动作数固定为 5；
2. numeric ID 顺序；
3. name -> ID；
4. ID -> ActionContract；
5. 五类论文语义；
6. CybORG 类型字段；
7. duration；
8. 非法 action ID；
9. 非法 action name。

GitHub code review 同时确认测试代码与实际共享契约一致。

---

## 10. 非阻塞改进项

当前没有阻塞 A2 的缺陷。

可选工程加固项包括：

- 更严格拒绝非整数数值 ID（例如 `1.5`）；
- 增加 action ID/name 唯一性自检；
- 增加 `duration_ticks > 0` 的通用校验。

这些不影响当前固定动作契约，也不需要在 A1 阶段继续扩展。

---

## 11. 尚未完成的 Gate A 内容

Gate A 仍处于进行中。

下一阶段必须继续完成：

```text
A2 — local-online action adapter
A3 — CybORG / CC4 action adapter 与真实 control_traffic 验证
A4 — replay / world-model checkpoint 兼容性与 cost/delay 规则
A5 — Gate A 总体验收
```

在 Gate A 全部通过之前：

- 旧 replay 不自动视为新动作语义下有效；
- 旧 world-model checkpoint 不自动视为正式模型；
- 不能直接进入正式 Step 4 集成。

---

## 12. 最终审核结论

```text
Shared action layer          PASS
Five-action semantics       PASS
Numeric ID contract         PASS
Legacy src isolation        PASS
Name / ID mappings          PASS
Input validation            PASS
Unit tests                  PASS
Stage scope                 PASS

FINAL STATUS: PASS
```

**Gate A1 正式关闭。**

当前下一阶段：

```text
Gate A2 — Local-online Action Adapter
```

A2 继续遵循“不修改旧 `src/cc4_client.py`，优先新增共享适配层”的策略。
