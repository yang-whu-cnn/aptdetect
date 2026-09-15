# Step 3-A1R — Four-Action Contract Revision 审核记录

> 项目：UG-CEM-APT  
> 分支：`ug-cem-apt`  
> 阶段：Gate A / A1R  
> 审核日期：2026-09-15  
> 最终状态：**PASS / 已完成**

---

## 1. 修订背景

2026-09-15 新小论文方案将正式高层动作空间冻结为：

```text
0 = no_op
1 = analyse
2 = remove
3 = restore
```

原 A=5 合同中的 `control_traffic` 已退出当前 LWM-RL / UG-CEM / CEM-APT 主对比动作空间。

---

## 2. Git 范围审核

相对 v2.1 文档冻结点，本阶段只修改：

```text
chapter2_region_detection/shared/action_contract.py
chapter2_region_detection/tests/test_gate_a_action_contract.py
```

未修改：

- Categorical CEM；
- UG uncertainty；
- world model；
- local adapter；
- CC4 evaluator；
- replay；
- PPO；
- legacy src。

范围符合 A1R。

---

## 3. 正式动作契约

```text
A = 4

0 = no_op
1 = analyse
2 = remove
3 = restore
```

语义：

- no_op = no operation
- analyse = intrusion investigation
- remove = user-level compromise removal
- restore = host reimaging

CC4 type metadata：

```text
Sleep
Analyse
Remove
Restore
```

duration：

```text
1 / 2 / 3 / 5
```

动作 ID 仅为 categorical ID，不表示连续强度。

---

## 4. 健壮性增强

A1R 同时增加：

- action ID 唯一性检查；
- action name 唯一性检查；
- ID 必须从 0 连续；
- duration 必须为正整数；
- 非整数 ID 拒绝；
- bool ID 拒绝。

这可以提前暴露 replay/config 中的动作编码错误。

---

## 5. Legacy incompatibility

旧 A=5 中：

```text
2 = control_traffic
3 = remove
4 = restore
```

新 A=4 中：

```text
2 = remove
3 = restore
```

因此旧 replay / WM / PPO checkpoint 不能通过“仍然是整数 0..4”来继续使用。

正式结论：

> 旧五动作 replay / WM / PPO 全部 legacy-only；A4 必须重采 replay 并重训正式模型。

---

## 6. 单元测试

用户本地执行：

```text
test_gate_a_action_contract.py   11/11 PASS
test_categorical_cem.py          16/16 PASS
test_ug_uncertainty.py           12/12 PASS
```

A1R 测试覆盖：

- A=4；
- ID 顺序；
- name->ID；
- ID->contract；
- 论文语义；
- CC4 action type；
- duration；
- control_traffic 已删除；
- legacy names 拒绝；
- invalid ID；
- non-integer ID。

Step2/Step3 regression 同时通过，说明动作数变化未破坏 CEM 与 uncertainty 模块。

---

## 7. 审核结论

```text
Four-action contract          PASS
ID mapping                    PASS
Paper semantics               PASS
CC4 type metadata             PASS
Duration metadata             PASS
Legacy action rejection       PASS
Input validation              PASS
CEM regression                PASS
UG uncertainty regression     PASS
Scope isolation               PASS

FINAL STATUS: PASS
```

当前下一阶段：

```text
Gate A2R — Local-online Adapter Revision
```
