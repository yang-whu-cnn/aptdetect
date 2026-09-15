# Step 3-A3.1 — Official CC4 Four-Action Wrapper Probe 审核记录

> 项目：UG-CEM-APT  
> 分支：`ug-cem-apt`  
> 阶段：Gate A / A3.1  
> 审核日期：2026-09-16  
> 最终状态：**PASS / 已完成**

---

## 1. 本阶段目标

在冻结正式 CC4 adapter 前，直接探测真实 `BlueFixedActionWrapper`，只围绕当前小论文四动作：

```text
0 = no_op   -> Sleep
1 = analyse -> Analyse(host)
2 = remove  -> Remove(host)
3 = restore -> Restore(host)
```

并验证：

- blue_agent_0..4；
- action_labels；
- action_mask；
- hosts；
- actions；
- pad_spaces=False / True；
- seeds 42 / 43 / 44；
- 不允许提前假设固定 raw action index。

---

## 2. Git 范围审核

相对 A3.1 开始点，本阶段提交恰好新增 6 个文件：

```text
chapter2_region_detection/formal_experiments/__init__.py
chapter2_region_detection/formal_experiments/data_collection/__init__.py
chapter2_region_detection/formal_experiments/evaluation/__init__.py
chapter2_region_detection/formal_experiments/probes/__init__.py
chapter2_region_detection/formal_experiments/probes/probe_cc4_four_action_contract.py
chapter2_region_detection/formal_experiments/training/__init__.py
```

未修改 legacy experiments、shared action semantics、CEM、UG uncertainty、WM 或 PPO。

---

## 3. Probe 结果

上传的 `four_action_contract.json` 共包含：

```text
3 seeds × 2 pad settings = 6 runs
5 Blue agents / run
```

所有 run 均成功发现：

```text
Sleep
Analyse
Remove
Restore
```

### pad_spaces=False

`blue_agent_0..3`：

```text
Discrete(82)
Sleep valid index = 49
16 Analyse host slots
16 Remove host slots
16 Restore host slots
1 subnet / 17 host entries
```

`blue_agent_4`：

```text
Discrete(242)
Sleep valid index = 145
48 Analyse host slots
48 Remove host slots
48 Restore host slots
3 subnets / 51 host entries
```

这些 index 只能作为当前 probe 的观察结果，正式 adapter 不得写死。

### pad_spaces=True

所有 agents 的空间统一到：

```text
Discrete(242)
```

`blue_agent_0..3` 会出现大量 padding Sleep slots，但只有一个真实有效 Sleep：

```text
Sleep total = 161
Sleep valid = 1
valid index = 49
```

`blue_agent_4` 本身已是最大空间：

```text
Sleep total = 1
Sleep valid = 1
valid index = 145
```

---

## 4. Invalid slot 语义

真实 wrapper 证明：

```text
[Invalid] Analyse ...
[Invalid] Remove ...
[Invalid] Restore ...
```

虽然 label 中仍保留原动作名，但：

```text
action_mask = False
underlying action_class = Sleep
action_repr = Sleep
```

因此正式 adapter 必须：

```text
labels + mask
    ↓
先过滤 mask=True
    ↓
再匹配 action family / target host
```

不能只靠 label substring，也不能把 invalid padded slot 当真实响应动作。

---

## 5. Seed variation

不同 seed 下实际有效 host 数会变化。

示例：`blue_agent_0` 有效 Analyse / Remove / Restore host 数：

```text
seed 42 -> 14 / 14 / 14
seed 43 ->  8 /  8 /  8
seed 44 -> 12 / 12 / 12
```

因此正式 resolver 必须基于每个 reset / 当前 wrapper 的：

```text
action_labels
action_mask
hosts
```

动态解析，不允许把某次 episode 的有效 host 集永久缓存。

---

## 6. no_op

真实 action list 同时存在：

```text
Monitor
Sleep
```

因此当前高层：

```text
0 = no_op
```

继续显式映射到：

```text
Sleep
```

Monitor 仍作为 CC4 的监控机制，不作为四动作中的 no-op。

---

## 7. 正式 adapter 冻结约束

A3.2 / A3.3 必须满足：

1. 不写死 49 / 145 或任何 raw index；
2. 先用 `action_mask=True` 过滤；
3. high-level action 只选四类 family；
4. Analyse / Remove / Restore 必须解析 host target；
5. target resolver 只使用当前可观察信息；
6. deterministic tie-break；
7. 没有合法目标时 fallback 到有效 Sleep；
8. 日志同时记录 requested high-level action、executed low-level label、target host、fallback reason；
9. 同一 resolver 必须供 Ours / UG-CEM / CEM 共用。

---

## 8. 审核结论

```text
Probe code scope                 PASS
5 Blue agents                    PASS
3 seeds                          PASS
pad_spaces False/True            PASS
Sleep discovery                  PASS
Analyse discovery                PASS
Remove discovery                 PASS
Restore discovery                PASS
hosts API                        PASS
actions API                      PASS
action_mask behavior             PASS
invalid-slot behavior            PASS
seed-dependent valid hosts       PASS
agent-4 larger action space      PASS
fixed-index assumption rejected  PASS

FINAL STATUS: PASS
```

下一阶段：

```text
A3.2 no_op -> Sleep
A3.3 deterministic shared host resolver
```
