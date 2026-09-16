# Step 3-A3.2+A3.3 — Four-Action CC4 Adapter + Deterministic Host Resolver 审核记录

> 项目：UG-CEM-APT  
> 分支：`ug-cem-apt`  
> 阶段：Gate A / A3.2+A3.3  
> 审核日期：2026-09-16  
> 最终状态：**PASS / 已完成**

---

## 1. 本阶段目标

在 A3.1 真实 wrapper probe 基础上冻结：

```text
0 no_op   -> dynamic valid Sleep
1 analyse -> Analyse(host)
2 remove  -> Remove(host)
3 restore -> Restore(host)
```

并提供 Ours / UG-CEM / CEM 共用的 deterministic target resolver。

---

## 2. Git 范围审核

相对 A3.1 完成点，本阶段恰好新增 4 个预期文件：

```text
chapter2_region_detection/shared/cyborg_action_adapter.py
chapter2_region_detection/shared/cyborg_target_resolver.py
chapter2_region_detection/tests/test_gate_a_cyborg_action_adapter.py
chapter2_region_detection/tests/test_gate_a_cyborg_target_resolver.py
```

未修改 legacy experiments、旧 action space、CEM、UG uncertainty、world model 或 PPO。

---

## 3. Resolver 审核

确认实现满足：

- 先检查 labels/mask 长度；
- action_mask=False 一律过滤；
- `[Invalid]` label 防御性过滤；
- 只保留 Sleep / Analyse / Remove / Restore；
- Monitor / traffic / decoy 不进入四动作空间；
- Sleep 动态解析，不写死 49/145；
- host-specific action 只从当前合法 family 中选择；
- observable host score 最高优先；
- 分数相同按 host 名字字典序、再 raw index deterministic tie-break；
- 无合法 observable target 返回 None，由 adapter fallback Sleep；
- 非有限 host score 明确拒绝；
- resolver 本身不读取 Red truth / compromise truth / future / test label。

---

## 4. Adapter 审核

确认：

- action_id 仍通过统一 `action_contract.py`；
- no_op 明确映射 Sleep，不映射 Monitor；
- 每次 resolve 都重新读取当前 wrapper 的 labels + mask；
- Analyse / Remove / Restore 共用同一个 resolver；
- 无合法 target 时 fallback 到当前有效 Sleep；
- resolution metadata 保留：
  - agent_name
  - requested_action_id/name/family
  - executed_index/label/family
  - target_host
  - fallback
  - fallback_reason
- 没有固定 raw action index；
- 没有 hidden-truth input；
- 没有 reward / WM / PPO / duration 逻辑越界进入本阶段。

---

## 5. 本地测试

用户报告：

```text
Gate A             58/58  OK
Categorical CEM    16/16  OK
UG uncertainty     12/12  OK
```

新增：

```text
target resolver    12
action adapter     13
```

原 Gate A 33 + 新增 25 = 58。

---

## 6. 审核结论

```text
Git scope                         PASS
dynamic Sleep                     PASS
no_op != Monitor                  PASS
mask-first filtering              PASS
invalid padded-slot protection    PASS
family-specific target selection  PASS
observable-score-only interface   PASS
deterministic tie-break           PASS
fallback Sleep                    PASS
requested/executed metadata       PASS
no fixed raw index                PASS
legacy action_id=4 rejection      PASS
Gate A 58/58                      PASS
CEM regression 16/16              PASS
UG regression 12/12               PASS

FINAL STATUS: PASS
```

下一阶段：

```text
A3.4 — multi-tick duration / next-decision availability probe
```
