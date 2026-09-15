# Step 3-A2R — Four-Action Local-online Adapter Revision 审核记录

> 项目：UG-CEM-APT  
> 分支：`ug-cem-apt`  
> 阶段：Gate A / A2R  
> 审核日期：2026-09-15  
> 最终状态：**PASS / 已完成**

---

## 1. 本阶段目标

把历史 A=5 local-online adapter 收缩到新小论文的 A=4：

```text
0 = no_op
1 = analyse
2 = remove
3 = restore
```

不重新设计 local simulator，不提前接正式 reward。

---

## 2. Git 范围审核

相对 A1R 完成点，本阶段仅修改：

```text
chapter2_region_detection/shared/local_action_adapter.py
chapter2_region_detection/shared/local_online_client.py
chapter2_region_detection/tests/test_gate_a_local_action_adapter.py
chapter2_region_detection/tests/test_gate_a_local_online_client.py
```

未修改 CEM、UG uncertainty、WM、PPO、正式 CC4 evaluator 或 legacy src。

---

## 3. Action semantics

审核确认：

- no_op：无主动响应效果；
- analyse：只提升 visibility，不直接削弱 threat；
- remove：降低 auth/process pressure；
- restore：降低 auth/process/outbound pressure，不消除外部 scan pressure；
- control_traffic 已完全移出 local adapter；
- action ID 不作为连续强度。

---

## 4. Local-online client

保留：

- partial observability；
- latent-state isolation；
- observable event generation；
- observable-event-derived risk_proxy；
- seeded reproducibility；
- D=8 local development StateSummary compatibility；
- reward=0.0 placeholder。

注释已从“五类动作”改为“四类动作”。

---

## 5. Legacy protection

A=4 后 action_id=4 明确拒绝，避免旧五动作 replay 被静默解释。

测试同时验证：

```text
action_id=2 -> remove
action_id=3 -> restore
```

---

## 6. 本地测试

用户报告：

```text
python -m unittest discover -s tests -p "test_gate_a_*.py"

Ran 33 tests
OK
```

组成：

```text
A1R    11
A2R.1   9
A2R.2  13
----------------
total  33
```

---

## 7. 审核结论

```text
Four-action local semantics      PASS
control_traffic removal          PASS
Categorical treatment            PASS
Latent isolation                 PASS
StateSummary compatibility       PASS
Reward boundary                  PASS
Reproducibility                  PASS
Legacy action rejection          PASS
Scope isolation                  PASS
Gate A tests 33/33               PASS

FINAL STATUS: PASS
```

当前下一阶段：

```text
Gate A3 — Official CybORG / CC4 Four-Action Adapter
```
