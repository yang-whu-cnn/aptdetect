# 最终论文方案与对比实验执行基线

记录日期：2026-09-18。

后续 ChatGPT/Codex 接手时，先读 `CHATGPT_HANDOFF_FINAL_EXPERIMENTS.md`；该文件汇总当前提交、有效产物、阻塞、精确续跑命令与完整后续路线。

## 1. 文档优先级

本项目后续实现与实验以桌面文件《融合LLM+WM+RL的APT动态响应小论文-20260917.docx》为最终方案来源。

仓库根目录《大模型+预测强化学习-电力系统APT检测.docx》是第一版论文方案。`chapter2_region_detection` 是在第一版源码上持续改造形成的目录，因此其中的旧代码和旧结果需要先核对动作、reward、状态、环境和评测协议，不能直接视为最终方案结果。

仓库根目录的 PDF 是 Table 1 对比方法的论文来源，用于核对算法并复现或适配 baseline。仓库中的开发计划和历史 Gate 文档服务于工程实现；与 20260917 最终论文冲突时，应更新工程协议。

## 2. 最终方法契约

最终方法链路为：

```text
LLM 生成 K 个候选响应计划及先验偏好
    -> World Model 预测各候选计划的回报与不确定性
    -> PPO 选择候选计划
    -> 执行所选计划的第一个动作
    -> 下一决策时刻重新规划
```

正式高层动作空间固定为四类：

```text
0 no-op
1 analyse
2 remove
3 restore
```

当前 `shared/action_contract.py` 已符合该动作定义。

## 3. 最终 reward

最终论文定义：

```text
r_t = -lambda_1 * r_delay_t - lambda_2 * r_fail_t

r_delay_t = (t - t_compromise) * I(host remains compromised at t)
```

多台受感染主机并存时，当前实现按各 host-level incident 的感染年龄求和。一个 incident 在区间结束时仍处于本区间 active 集合，则贡献 `global_tick_end - t_compromise`。该定义使延迟惩罚随感染持续时间增加。

`r_fail_t` 使用 CC4 Host Work Fail 原始负奖励。最终论文的 Operation Failure Penalty 包括攻击或响应造成的正常任务失败，因此 reward 统计该 Blue agent 管辖范围内所有 Host Work Fail，不再限定为已经感染的主机。为兼容已有 replay schema，历史字段名 `incident_host_lwf_*` 暂时保留，但新版本语义是区域内全部 Host Work Fail；新产物必须记录 reward 协议版本。

旧实现每个 active incident 每 tick 只计固定 1，且只统计 active-incident host 的 LWF，与最终论文不一致。旧 replay、reward predictor、PPO provisional 权重和 B0.2 报告可作为历史工程证据，不能作为最终论文实验产物。

2026-09-18 已核验 `outputs/formal_replay_v2/train.jsonl` 与 `validation.jsonl` 的首条记录：它们含 `incident_event_ids`、`incident_active_ticks` 和旧范围的 `incident_host_lwf_raw_penalty`，但不含 `incident_delay_penalty`，也没有每个事件的 `t_compromise`。因此无法从现有 decision-epoch 记录无损恢复感染年龄；旧记录还漏掉健康主机上的 LWF，亦无法补算最终 `r_fail`。结论是必须重新采集 final-reward replay，不能只改标签或在旧 reward 上继续训练。

## 4. 三张结果表

Table 1 主比较：UAMCTS、RSMBRL、CARL、DCA、PriorRL、TERLA，以及本文 LWM-RL。每种方法统一接入 D27、A4、CC4、target resolver 和正式评价程序。第一版五动作或旧 reward 的结果不得直接填入。

Table 2 模块消融：RL-Only、LLM-RL、WM-RL、LWM-RL。每个变体独立初始化和训练，不能在完整模型训练后简单屏蔽输入冒充训练消融。

Table 3 reward 消融：Delay-Only、Fail-Only、Full-Reward。分别对应 `(lambda_1, lambda_2) = (1,0)、(0,1)、(1,1)`。需要同时冻结 PPO 真实 reward 和候选计划预测回报的消融语义。

三张表统一报告：CC4 Official Reward、Operation Failure Penalty、Recovery Precision、Recovery Time，所有单元格使用 mean ± standard deviation。

## 5. 正式评价协议

最终论文写明：FiniteStateRedAgent；100 个 test episodes；每个 episode 500 timesteps；每项实验独立重复五次。

五次重复指五次独立训练/运行重复，不能用五个 Blue agents、五个区域或同一次训练的五个 episode 代替。每次重复都保存训练 seed、环境 seed、checkpoint、配置和逐 episode 原始指标。

Recovery Precision 按论文 `TP / (TP + FP)`，需要在实现前冻结 TP/FP 与 Remove/Restore 成功事件的对应关系。Recovery Time 按受感染主机从 compromised 到 normal 的时间计算，同时报告未恢复 incident 数，避免只平均成功恢复事件造成偏差。

## 6. 当前状态与执行决定

已完成的 L/M provisional PPO 和 B0.2 属于旧 reward 协议。GPT-5.6 Sol 的付费 provisional 阶段暂停，直至最终 reward、replay、reward predictor 和正式评价协议完成对齐。

执行顺序固定为：

1. 修改 incident bookkeeping 和 replay contract，落实感染年龄延迟项与全区域 LWF。
2. 完成定向单元测试和小规模无付费 smoke。
3. 判断旧 replay 是否包含重标注所需信息；无法可靠重标注时重新采集 train/validation replay。
4. 重新训练并验证 response reward predictor；状态转移 WM 是否重训由 replay/分布审计决定。
5. 重新进行最终协议下的 Ours provisional、B0.2 和正式训练。
6. 实现统一正式评价与 Table 2、Table 3 变体。
7. 适配 Table 1 baseline，冻结后运行 100×500、五次重复并生成三表。

任何付费 LLM 正式运行都必须记录模型 ID、prompt/version、cache namespace、token、费用和结果协议版本，避免把旧 reward 缓存或权重混入最终实验。

截至 2026-09-18，本清单第 1 步和第 2 步的核心代码已完成：配置写入 `final_paper_20260917_v1`，replay 新增 `incident_delay_penalty`，provisional manifest 升级为 v2 并绑定新 reward 协议。关键契约测试 86 项通过。另用 seed 1000 运行了 10 ticks 的无付费 CC4 采集 smoke，生成 45 条 decision transitions，45 条均含新字段；该短轨迹未发生 incident，因此它只验证真实采集链路与 schema，不能验证真实攻击下的数值分布。第 3 步的结论是必须重采集；在新 replay 和新模型产物生成前，不运行 H 付费扩展和最终表格实验。

随后已完成 final-reward replay 重采集：train 为 32 seeds × 500 timesteps、43,297 transitions，validation 为 8 seeds × 500 timesteps、10,796 transitions。两组均无新字段缺失，逐条复算 reward 公式均为零不一致；文件哈希、源码哈希、seed 和覆盖统计见 `FINAL_REWARD_REPLAY_MANIFEST.json`。原始 JSONL 共约 85 MB，保存在本地忽略目录，不提交 Git；manifest 提交 Git 用于追溯和校验。新 delay 标签最大值超过 20,000，明显改变旧 reward 尺度，训练 reward predictor 前必须检查其标签标准化和数值稳定性。

基于新 replay 已重新训练 absolute/delta WM 和 final-reward predictor。A4.5b 仍选择 absolute：H4 RMSE 0.130383，优于 persistence 0.156583，质量门通过。reward predictor 的 WM-H4 RMSE 为 2313.807，优于常数基线 4919.828，Spearman 为 0.7022，8/8 validation episodes 为正，质量门通过。action consistency 与重新绑定本次 A4.5b 报告的 B0.1 均通过。旧 B0.1 脚本原先只允许复现旧 checkpoint 的硬编码指标，现保留历史默认值，并支持通过 `--reference-report` 审计同一新协议 checkpoint。模型和报告哈希见 `FINAL_REWARD_MODEL_MANIFEST.json`。

final-reward provisional strict manifest 已升级为 v3，强制绑定新 WM/reward predictor 哈希，并使用 `outputs/lwm_rl_final_20260917` 的独立输出与 cache namespace。L 档 2,000-transition 阶段已尝试启动，但在第一次 API 调用前因运行环境未配置 `OFOX_API_KEY` 停止；live API calls、cache records、probe transitions 和费用均为 0。配置密钥后应在相同独立目录重新 fresh 启动，无需 `--resume`。
