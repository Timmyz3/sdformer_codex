# Local5 联合 Relation Memo 统计合同与三轮复审

## 1. 目的

本分析器只在最终 epoch29 的 same-window all-head 正式 trace 完成后运行，用于决定 Relation Memo 是否值得进入 checkpoint-bound 单顶层 RTL。

旧 1.342x 模型不再用于门槛判断，因为它独立 bootstrap head，破坏了 head 相关性和 head 条件分布。

## 2. 估计口径

### 2.1 Joint window

每个联合观测必须同时包含同一 `sample/stage/block/window` 的全部 head：

- S0：3 heads。
- S1：6 heads。
- S2：12 heads。
- S3：24 heads。

分析器逐条验证 canonical head 顺序、selection plan window、payload offset 和 13800 个 group 的完整性，不再生成 synthetic head 组合。

### 2.2 Horvitz-Thompson 全帧估计

每个 sample/block 均匀抽取一个 window，纳入概率为 `1/W`。该观测的全 block window 总量估计权重为 `W`：

`estimated total = observed window cost x batch_windows`。

recompute 与 Relation Memo 两侧使用完全相同的权重；headline speedup 为两侧 HT total 均值之比。

### 2.3 Sequence-cluster 置信区间

100 个 valid 样本来自 18 个 sequence，同一 sequence 内样本不视为独立。正式 CI 使用 18 个 sequence 的整簇 percentile bootstrap：

1. 先把每个 sequence 内所有 sample 的加权 baseline/candidate 周期求和。
2. 每次 bootstrap 从 18 个 sequence cluster 中有放回抽 18 个 cluster。
3. 计算配对 ratio。
4. 报告 2.5% 和 97.5% 分位数。

正式 cohort 必须同时满足：100 个非空 sequence key、恰好 18 个 sequence、重算 sequence-key SHA256 一致、重算 sequence_counts 一致。

## 3. Fail-closed 输入绑定

分析器会重新验证：

- 固定 seed 20260809。
- 每个计划 window 等于规定 SHA256 PRF 的重算结果且地址合法。
- source cohort manifest 路径和 SHA256。
- plan、manifest、cohort 的 cohort SHA 一致。
- run identity 与 plan/checkpoint/cohort 一致。
- GPU exclusivity audit 为 PASS，并绑定当前 identity/manifest/payload SHA256。
- payload、cohort 和 manifest 完整。

因此不能通过人工选择高收益 window 后再自报 `pi=1/W` 的方式通过分析器。

## 4. 周期模型

延续旧 Relation Vault 的公平口径：

- relation build：450 cycles。
- 强基线：每个 head/output-tile 计 `max(450, projection service)`。
- Relation Memo：首遍 build；驻留 head 后续 replay；容量 miss 与非 admission head exact recompute。
- 固定 clear/final drain 周期两侧一致。
- 容量主配置：7 KiB、112-bit relation record。
- admission 主配置：critical-only；first-fit-all 作为单变量消融。

该结果仍是 `[prof]+[模型]`，不是 RTL 周期、ASIC PPA、功耗或 EDP。

## 5. 晋级门槛

只有 critical-only 7 KiB 同时满足以下条件，才进入最终单顶层 RTL：

1. Sequence-cluster bootstrap 的 speedup 95% CI lower 不低于 1.15x。
2. 输入为 1200 个真实 joint windows、13800 个真实 head groups。
3. 所有 provenance 与 GPU audit 检查通过。

模型 PASS 后仍必须完成：同 trace recompute 与 pack/replay/fallback、相同端口和反压、Acc32 逐项零失配。模型 PASS 不能直接写成 DATE 架构贡献。

## 6. 三轮独立复审

| 轮次 | 发现 | 整改 |
|---:|---|---|
| 1 | 未重算 PRF；sample bootstrap 忽略 sequence 相关；follower 可复用陈旧结果 | 重算 seed/PRF/source cohort/GPU audit；改 sequence cluster；绑定输入与源码 |
| 2 | 已有 report 的早返回绕过 upstream 检查；未硬断言 18 sequence | 复用同时要求 upstream 完整；重算 sequence hash/count 并固定 18 cluster |
| 3 | 无剩余阻断或高风险 | 批准启动纯 CPU follower |

## 7. 当前状态

- 分析器：`scripts/analyze_local5_joint_relation_memo.py`。
- 等待器：`scripts/watch_local5_joint_relation_memo.py`。
- 单元测试：9/9 PASS。
- CPU follower PID：3139466。
- 当前状态：等待 Local5 same-window all-head profile；不占 GPU。
