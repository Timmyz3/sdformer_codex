# Motion TESC 到 Gated-K 闭环与 HIFP 维护回归

## 本轮结论

Motion 没有停止推进。本轮完成了两项不同性质的工作：

1. 重新执行 HIFP 的 PPDI/IBF 四模式、四 stage 真实 T162 回归，全部逐 Acc32 通过；
2. 将 TESC-WD 从原 weighted-SCS 前端继续接到 exp LUT、Shiftmax、Q1.7 gate 和 K-pair 展开，并与原 H67 row engine 完成合成 T450 gated-K miter。

第一项是现有机制维护证据，不增加创新性；第二项首次关闭 TESC 的 quotient boundary 正确性缺口，使其从 `[prof]+[模型]+前端RTL` 推进到完整 attention-row 输出边界 `[rtl]`。

## HIFP 维护结果

`sim_hitflow/run_gatestack_ppdi_ibf_real_trace_checks.sh` 已重新执行：

| 模式 | 总周期 | 相对标量 RMW | 结果 |
|---|---:|---:|---|
| 标量 + RMW | 53,910 | 1.000× | 四 stage Acc32 PASS |
| PPDI + RMW | 49,350 | 1.092× | 四 stage Acc32 PASS |
| 标量 + IBF | 50,295 | 1.072× | 四 stage Acc32 PASS |
| PPDI + IBF | 45,735 | 1.179× | 四 stage Acc32 + S0 SVA PASS |

边界保持不变：输入是旧 Motion `sample0/window0/T162`，只覆盖 projection，manifest 没有最终 checkpoint/config SHA。它不能替代 T450 多样本证据，也不能把 1.179× 写成系统加速。

## TESC-WD 完整数据流

新增顶层 `h67_temporal_quotient_shiftmax_gate_top`：

```text
{Q0,Q1,K0,K1,pair_id}
        |
        v
两个 bit-exact Motion-XOR Q7 score
        |
        v
score quotient
  equal -> {score, temporal_mask=11, active_k_mask}
  diff  -> 两个单时间 descriptor
        |
        v
weighted SCS directory
  class multiplicity += popcount(temporal_mask)
        |
        v
exp LUT + row sum + Shiftmax Q1.7 gate
        |
        v
按 active_k_mask 从 K-pair store 展开
        |
        v
{token_id, K_bits, gate_q17, last}
```

架构边界的关键是：只在归一化域把相等 score 取商，分母仍计入两个 token；到 gated-K 输出时重新展开 K0/K1。它不删除 token，不假设 K0=K1，也不把相同 gate 当作相同投影 destination。

## RTL 结果

完整报告：`results/h67_tesc_gated_k_miter_20260805/report.md`。

| 验证项 | 结果 | 等级 |
|---|---:|---|
| Icarus 多 seed | 5/5 PASS | `[rtl]` |
| Icarus T450 preserve-mean=0/1 | 2/2 PASS | `[rtl]` |
| Verilator T450 + SVA | 1/1 PASS | `[rtl]` |
| gated-K 累计逐项比较 | 1,418，零失配 | `[rtl]` |
| Icarus/Verilator T450 计数 | 完全一致 | `[rtl]` |
| focused lint | 0 输出 | `[rtl]` |
| candidate/baseline Yosys check | 0 problems / 0 problems | `[rtl]` |

T450 的确定性合成向量结果为：225 pair、450 token、366 个非零 K 输出、383 个 quotient descriptor、67 个 equal pair、67 个 occupied class。两种 preserve-mean 设置均逐项等价。

合成向量的 exp transaction 计数为 TESC 390、原 row engine 746，只能证明实现中的共享门控因果成立，不能作为真实 workload 能耗收益。真实动机仍来自旧 T162 profile100：双 K 有效 score 相等率 86.93%、SCS active entry 模型降低 22.21%、全 SCS 指数事务模型降低 19.40%。

## 相对已有工作的本土化差分

| 来源 | 借用 | Motion 本土化差分 |
|---|---|---|
| Bishop TTB | token-time pair 是调度原子 | 不做 ECP 或 dense/sparse 双核，只在 Q7 等价类内取商 |
| Phi 两级 pattern/residual | common/residual 分层 | common 是 exact score class；residual 由 Motion-XOR 和 RNE 决定 |
| Prosperity exact reuse | detector/table 成本必须核算 | 不搜索历史 product；只比较同一已知时间对，K 在输出边界展开 |
| FireFly-T 时空数据流 | 时间维映射为物理数据流 | temporal mask 同时驱动 SCS multiplicity 与 gated-K expansion |
| SpAtten cascade issue | 后端只处理必要工作 | 不剪 token/head，不改变 Shiftmax 分母或支持集 |

可辩护的机制名称仍是：**TESC-WD，Temporal-Equivalence Score Coalescing with Weighted Directory**。当前可写为“精确的归一化域 quotient 与 gated-K 可逆展开数据流”，不能写成“首次时间打包”或“首次 exact reuse”。

## 当前不能宣称

1. TESC 已在最终 Motion checkpoint/T450 ordered trace 上获得 19.40% 能耗或周期收益；
2. K-pair store 已按同步 SRAM macro 时序闭合；
3. TESC 已连接 projection、ATLIF、skip 或 full encoder；
4. Yosys 结构可读等于 ASIC 面积/PPA；
5. HIFP 的 T162 单样本 1.179× 可以外推到 T450 或系统。

## 下一唯一门槛

等待 Motion T450 profile100/all12 trace 释放后，优先重放 TESC，而不是再新增第三个 Motion 机制：

1. 报告 equal pair、quotient descriptor、active entry、class/exp transaction 的 mean/p50/p95/p99 和最差 stage；
2. 用同一真实 trace 同时驱动原 row engine 和 TESC，逐 gated-K 零失配；
3. 把两者都改成相同同步 K SRAM 延迟、相同输出反压；
4. 生成 VCD/SAIF，比较 score+SCS+K-store 动态活动；
5. 若净能量下降不足 15% 或面积归一收益为负，TESC 降级为子机制而非主贡献。

Local5 仍是当前主线，等待新 rank-1 后接 v2 theta-folded production contract；Motion 的本轮闭环不会改变该优先级，也没有复用 Local5 结果冒充 Motion 证据。

## TESC 的架构化迭代：RQTB

在不改 Motion RTL 的前提下，本轮新增 `Reversible Quotient Token Bundle` 筛选模型，结果见 `results/motion_rqtb_screen_20260806/report.md`。RQTB 把 TESC 的 score 等价取商固化为：

```text
common 1x16-bit slot / split 2x16-bit slot
  -> weighted SCS multiplicity
  -> Shiftmax
  -> active-K bank选择
  -> gated-K原序展开
```

旧 crop/W9/T162 profile100 给出的 slot `-49.35%`、active entry `-22.21%`、exp 事务模型 `-19.40%` 均为 `[prof]+[模型]`，不是周期或能量。RQTB 不是 TESC 之外的新数学优化，论文中只能把二者合并为一条“可逆归一化域商流”贡献；16-bit FIFO、双 K bank 不能单列创新。

独立 DATE 复审仍为 `3/5，Reject/Major Revision`。唯一晋级门槛是 fullres W15/T450 下 fixed-TTB/RQTB 同步 K-SRAM、同反压 RTL 对照。真实 trace 释放前不扩 RQTB RTL；这不代表 Motion 停止，而是避免在关键分布未知时冻结错误 packet/macro 合同。

已增加纯 CPU follower `scripts/watch_motion_rqtb_fullres.py`。它等待 H67 fullres T450 的 profile、all12 trace 与组件 RTL 审计全部完成，随后自动执行 fullres TESC 分析和 RQTB 筛选，输出到：

- `results/motion_temporal_equivalence_fullres_t450_20260806/`；
- `results/motion_rqtb_fullres_t450_20260806/`。

watcher 当前 PID 为运行时状态，不作为论文证据；完成标志写入 `results/motion_rqtb_fullres_watcher_20260806.log`。该 follower 不调用训练或推理入口，不占 GPU。

## DATE 独立评审

本轮完成后由独立子代理严格按 DATE 硬件架构标准评审，不因 RTL 行数或工程工作量额外加分。

- 评分：`3/5`；
- 建议：`Reject（大修后重审）`；
- 正确性：合成 T450 gated-K 局部 RTL 闭环可信，“归一化域取商、分母保留 multiplicity、输出端展开 K0/K1”的语义边界正确；
- 创新性：现阶段只能作为 SCS/attention 前端子机制，不足以独立列为 DATE 主贡献；
- 文献差分：已能避免与 Bishop/Phi/Prosperity/FireFly-T/SpAtten 概念混淆，但尚缺同 workload、同接口、同资源下的定量对照；
- 严重缺口：真实 T450 分布、同步 SRAM/物理开销、projection 及 encoder 系统边界。

评审指定的下一唯一 Motion 门槛与本文前述判定一致：使用同一真实 T450 ordered trace，在相同同步 K SRAM 延迟、相同输出反压下对比 baseline 与 TESC-WD，完成逐 gated-K 零失配、mean/p50/p95/p99 分账与 VCD/SAIF 活动对照。真实 checkpoint 未释放前可执行 CPU-only 多 seed 边界模型，但只记为 `[模型]`，不替代 `[prof]`。未通过净动态能量降低 15% 且面积归一吞吐不为负的晋级门槛时，TESC-WD 应降级为 Motion SCS 子机制。

评审同意“Local5 当前主线、Motion 实质推进”的双线优先级：Local5 先闭合 theta-folded 部署合同和端到端 RTL-exact；Motion 继续 TESC 真实回放与 HIFP 证据维护，但不在缺真实 T450 证据时同时扩展更多 RTL 机制。
