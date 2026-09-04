# Local5 GASR-2C 单 Bank RTL 与第七轮 DATE 复审

## 1. 本轮边界

本轮只采用当前机器上可复现的文件、日志和脚本。任何外部机器上的 Claude 结果均不计入完成度、性能或论文结论。

本轮目标是关闭第六轮评审提出的单 bank 门槛：在相同单端口 1RW 存储合同下，实现 direct-RMW 与 GASR-2C，使用真实 Local5 post-G0 source-major trace 做 Acc32、周期和事务 A/B 对照。

## 2. 新增 RTL

- `rtl_qfit/qfit_single_port_acc_memory.sv`：每拍最多一条读或写命令的同步 1RW 合同；
- `rtl_qfit/qfit_direct_1rw_acc_bank.sv`：公平 direct 基线，首触直接写、复访同端口 read-modify-write、lazy-zero；
- `rtl_qfit/qfit_gasr2c_acc_bank.sv`：active/next 双驻留上下文，active 在寄存器内精确累加，spare 通过同一个 1RW backing SRAM 回写或预取。

GASR 的 `prepare_ready` 表示“指定地址已经准备完成”，不是普通 ready/valid 接口中的“当前可接收请求”。五 bank 集成时必须把它重命名或通过适配器转换，避免误接。

## 3. 真实向量与金参考

生成器：`scripts/generate_local5_gasr_singlebank_vectors.py`。

输入：

- qualified Local5 post-G0 profile100；
- 原 `ordered_term_items.npz`；
- 原完整投影 `expected_acc.memh`。

bank0 向量规模：

- 100 个分层抽样组；
- 8,948 个有 bank0 目标的 active source；
- 33,271 个 Acc32 向量更新；
- 90 个 bank 地址、OUT_DIM=2。

生成器把 gate、K lane 和原权重函数重新计算为两个 Acc32 增量，并把结果与原完整投影金向量逐项交叉检查，结果为 `18,000/18,000 PASS`。

## 4. 公平 RTL 结果

主结果目录：`results/local5_gasr2c_singlebank_postg0_rtl_20260803/`。

一键入口：`sim_new_arch/run_local5_gasr2c_singlebank_checks.sh`。

| 指标 | direct-1RW | GASR-2C | 变化 |
|---|---:|---:|---:|
| deterministic 执行周期 | 64,008 | 39,783 | `1.609x` |
| 执行期 SRAM 读 | 30,637 | 3,159 | -89.69% |
| 执行期 SRAM 写 | 33,271 | 5,793 | -82.59% |
| 执行期 SRAM 总事务 | 63,908 | 8,952 | -85.99% |
| Acc32 | PASS | PASS | 零失配 |

周期从 `run_start` 释放计到 `flush_done`，不含结果读回；事务同样不含结果读回。deterministic 流用于性能，随机输入空泡只用于握手和 bit-exact 压力验证。

## 5. 分布结果

100 组中 55 组非空；逐组 win/equal/loss 为 `33/0/22`。非空组加速分布：

- p50：`1.303x`；
- p95：`1.872x`；
- 最差：`0.350x`。

这说明 GASR 的固定 prepare/activate 开销会伤害低复用窗口。聚合 `1.609x` 由高工作量组主导，不能写成“每个窗口均加速”，也不能直接外推为五 bank 或完整 post-score 顶层结果。

## 6. 验证签核

| 项目 | 结果 |
|---|---:|
| Icarus 真实100组 deterministic | PASS |
| Icarus 真实100组随机输入空泡 | PASS |
| Verilator + SVA deterministic | PASS |
| Verilator + SVA 随机输入空泡 | PASS |
| Verilator RTL lint | PASS |
| Yosys 层次、memory、check、stat | PASS |

SVA 覆盖输入阻塞稳定性、地址范围、active/prepared 槽一致性、双槽地址唯一性、更新地址与 active 槽一致性以及单端口命令范围。当前 SRAM 仍是固定一拍同步合同，不是已经完成的真实宏 PPA。

## 7. 复用强度分层只作为下一候选

同一 cohort 的 post-hoc 结果显示，使用

`updates / active_target_sources >= 2`

选择 GASR，否则选择 direct-1RW，可把估算总周期降到 39,421，相对 direct 为 `1.624x`，且被选中的 GASR 组在当前数据上无退化。

这个阈值对应“每个已准备 source 至少发生两次更新”的自然盈亏点。它借鉴 Bishop 的密度分层，但计划在同一个 bank、同一个 SRAM 上切换 direct/source-resident 模式，不复制 dense/sparse 双核。

当前它仍是同 cohort 的 `[模型]`，没有选择器 RTL和留出集，不能列为论文已完成贡献。

## 8. 第七轮独立 DATE 复审

| 维度 | 分数 |
|---|---:|
| 新颖性 | 3.0/5 |
| 完整度 | 3.0/5 |
| 验证 | 4.0/5 |
| 实验 | 3.0/5 |
| 公平性 | 3.5/5 |
| 总体 | 3.0/5，Borderline Reject |

评审结论为：**ALLOW 五 bank 扩展**。单 bank 已通过工程晋级门槛，但不是论文系统结论。

评审认可：

- `[rtl]` 单 bank `1.609x` 与 SRAM 事务下降 85.99%；
- `[rtl]` 原完整投影 Acc32 金参考交叉检查；
- `[rtl]` 双仿真器、SVA、随机空泡、lint 和 Yosys；
- 公平基线使用相同 1RW、位宽、深度和 lazy-zero。

评审拒绝外推：

- 五 bank 综合周期；
- 真实前端持续供给；
- 跨 bank 反压；
- SRAM macro PPA 与活动功耗；
- 端到端收益。

## 9. 顶层代码审查发现的新缺口

现有 `qfit_source_multicast_term_builder` 只有一个 descriptor 上下文。扫描当前 source 的全部 lane/gate term 时，`descriptor_ready=0`，导致 relation frontier 无法发出下一 source。因此，当前完整顶层不具备单 bank TB 中的 lookahead 能力。

五 bank 不能简单复制五份 GASR。正确扩展至少需要：

1. 双上下文 descriptor builder，允许当前 source 发 term 时缓存下一 descriptor；
2. 从 relation frontier 的 active-index/read-issue 位置提取 geometry-ahead 旁带，在 K/gate SRAM 返回前给五 bank 目标地址；
3. 当前 source 最后一个 term 提交时，原子 activate 下一 source 的相关 bank；
4. 下一 source 未准备完成时，只反压 source 边界，不破坏 term 内连续发射；
5. 无目标的 bank 跳过 prepare/activate，保持自己的相关 source 序列；
6. 与 direct-1RW 五 bank 在相同前端、相同 SRAM 合同和相同 T450 100 组下 A/B。

## 10. 下一唯一优先级

实现真实五 bank 集成顶层：由 word-skipper、relation frontier 和双上下文 builder 在线产生 geometry-ahead source-major 流，同时驱动五个 GASR bank；回放 T450 profile100 和随机反压，报告 Acc32、端到端周期、bank stall、SRAM事务及逐组分布。

该结果决定 GASR-2C 能否从“离线理想流上的单 bank 驻留”升级为 DATE 可辩护的系统数据流贡献。
