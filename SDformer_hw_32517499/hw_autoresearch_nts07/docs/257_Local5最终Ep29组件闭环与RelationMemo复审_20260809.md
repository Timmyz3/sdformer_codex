# Local5 最终 Epoch29 组件闭环与 Relation Memo 复审

> 状态更新：本文表 130 行所述“score→relation→term→projection 单顶层未完成”
> 已由 `docs/265_Local5全链数值合同与可控反压单顶层审计_20260809.md`
> 在通用 RTL 结构范围内关闭，并补三组固定种子 term 反压。equal-plus10 最终
> checkpoint、同窗 all-head fullres trace 与 12-block 调度仍未关闭，本文历史
> checkpoint 证据边界不变。

## 1. 本轮结论

Local5 当前可以冻结为**算法候选已通过、三个关键硬件组件分别 bit-exact、端到端部署尚未签核**。

- `[prof]` 最终候选为 epoch29：AEE 1.3286、AAE benchmark 5.6594、总脉冲 82.8799G、firing 6.4480%、spike-energy proxy 73271.32 uJ。
- `[prof]` post-G0 接受审计覆盖 100 个样本、12 个 attention block，13/13 项绑定检查通过。
- `[rtl]` checkpoint-bound score/Shiftmax5、投影部分累加器和 ATLIF DP-TME 三条组件链均通过整数金参考；这三条结果尚未串成包含跨 head 求和、bias、动态 BN、requant、residual 和 decoder 的全网络 RTL。
- `[prof]+[模型]` 用最终 epoch29 trace 重跑 7 KiB exposure-aware Relation Memo 后，整帧周期代理为 1.342x，旧 checkpoint 的 1.333x 结论没有失效。
- `[rtl]` 公平五 bank GASR 路径把 SRAM 事务减少 80.03%，但周期为 109730，对照 Direct 的 109230，实际慢 0.46%。因此 GASR 不能作为当前周期加速贡献。
- `[待验证]` 当前最重要的 Local5 架构门槛不是再扩一套 bank，而是取得同一 window 的完整 head 联合分布，并把 Relation Memo 的 pack/replay、fallback 和现有 relation 宏端口在最终 trace 上闭合。

## 2. 最终算法与输入绑定

| 项目 | 结果 | 证据 |
|---|---:|---|
| checkpoint | epoch29，SHA256 `6e0e92a56229f72f77b2868911f087bf3575e67ab0f2ef4752bf166cdcec993b` | `[prof]` |
| 分辨率 | 480x640，crop=null | `[prof]` |
| window | 2x15x15，T=450 | `[prof]` |
| 模块覆盖 | ATLIF 105、Shiftmax 12、overlay 210、missing/unexpected 0 | `[prof]` |
| post-G0 接受 | 100 samples、12 blocks、13/13 checks | `[prof]` |
| AEE | 1.3286 | `[prof]` |
| AAE benchmark | 5.6594 | `[prof]` |
| total spikes | 82.8799G | `[prof]` |

算法排序来自：

`../neuron_experiments/H9_bipolar_self_attention/results/dsec_fullres_w15_H66d_local5_bb1e4_ft30_20260805/profile_ranking_valid825.md`

接受审计来自：

`results/local5_fullres_bb1e4_postg0_acceptance_20260805/acceptance.json`

## 3. 三条 checkpoint-bound 组件证据

### 3.1 Score 与 Shiftmax5

- `[rtl]` 100 groups、45000 个 score/gate 向量。
- 独立 trace 金参考、Icarus、Verilator 和 Yosys 均通过。
- 证据边界只到 post-G0 Q/K score 与 Shiftmax5，不包含 relation transpose、投影或全网络。

报告：`results/local5_bb1e4_checkpoint_score_shiftmax_rtl_20260805/report.json`

### 3.2 Source-major term 到投影 Acc32

- `[rtl]` 100 组 Direct 和 100 组 GASR，44883 terms、137179 updates。
- checkpoint theta 已折叠到 dyadic INT8 projection weight；运行时 K 仍为 1-bit event。
- 100/100 组 Acc32 逐元素零失配。
- 未覆盖跨 head 完整 C 维求和、bias、动态 BN、requant、residual 和 decoder。

公平周期与事务结果：

| 路径 | 周期 | 相对 Direct | SRAM 事务 | 相对 Direct |
|---|---:|---:|---:|---:|
| Direct | 109230 | 1.000x | 261827 | 1.000x |
| GASR | 109730 | 0.995x | 52287 | -80.03% |

该负结果说明：五 bank relation 压缩显著减少了存储访问，但同步、term stall 和发射开销抵消了周期收益。它是否降低能耗只能由 SRAM macro 与 SAIF/PTPX 证明，不能从事务数直接外推。

报告：`results/local5_bb1e4_qgasr2c_fivebank_postg0_rtl_20260805/report.json`

### 3.3 ATLIF DP-TME

- `[rtl]` 81 条真实 checkpoint 命令，其中 T10 45 条、T2 36 条。
- 25920 个 hidden 和 25920 个 event 相对整数金参考均为零失配。
- 覆盖 input/output backpressure、SVA、Icarus、Verilator、lint 和 Yosys check。
- `[prof]+[待验证]` 静态 INT8 数值桥相对浮点有 1177/25920，即 4.5409% 的局部 event flip；因此 `deployment_accuracy_signoff=false`。

这里的正确表述是“ATLIF 整数合同 RTL-exact”，不是“量化网络精度已签核”。完整 valid825 还必须采用静态 site scale 与下游 event-times-threshold folding 重新推理。

报告：`results/local5_bb1e4_checkpoint_atlif_dptme_rtl_20260805/report.json`

## 4. 跨仿真器 TB 竞态修复

原 ATLIF testbench 在 `posedge` 后读取 `step_ready`。Icarus 看到 NBA 更新前的 ready，而 Verilator 看到状态更新后的 ready，导致首拍已被 DUT 接收后，testbench 仍等待下一次 ready，表现为 Verilator 长时间 100% CPU 且日志停在 command 0。

修复后的驱动合同是：

1. 只在 `negedge` 驱动 payload 与 valid。
2. 在同一相位等待 ready。
3. 只跨越一个明确的 `posedge` 完成接收。
4. 在该边沿后延迟 0.1 ns 撤销 valid，消除 active/NBA 竞态与重复接收。

修复后 Icarus 与 Verilator 均在 81 命令、25920 hidden/event 上零失配完成。该修复只提高验证可信度，不是论文架构贡献。

## 5. 最终 Epoch29 Relation Memo 复审

旧 DSE 使用 checkpoint `e7da...`。本轮改用最终 epoch29 manifest 与 payload：

- manifest SHA256：`c643a04476f75e351ea7874e3a19f56ed0183b8cfa597723358ebae0509b6201`
- payload SHA256：`2c98796d3c36afabdf7676c1ac205bc56c19c1a1135877c0054e99f108545810`
- 100 samples、4800 个真实 per-head groups、full-resolution T450。
- 单测 7/7 通过，DSE 每个配置使用 20000 次 bootstrap。

7 KiB critical-only admission 结果：

| Stage | 驻留 head 比例 | 全 head fit rate | 周期代理加速 | relation build 减少 |
|---:|---:|---:|---:|---:|
| S0 | 79.76% | 92.22% | 1.341x | 53.17% |
| S1 | 96.30% | 99.60% | 2.466x | 80.25% |
| S2 | 69.12% | 5.83% | 1.392x | 63.36% |
| S3 | 38.93% | 0.00% | 1.114x | 37.30% |
| 全帧加权 | - | - | 1.342x | 59.01% |

结果文件：`results/local5_relation_vault_dse_final_ep29_20260809/report.json`

### 5.1 可以保留的架构主张

Relation Memo 的主张不是 SRAM、ping-pong 或缓存本身，而是 Local5 的 relation 在 output tile 维度保持不变后，首遍生成的 exact topology operand 可以跨 tile 重放；admission 只缓存 `projection service < 450`、即 relation build 暴露在关键路径上的 head。容量 miss 和非关键 head 均精确回退重算。

这相当于把 Bishop/Phi 的 workload stratification、Prosperity 的 exact reuse 和 TTB/STT 的 stream packing 本土化为：**由实测 latency exposure 决定驻留的 Local5 五邻域拓扑操作数重放**。

### 5.2 仍不能声称的内容

- 当前联合 head 分布来自真实 per-head group 的 bootstrap，不是同一 sample/block/window 的全 head 联合采样。
- 1.342x 是强基线下的周期模型，不是 RTL 周期、ASIC PPA、功耗或 EDP。
- 现有 Relation Memo RTL 已通过混合 tile exact miter，但尚未绑定最终 epoch29 的 same-window all-head trace。
- relation 宏端口与 FCSR/TCFM5 的最终共享、随机反压和 fallback 时序尚未形成 checkpoint-bound 单顶层证据。

## 6. 当前 Local5 完成度

| 层级 | 状态 | 判定 |
|---|---|---|
| 算法 fullres valid825 | 已完成 | `[prof]` |
| 12-block/module/checkpoint 绑定 | 已完成 | `[prof]` |
| score/Shiftmax5 组件 | 已完成 | `[rtl]` |
| source-major term/投影部分 Acc32 | 已完成 | `[rtl]` |
| ATLIF DP-TME 整数组件 | 已完成 | `[rtl]` |
| score→relation→term→projection 单顶层 | 未完成 | `[待验证]` |
| ATLIF 静态量化 valid825 | 未完成，局部 flip 4.54% | `[待验证]` |
| 12-block 时间复用调度 RTL | 未完成 | `[待验证]` |
| SRAM macro、DC/STA/SAIF/PTPX | 未完成 | `[待验证]` |
| full encoder/full network RTL | 非当前已证范围 | `[待验证]` |

## 7. 下一轮唯一晋级门槛

在继续扩展 Local5 RTL 前，先补**最终 epoch29、同一 sample/block/window 的完整 head 联合 trace**。用它完成：

1. 实测 7 KiB 容量下的 joint occupancy、critical-head 命中率与 fallback 比例。
2. 用完全相同的 head/window 顺序比较 recompute 与 pack/replay 周期。
3. 若实测仍有不低于 1.15x 的强基线收益，再把最终 trace 接入 Relation Memo 单顶层，做容量边界、随机反压与 Acc32 miter。
4. 若低于门槛，Relation Memo 降级为存储能耗候选，不再扩大 RTL；主线回到更高收益的数据流机制。

Motion 线不受该门槛暂停。H67 fullres T450 profile 与 RQTB follower 已并行运行，完成后独立决定 Motion 的新机制，不使用 Local5 的结论替代 Motion 证据。
