# M535｜DATE 纸面加速指标独立复算与可引用性验证

日期：2026-08-27 CST  
问题：当前 H67 结果中，哪些倍率可安全进入论文，哪些仍需修订或只能带限定语分享？  
总体判断：**Needs revision（主表）/ Share with caveats（局部表）**。

## 1. 数据与方法

- M528 冻结结果：`m528_h67_single_port_same_ledger_recompute_result_r1.json`，SHA256 `778c8e1bed6a19852c14bc61e00761f798008d67042b7a74efbaaffdde4b3de1`。
- 顶会方法审计：M532 `README.md`/machine JSON，SHA256 分别为 `b1ac5e02...77aec` 与 `3af778cc...a59d`；内外 seal 复核通过。
- 按原始整数重新计算比值，不从四舍五入后的表格反推。
- 核对 scope、checkpoint/trace identity、资源分母、证据等级与禁止外推字段。

## 2. 复算结果

| 指标 | 原始整数复算 | 结论 |
|---|---:|---|
| M528 / M468 strongest-zero | `760,350,133 / 435,293,339 = 1.746753430105x` | 与收据一致，可进局部 exact CPU 表 |
| M528 / same-coordinate bit | `757,946,784 / 435,293,339 = 1.741232213066x` | 与收据一致，必须与上一分母并列 |
| M504 all-write / M528 dead-write | `456,016,645 / 435,293,339 = 1.047607680025x` | 只能作为 liveness 消融 |
| M472 official replay product / bit | `556,188,432 / 226,140,006 = 2.459487119674x` | 只能标 external official-artifact mapping |
| C2 K8 / 单端点 K1 | `429,716,335 / 90,196,785 = 4.764209001463x` | 低带宽 scaling；同页必须给 K1x8 |
| M528 240 KiB 容量余量 | `245,760 - 213,376 = 32,384 B` | 宏取整容量门通过 |

M528 的 `-41.91%/-63.57%` 只能用于 parent scratch logical access traffic；源收据明确写明它不是物理 SRAM/DRAM 能量。M528 自身 `date_headline/energy/rtl/vcs/synopsys_ppa/system_speedup` 均为 `false`，因此不能升为系统结果。

## 3. 高影响问题

1. **High｜系统主表仍无 admitted 行。** `1.794--1.823x` 仍是 decoder/memory 未闭合的 analytical envelope；摘要若提前使用，会把分析敏感性误写成周期仿真结果。
2. **High｜M528 尚缺功能与物理闭环。** 目前可用句子是“四层 bottleneck Conv、51.84M source-row、单序列的 exact CPU same-ledger 候选”；在 VCS/SVA、DC/STA、SRAM/DRAM 能量和 decoder-complete simulator 之前，不得写成 ours architecture headline。
3. **High｜C2 分母阶梯必须完整。** `4.7642x` 对单 K1 是合法的 iso-lane/低带宽 scaling，但隐藏等服务 K1x8 的约 `1.01--1.04x` 会造成资源不公平。最终价值应由 matched area、throughput/mm² 与 energy/source 决定。
4. **Medium｜外部结果必须隔离。** Prosperity/Phi 原论文数字以及 M472 `2.459487x` 可以进入相关工作和 external mapping 表，不能成为 H67 自研 RTL 分子。
5. **Medium｜有损主表为空。** PAFT valid825 的既有 `ΔAEE=0.0293279696` 未过 `0.02` 门；不能用 Phi/Bishop 的有损倍率代替 H67 自己的 accuracy-cycle identity。

## 4. 论文当前允许的三层表

- **Table A（系统）**：保留空位，等待 decoder-complete、memory-inclusive、至少三序列的统一 simulator 直跑结果。
- **Table B（本地机制）**：写 M528 `1.746753x/1.741232x`、C2 `4.7642x` 与 K1x8 companion、C3 directed VCS/待 matched PPA；每格带 evidence tag。
- **Table C（外部/映射）**：写 Prosperity/Phi/FireFly-T/Bishop 原报告值及 M472 official replay，明确 `not ours RTL/system`。

## 5. 可分享的限定语

当前最强且可复算的本地周期结果是：

> On 51.84M frozen source rows from four H67 bottleneck Conv3x3 layers, the single-port parent path reduces same-ledger cycles from 760.35M to 435.29M, yielding 1.747x over the strongest-zero schedule and 1.741x over the same-coordinate bit-sparse schedule. This exact CPU result remains pending RTL/PPA, energy, and full-network admission.

主表转为 `Ready to share` 的最小条件：同一冻结 H67 workload 直接重跑 Dense96/PTB-like/K1/K1x8/Ours，补齐 decoder、memory stalls、logic/SRAM/DRAM energy，并至少覆盖三条 sequence 或低/中/高 event-density 分层。

`docs/359_DATE终局冻结_20260813.md` 未修改，SHA256 应保持 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
