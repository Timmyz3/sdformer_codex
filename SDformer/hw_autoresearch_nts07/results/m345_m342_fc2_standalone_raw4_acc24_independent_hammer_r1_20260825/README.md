# M345：M342 FC2 standalone raw4→Acc24 独立打铁评审

结论：**80/100，P0/P1/P2 = 0/3/5。M342 作为 exact-SHA、directed、standalone raw4→Acc24 集成里程碑 GO；5.281374845× 只能带 bundled-memory 限定承认。same-bandwidth、physical、完整 FC2/FFN、system 与 paper headline 全部 NO-GO。**

我用冻结 filelist 和 seed `342025` 重新做了一次全新 VCS compile/simulation。四组非零结果完全复现：

| B | events | K8 cycles | K1 cycles | K8 speedup |
|---:|---:|---:|---:|---:|
| 1 | 20 | 42 | 160 | 3.809523810× |
| 2 | 41 | 112 | 602 | 5.375000000× |
| 4 | 90 | 410 | 2,566 | 6.258536585× |
| 8 | 110 | 1,027 | 6,235 | 6.071080818× |

几何平均为 `5.281374845×`。零事件两边均为 15 cycles。原 M342 `SHA256SUMS` 的 98 项全部复验通过；重跑没有 assertion failure、numeric mismatch 或日志 metric drift。

## 公平性与最强可承认指标

K8/K1 确实使用同一套确定性 raw bitmap、同一个 weight 函数、L4、O8/FIFO4、Acc24、result/done ready 公式和相同 memory-model RTL。M219 不是意外多出状态的慢服务：它与 M218 的控制骨架一致，只把 group/request/skid/adder 数据面裁成原生 K1。

但二者**不是等 aggregate memory bandwidth**：memory model 每拍只接受一个 bundled request；K8 一个请求可同时激活 8 个 128-bit bank word、返回最多 1,024 bit，K1 请求只激活并返回一个 128-bit word。四组非零测试的 active-bank work 都是 8,052 次；coverage 对应 K8 约 1,218 个请求、K1 8,052 个请求，K8 平均每请求合并 `6.610837` 个 bank read。因此最强合法表述是：

> Exact-SHA Synopsys VCS directed standalone raw4-to-Acc24 cycle geomean of K8 over K1 is 5.281374845× across B={1,2,4,8}, under one bundled request/cycle, eight logical 128-bit banks, L4, O8/FIFO4 and deterministic result/done stalls.

这说明打包优势在已连接 service、FIFO、Acc24 context 和 commit 下没有消失；但不能写成对“8 路独立 scalar issue、同 1,024-bit/cycle 带宽”基线的提升，也不是物理频率归一化结果。

周期口径是 header_accept 到 service token_done_accept，公式 `done-start+1`，含 raw backpressure、memory、FIFO、Acc24、result 与 done stall；不含 header 接受前等待、BN2/SN2/requant、FC1 或物理 SRAM。42/160、112/602、410/2566、1027/6235 都按这个 inclusive 定义复现。计数器存在一个 P2：start task 与另一个 posedge always block 共用 blocking `cycle_count`，VCS 确定复现，但最好改成单一 clocked monitor 消除调度竞争。

## 三个 P1

1. **带宽公平性。** 必须补 K1×8 独立 issue 或把 K8 1024-bit port 串行化，之后才能说 same-bandwidth/same-resource。
2. **M218/M219 SVA 实际没有运行。** 两个 assertion 文件被解析，但没有 bind/instance，elaboration 和 `assert.report` 只有 M342/M216。需要绑定后重跑 FIFO、outstanding、response/context、done-empty 和 stall stability。
3. **OOO 与跨 cap 事务等价证据不足。** `generation+1 < cumulative_requests` 不能证明 response reorder；active-bank 只检查总数，weight 函数也非单射。应增加真实 younger-before-older scoreboard，并逐项比较 K8 展开后的 `(block,slice,bank,channel)` 与 K1 multiset。

header mutual-ready fork/join、done bridge、full8、raw/request/result stall、两次 midflight POR、两次非法 header 和两次 spurious response 均已覆盖。但 M216 的 terminal-header-chain、partial-close、same-cycle done/load、bank-sum-48 和 local protocol-attack cover 为 0，应补 directed case 或正式 waiver。当前性能也只是四组 synthetic case，不是冻结 120-record FC2 trace replay。

## 新思下一步条件

- DC：K8/K1 分参数、同约束、同 effort；同时报告 3 ns QoR 与 Fmax，按实际频率和面积归一化。memory 先按显式 macro/black box，logic-only 不能称 paper PPA。
- Formality：在 service SVA 修复并通过 VCS 后，分别做 K8/K1 RTL→DC netlist equivalence，参数、black box、cut point 固定，0 unresolved/aborted。
- PTPX：用相同冻结 trace 和 stall 内容分别导 SAIF，报告映射覆盖；把 K8 1,024-bit 与 K1 128-bit 的 macro/I/O energy 纳入，比较 energy/token，不只比较平均功率。
- 在 physical/headline admission 前，必须补 frozen 120-record replay 和 equal-bandwidth baseline。

本评审未修改 M342、既有 RTL、合同或 `docs/359`，未运行 Verilator。
