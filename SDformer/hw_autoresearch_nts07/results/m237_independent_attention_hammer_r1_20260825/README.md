# M237 独立注意力打铁评审

结论：RQTB 可以作为注意力子模块的一项硬件贡献，当前评分 **81/100，P0=0、P1=6、P2=4**。它已经完成匹配的冻结 VCS 与 Synopsys DC A/B，但不能作为全注意力、全网或 paper-ready PPA headline。

## 这次真正闭合的结果

- 模块边界：MSSB5 精确 Q/K score front → 可逆 T=2 quotient/slot FIFO → 163-class weighted directory → Shiftmax → 双同步 K-bank → gated-K 输出。
- 唯一 A/B 参数差异：`QUOTIENT_ENABLE`，Fixed=0、RQTB=1。两边均为 HEAD_DIM=32、PAIRS=225、FIFO=32、MSSB5=1、MEMORY_IMPL=0。
- VCS V-2023.12-SP1，H67 ep35 sample0，12 个 attention block、138 个 head、每个 head 一个冻结 T450 窗口：
  - Fixed 112,589 cycles；RQTB 94,891 cycles；**1.18651×**。
  - slot 62,100 → 34,099，减少 **45.09%**；28,001/31,050 pair 的两时刻精确 score 相等。
  - production RTL 与 DC-compatible snapshot 均逐输出通过，0 mismatch、0 assertion failure。
- DC V-2023.12-SP3，TSMC 28nm HPC+ NLDM，3.0ns，同 SDC/max/min library：
  - Fixed：134,076.60 µm²，123,796 leaf / 26,024 sequential，setup +0.0008ns，hold +0.0095ns。
  - RQTB：135,760.46 µm²，125,761 leaf / 26,157 sequential，setup +0.0008ns，hold +0.0099ns。
  - RQTB 面积开销 **1.2559%**；面积归一吞吐 **1.17179×**。
  - 两边 17/17 工件齐全、0 constraint violation、macro count=0。

## 最强可辩护创新

“无损的 equality-conditioned temporal quotienting”：当 T=2 的两个精确 Q/K-derived score 相等时，只向 class/Shiftmax 服务提交一次 score/class transaction，同时保留两个 temporal identity mask 与独立 gated K read。它不是近似剪枝，也没有改变注意力输出。

算法侧最值得反哺的 KPI 是 exact temporal equality。当前 90.18% pair equality 直接对应 45.09% slot reduction。后续可以尝试 accuracy-guarded Q/K 量化或 temporal-consistency regularizer，但必须同时报告精度、equality fraction 和预测 cycles，不能只优化硬件代理。

## 不能越界的口径

- 这是组件 RTL speedup，不是全注意力或全网实测。冻结账本中该 core 约占 fixed envelope 的 0.5889%，代入后的 modeled full-network envelope 只有约 **1.00091×**。
- `MEMORY_IMPL=0` 把 K-bank、FIFO、class histogram、descriptor store 都综合成触发器/逻辑；因此绝对面积和能耗不是 SRAM PPA。
- setup margin 仅 0.0008ns，且是 ideal-clock/ZeroWireload DC，不是布局布线后频率。
- 尚缺两边 Formality、matched SAIF/PTPX、macro-aware PPA 与 exact-MSSB5 10-sample generalization。

## 下一步优先级

1. 同容量/同端口/同延迟宏合同下重做 Fixed/RQTB A/B，并跑提取后 PrimeTime。
2. 两边 RTL↔netlist Formality；同时形式化证明 production MSSB5 与兼容 generate 写法等价。
3. 同 138 行生成匹配 SAIF，跑 PTPX，报告 energy/row、energy/token 与 EDP。
4. 用 MSSB5=1 重跑 10-sample/1380-row，并补 no-stall、stress-stall、all-window sensitivity。
5. 零输出 row bypass 若要合并，作为 RQTB 之后单独的增量 ablation；不能把 shared 的 87,034 cycles 归因于 RQTB。

核心合同见 `m237_attention_rqtb_matched_ab_contract_r1.json`，独立重算见 `m237_independent_recompute.json`，打铁 findings 见 `m237_independent_hammer_review_r1.json`。
