# M148 independent hammer review r1

结论：**89/100，P0=0，P1=2，P2=4**。

M148 在它真正实现的窄范围内质量很好：它是一个 unsigned presence-only 的 `(destination, source)` K4 tuple packer。exact-SHA 生产 VCS/DC 均被 fresh replay；独立的 53.6 万行 exhaustive/random VCS hammer 没有发现 tuple 选择、顺序、tail、fallthrough、II1、stall 或 sequence quarantine 的功能错误。

但它不是 M147 所缺的 same-destination combine engine，也没有承载 signed/negate 元数据。因此 M148 不能把“descriptor 数量减少”直接变成 engine cycle 减少，不能挽救或接纳 M147 的 1.805434x 理想机会。

## 精确证据复现

- 封存 VCS 的 7 个输入、4 个输出，封存 DC 的 20 个 evidence 条目，以及 M147 independent review 的 4 个 immutable 条目全部通过 SHA256 校验。`docs/359` 仍是 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`，未修改。
- Synopsys VCS V-2023.12-SP1 fresh replay 精确复现生产 PASS：68 rows、1,901 events、506 mosaic descriptors、558 block-K4 reference descriptors、52 descriptor savings、83 stalls、419 II1 pairs、2 protocol attacks。8 项 cover 分别为 1/68/442/1/56/8/83/18，零 assertion failure。
- Synopsys DC V-2023.12-SP3 使用 exact RTL/filelist/SDC/Tcl 和同一 TSMC28 HPC+ 库重新综合，复现 2,183.957994 um2、3,575 cells、199 sequential cells、466 ports、53 logic levels、critical path 1.9234 ns、setup +0.6266 ns、hold +0.0002 ns、0 macro。
- Fresh mapped Verilog 与 seal 仅创建时间不同；area 数值完全一致，QoR 仅时间戳和编译耗时不同。Mapped DDC 重新打开后面积、cell 数、setup/hold 状态也一致。
- Fresh DC 和 DDC audit 均无 TIM-209、OPT-150、ELAB-312、Error 或 Fatal；`check_design=1`、`check_timing=1`，五类 constraint 无 violation。

## 独立 exhaustive/random VCS hammer

独立 TB 共验证 535,622 行、4,470,941 个 source events 和 1,318,990 个 descriptors：

- 524,288 行：8 个 destination 分别穷举全部 65,536 种 16-bit source mask；
- 7,168 行：28 个无序 destination pair 各自穷举全部 `16 x 16` source pair；
- 4,096 行：确定性生成的完整 128-bit 随机 dense/sparse mask；
- 额外 directed 行：all-zero、all-one、bit127、同 destination、跨 destination、tail1/2/3、fallthrough stall、active stall 和 64-row II1 burst。

每一行都用独立 linear-first-four oracle 核验：

- 全部输出 tuple 与输入 set-bit 的 `(destination, source)` 有序多重集完全一致；
- descriptor 数严格等于 `ceil(total_popcount / 4)`；
- 每行和全局 event presence 均无丢失、重复或 dirty padding；
- zero row 只占一次 row acceptance、无 descriptor，并在该接受沿产生 done；
- 第一 descriptor fallthrough，连续 ready 时 descriptor II=1；inactive fallthrough 和 active 两种 stall 下 payload 均稳定；
- wrong initial sequence 与 active wrong sequence 均 sticky quarantine，正确 next sequence 可在 active row 后等待并接受，reset 可清 fault/active 并从 sequence 0 重启；
- 单 outstanding 模式下 `ffffffff -> 00000000` modulo wrap 正常。

## P1-1：packer 不是 combine engine

M148 只把至多四个 tuple 装进一个 descriptor；它没有 destination accumulator、coefficient arithmetic 或同 destination 多更新合并路径。一个 descriptor 中四个 tuple 指向同一 destination 时，M148 本身没有让下游在一个 cycle 内完成四次更新。

M147 correction overlay 已证明这是主导问题：47,037,211 个 mosaic descriptors 中，35,725,177 个含重复 destination，占 75.95%；9,918,824 个四个 tuple 全指向同一 destination，占 21.09%。没有 same-destination combine 时需要 137,150,654 cycles，相对 M143r2 B4 只有 0.987680x，即约慢 1.247%。所以：

**M148 的 `ceil(popcount/4)` 是 descriptor packing 结论，不是 engine-cycle 结论；它不能接纳 M147 的 1.805434x。**

真正的下一模块必须是可综合、非饱和的 same-destination combine/update engine，证明 accumulator 位宽和 overflow，并把实测 ready/stall recurrence 接回 M147。

## P1-2：signed/negate 仍未承载

M148 输入只有 `row_event_mask[127:0]`，tuple 输出只有 destination、source 和 valid，没有 `row_negate_mask` 或 per-tuple sign。当前 exhaustive 证明因此仅是 presence conservation，不是实际 signed parent-delta 的数值等价。

同样，2,183.96 um2 的 DC 数字没有包含 sign selection/storage；M147 的 17.56% descriptor slot-bit lower bound 也明确未计 sign、valid、count、alignment 和 ECC。需要增加 128-bit negate mask 与严格对齐的 per-tuple sign，再重跑 tuple+sign oracle、DC 和下游 arithmetic proof。

## DC 与接口边界

DC setup 余量不错：关键路径 `active_q_reg -> descriptor_source[3][3]` 为 1.9234 ns，在 3 ns 合同下有 +0.6266 ns。面积、sequential cell 和 logic-level 均通过合同阈值。

但它仍是 ideal clock、ZeroWireload、无 routing、无 macro、无 power 的 logic-only cut；hold 只有 +0.0002 ns。zero row 的 done 与 row acceptance 同沿且没有 done-ready，mid-active reset 会直接取消 residual tuples，这些必须作为集成合同冻结。当前也没有 M148 RTL-to-netlist Formality seal。

## Findings

- **P0 (0)：** narrow unsigned tuple-packer scope 内无阻断功能错误。
- **P1 (2)：**缺少 same-destination combine，M147 1.805434x 仍不成立；缺少 signed/negate 元数据，尚非实际 workload 等价模块。
- **P2 (4)：** production SVA 未直接表达完整 multiset conservation/descriptor recurrence；无 Formality；DC 仍 prephysical、无 macro/power；same-edge done 与 reset cancellation 需冻结集成合同。

## 建议下一步

1. 不再继续优化 presence-only packer 的 descriptor 数；M148 已经把该窄问题做对。
2. 优先实现 same-destination combine/update engine：对 descriptor 内 destination 分组，合并相同 destination 的 signed coefficient contribution，证明非饱和 accumulator 位宽和 overflow。
3. 同时给 M148 增加 negate/sign sideband；否则 combine engine 无法恢复实际 signed arithmetic。
4. 用 M147 heldout descriptor stream 测量真实 combine occupancy、ready/stall 和 cycle recurrence。只有实测结果优于 135,461,009-cycle M143r2 B4，才恢复 incremental speedup 讨论。
5. 再跑 exact-SHA Formality、matched-frequency PT/PTPX。当前 1.805434x、4.6845x 和 3.2719x 均不能作为 M148 的加速结果。

机器可读结论见 `m148_independent_hammer_review_r1.json`；本目录核心证据由 `manifest.sha256` 固定。
