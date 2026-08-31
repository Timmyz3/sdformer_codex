# M229 independent hammer review

**Score: 84/100. P0: 2. P1: 7. P2: 4.**

M229 确实修掉了 M227 的核心结构缺陷。它不再把每个 source 强制放进互斥的 `REQUEST -> WAIT -> REPLAY` 三段状态，而是用四个物理 credit 将 descriptor、weight request/response 和当前 replay 解耦。独立重算两组 clean directed case 后，F1/F2/F4 都精确等于“所需 replay beat + 每组固定 4-cycle fill/drain”：

| variant | clean replay beats | observed clean cycles | fixed overhead | speedup vs F1 |
|---|---:|---:|---:|---:|
| F1 | 111 | 119 | 8 | 1.000000x |
| F2 | 62 | 70 | 8 | 1.700000x |
| F4 | 38 | 46 | 8 | 2.586957x |

F1/F2/F4 分别命中 52/52/53 次同拍 `weight_req_accept && acc_update_accept`，而且 full-credit、full-fanout、request stall、update stall、fault、done cover 均非零。因此，“M227 每个 source 多出两个串行非更新周期”的问题，在当前一拍响应、directed service-island 合同下已经真正修复，不只是 premodel 修辞。

功能证据也成立。三档各自处理 64 个 descriptor、276 次 context update、三种攻击；INT8 signed add/sub 对 96 lane 逐值比对，最终八个 context bank 全量核对；descriptor/request/response 守恒为零 mismatch。wrong-source response 被拒绝，Acc19 overflow 当拍禁止 write accept 并进入 sticky fault。VCS 175 个 seal 条目、DC 顶层四个 manifest 条目，以及当前 input/runner receipts 均独立重哈希为零 mismatch。

## Layered verdict

| scope | verdict |
|---|---|
| M227 serial-bubble repair | **GO** |
| exact-SHA directed VCS / arithmetic / conservation | **GO** |
| credit、overlap、stall、identity、overflow fail-close | **GO，限 directed coverage** |
| TSMC28 3 ns matched DC | **GO，限 pre-macro logic feasibility** |
| `1.7x / 2.586957x` | **GO，限两组 clean directed service island** |
| throughput/area | **PROVISIONAL diagnostic only** |
| H67 100-record FC1 trace | **NO-GO，未执行** |
| complete FC1/FFN/system/headline | **NO-GO** |
| paper PPA / energy | **NO-GO** |

## Independent DC recompute

| variant | cell area | cells | seq cells | setup receipt | hold receipt | ports |
|---|---:|---:|---:|---:|---:|---:|
| F1 | 18,219.222 µm² | 18,403 | 3,402 | +0.0025 ns | 0.0000 ns | 4,813 |
| F2 | 24,013.206 µm² | 23,421 | 3,402 | +0.0682 ns | 0.0000 ns | 8,465 |
| F4 | 35,715.078 µm² | 33,001 | 3,402 | +0.0242 ns | 0.0000 ns | 15,769 |

以当前 **island-only** cell area 直接相除，F2/F4 面积为 F1 的 `1.318015x/1.960297x`，directed service throughput/area 为 `1.289819x/1.319676x`。这些复算与生产收据一致，但不能叫 physical throughput/area。

原因是外部 Acc port cut 正在吞掉随 FANOUT 增长的硬件。每增加一个 fanout slot，DC 顶层端口恰好多 **3,652 bit**：`1824 read + 1824 write + 3 context + 1 valid`。TB 用 `bank[acc_update_context]` 免费完成了八个 bank 到 F 个 arithmetic slot 的选择；实际 8-bank read mux/write demux、bank enable、布线或等价的 bank-local datapath 尚未综合。共同的 14,592-bit Acc19 容量也未计，M228 presence/sign 状态、weight SRAM 容量/decoder/768-bit port、latency 和 energy 同样未计。

共同容量加入后，归一化 throughput/area 数学上可能提高，因为共同面积会稀释 lane 增量；例如把 14,592 bit 暂按本次 sequential-cell area/bit 折成 29,417 µm²，F2/F4 的 directed sensitivity 会变成约 `1.516x/1.892x`。这只是 common-area sensitivity，不能抵消遗漏的 fanout-scaled mux/demux、布线和宏端口能耗，也绝不是 paper PPA。

## P0 findings

1. **M229-P0-01 — 尚无可执行的 100-record recurrence。** 当前 clean server 是 always-ready、一拍 response，descriptor 已经由 TB 构造好。`1.7x/2.586957x` 没有收费 M228 mask construction、active-chunk walker、group boundary、有限 SRAM latency/credit 占用，也没有保持冻结 H67 100-record 的真实顺序。下一门必须把 first-fill、request、response bubble、replay、scan、drain 和 M225 既有 overhead 分项计入，并报告 per-record min/mean/max。
2. **M229-P0-02 — throughput/area 的物理边界不完整。** 14,592-bit accumulator capacity 与 fanout-scaled bank plumbing 都在 port cut 外，mask/weight SRAM 也未进入 DC/PTPX。必须加一个 matched eight-bank Acc19 wrapper，在 F1/F2/F4 中保持相同容量与宏合同，同时显式综合各自的 read-select/write-demux 或 bank-local 结构；之后才能以 trace throughput/area 选 F2/F4。

## P1 findings

1. VCS 只有三组 deterministic case；合法 out-of-order response、延迟大于一拍、response bubble 和随机 trace replay 未覆盖。
2. wrong identity 只攻击 source；wrong slot/tag/epoch、duplicate、response-before-request、同拍 zero-latency response 未分别命中。
3. done backpressure、full-queue 同物理 slot pop+push、busy reset、last 边界组合未覆盖。
4. DC 是 ideal clock、ZeroWireload、0 macro；F1 setup 余量仅 2.5 ps，hold 收据为 0，F4 有 75 logic levels 和 high-fanout warning，不能外推 routed 3 ns。
5. 尚无 mapped-netlist Formality、gate VCS、PT STA、SAIF/PTPX。
6. 生产 contract 仍保守标为 `admission.vcs=false/dc=false/throughput_per_area=false`；证据已能独立准入 service island，但 admission metadata 尚未收口。
7. DC 顶层 manifest 只封三个子 manifest 与 `RUN_COMPLETE.txt`，没有把 input/preflight/runner receipts 自身纳入顶层链。本 review 已把当前身份重新封存，但生产 seal 应自包含。

## P2 findings

1. `cp_fault=7` 是三个攻击后的 sticky fault 周期，不是七类攻击。
2. `ap_update_stable` 把外部输入 `acc_read_data` 的稳定性写成 assertion；正式 wrapper/formal 应把它转成 bank 保证或 assumption。
3. DC 对 synthesis-time geometry `$fatal` 报 `VER-104` 并忽略；本次 exact runner 参数正确，但 synthesis fail-close 还应由 runner/elaboration check 保证。
4. 合法 response 永远 ready；只有在 fixed-latency SRAM contract 明确且可执行后，这个接口假设才可接受。

## Minimum next gate

下一里程碑应是 **M230 trace-executable bank-complete closure**：

1. 将 M228 改成 sparse active-chunk / generation-valid producer，只发非空 chunk，避免 broad clear 和 12-chunk 固定扫描吞掉 F4 收益。
2. 用四 credit 与参数化固定 SRAM latency 逐条执行冻结 H67 FC1 100-record descriptor stream，给出 scan/request/fill/replay/drain 分项。
3. 接入显式 eight-bank Acc19 adapter，并对三档使用相同 14,592-bit 容量、weight SRAM 和 mask SRAM 宏合同。
4. 只依据 trace throughput/area 与 energy 选择 F2 或 F4；胜点再跑 exact VCS、DC、Formality、PT、SAIF/PTPX。

在两个 P0 关闭前，M229 可以作为有创新性的 **credit-decoupled held-weight service island**，但不得写成 FC1、FFN 或系统加速，也不得把 `1.7x/2.586957x` 放进摘要。本文 review 没有修改生产 RTL/脚本、论文或 `docs/359`。
