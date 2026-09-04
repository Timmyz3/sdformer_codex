# M102 bit-sparse physical baseline 前置独立打铁 r1

日期：2026-08-24  
边界：只读审计 M78/M88 周期模型、已有 production RTL 和 M101 claim boundary。未修改生产 RTL、合同、结果或 `docs/359`，未运行 VCS/DC。

## 结论

评分 **72/100**，`P0=2 / P1=4 / P2=3`。

**当前 NO-GO：仓库中没有可以直接作为 M88 `bit_sparse_cycles` 物理分母的同范围 standalone RTL，也不能把 M88 的 `1.409375695x` 乘以 M85/M99 的同功能频率比。**

**后续 GO：按本评审给出的最小 fixed-INT8 96-lane weight-service 规格实现 M102 baseline，同时构建包含 PWP 与 correction/fallback weight path 的 candidate service top，再做同流 VCS/Synopsys A/B。**

只做 baseline RTL、然后拿它与 M99-alone Fmax 相除仍然不够。M99 是 PWP metadata/map/unpack logic island；M88 candidate cycles 还包含 188,148,490 个 correction weight vector-op、58,969,374 个 PWP vector-op、block escape，以及解析/准备开销。物理 numerator 必须覆盖相同服务集合。

## `bit_sparse_cycles` 的精确定义

M78 从 5 个 valid825-internal heldout samples、4 个算子、每算子 432 个 phase、每个 16-source partition 的真实 activity mask 出发。对每个 mask：

1. `popcount(mask16)` 是每个 output block 的 active-source vector-op 数；
2. 共有 8 个 output block，每个 block 有 96 个 INT8 output-lane weights；
3. SHARED_32B 模型每个 vector-op 读取 `96 lanes x INT8 = 96 B`；
4. 逻辑端口每拍 32 B，所以固定收费 **3 cycles/vector-op**；
5. 输入已经是 active source 的紧凑 vector-op，扫描 16-bit mask/枚举 source index 本身没有另收费；
6. 每个 phase 加 2-cycle compute tail，每个 sample 首 phase 加一次 12,288 B weight image 的 384-cycle preload；后续 phase preload 与 compute 取 `max()` 重叠。

冻结总账为：

| 项目 | 数值 |
|---|---:|
| active-source vector-op（8 blocks 后） | 371,461,096 |
| weight service cycles | `371,461,096 x 3 = 1,114,383,288` |
| 5 次初始 weight preload | `5 x 384 = 1,920` |
| phase tails | `5 x 1,728 x 2 = 17,280` |
| M78/M88 `bit_sparse_cycles` | **1,114,402,488** |
| on-chip weight read bytes | 35,660,265,216 B |
| shared-DRAM weight bytes | 106,168,320 B |
| one / two weight phase buffers | 12,288 / 24,576 B |

M88 对 baseline 使用两个有限 weight slots，但所有 phase 的 compute 都远长于 384-cycle preparation，因此 midstream refill stall 为 0，最终 denominator 与 M78 完全相同。它不是 VCS cycle count，而是 always-ready、precompacted-source、32 B/cycle port model。

## candidate 的对应收费

cap11 SHARED_32B 的真实工作为：

| 服务 | ops | cycles/op | service cycles |
|---|---:|---:|---:|
| signed correction/fallback weight | 188,148,490 | 3 | 564,445,470 |
| PWP width8 | 11,164,284 | 3 | 33,492,852 |
| PWP width9 | 32,360,036 | 4 | 129,440,144 |
| PWP width10 | 13,936,011 | 4 | 55,744,044 |
| PWP width11 | 1,509,043 | 5 | 7,545,215 |
| **合计** | 247,117,864 | — | **790,667,725** |

M88 再加每 phase 2-cycle tail、2-cycle synchronous fill 和每 sample 838-cycle startup，得到 bounded candidate `790,706,475` cycles。于是：

- service-only ratio：`1,114,383,288 / 790,667,725 = 1.409420484x`；
- M88 bounded ratio：`1,114,402,488 / 790,706,475 = 1.409375695x`。

这两个数很接近但不能混用。前者适合 port-cut vector-service island；后者只有在 DMA/parser/writer/matcher/packer/controller 也进入对应物理 top 或被证明不降低 Fmax 时才可与 physical Fmax 组合。

## 仓库 RTL 盘点

只读 inventory 覆盖 `rtl*` 下 276 个 RTL 文件、284 个 module。没有 module 同时实现：

`precompacted active source + output block -> 3 x 256-bit weight service -> 96 x signed12 output`

并绑定 M78/M88 的 source/block/phase ledger。

最接近的现有代码均不能直接充当 denominator：

- `rtl_m79/precision_elastic_pwp_beat_assembler.sv`：fixed width8 时确实能把三个 256-bit beat 组装成 96-lane signed12，但没有 16-source x 8-block weight address mapper；其 command/beat 协议原本也有额外 command cycle。
- `rtl_m82/zero_bubble_elastic_pwp_stream.sv`：descriptor-on-first-beat 后可达到 fixed8 II=3，是最适合复用的公共 assembler，但仍没有 weight mapping/service top。
- `rtl/sparse_mac_pe.v`：默认 8 lane，语义是每 lane 一个 spike/weight 累加器，不是一个 active source 广播到 96 output lanes，也没有 32 B x 3 beat service。
- `rtl_hitflow/gatestack_hatf96_weight_coalescer.sv`：三个 32-lane bank 并行返回 96 B，等效带宽是 96 B/cycle，不是冻结的 32 B/cycle denominator。
- M85/M99：只服务 PWP record，escape 仍是零 payload control；明确不实现真实 bit-sparse fallback/accumulator ordering。

## 最小且公平的 baseline RTL 规格

建议 top：`m102_bit_sparse_weight_stream`。

### 冻结几何与接口

- `SOURCES=16`、`OUTPUT_BLOCKS=8`、`LANES=96`、`WGT_W=8`、`OUT_W=12`；
- synchronous active-high `rst_core`，与 M99 相同的 `clk_core`；
- beat-level request：`lookup_valid/ready`、`source[3:0]`、`block[2:0]`、`beat[1:0]`、`tag[31:0]`；
- port-cut memory：`bank_words[255:0]` input、`bank_row_addresses[8*10-1:0]` output；
- output 与 M99 完全一致：`output_valid/ready/tag/width/escape/values[96*12-1:0]/accept`，另有 `protocol_error/busy`；
- request 明确是 **precompacted active-source op**。mask scanner/popcount/enumerator 若不进入 top，必须在 claim 中继续列为 port cut。

### 地址与状态机

- 每 phase weight image：`16 x 8 x 96 B = 12,288 B = 3,072 words`；
- 8 个 32-bit banks，共 3,072 个 32-bit words；每 bank 384 rows，也对应每 phase 384 个 logical 256-bit beats；
- `base_word = (source x 8 + block) x 24`；
- `logical_word = base_word + beat x 8`，用与 M85 相同的 bank rotation 产生八个 row address；
- beat 必须严格为 `0,1,2`，source/block/tag 在 continuation 中锁定；非法序列 fail-closed；
- 三个 beat 组装成 96 个 signed INT8，逐 lane sign-extend 到 signed12；`output_width=8`、`output_escape=0`；
- 复用 M82 的 descriptor-on-first-beat elastic semantics，使 always-ready output 下相邻 vector start II 精确为 3，且 output stall 时 payload 稳定。

这个 top 不应内建 24 KiB 双 buffer 寄存器阵列；先与 M99 一样使用 memory port cut。宏容量/端口另做一张 ledger。

### VCS admission

- directed：source 0/15、block 0/7、正负/`-128`/`127`、连续 II=3、随机 output backpressure、非法 beat order/identity mutation；
- actual：按 M78 frozen digest replay 全部 371,461,096 op，或使用 phase-level SHA/counter 压缩回执并抽样 bit-exact data；
- 断言：无双接收、continuation identity stable、beat index monotonic、exact three-beat completion、output stall stable、96-lane signed extension、fault sticky/reset recovery；
- counters 必须精确回执 vector-op、beat=`3xop`、phase、source/block population 和 stall。

## 必须同时补的 candidate numerator

物理吞吐不能拿 baseline top 直接对 M99-alone。至少需要 `m102_combined_candidate_service_top`：

- 同一 command stream 区分 `WEIGHT` 与 `PWP`；
- WEIGHT 路径复用上述 3-beat baseline mapper，承担所有 correction 和 362 次 block-local escape 的真实 fallback work；
- PWP 路径使用 M99 的 3/4/4/5 beat map/unpack；
- 两类访问在 aggregate 上严格共享一个 256-bit service slot/cycle，不允许因两套 memory ports 偷偷并行；
- 同一 96x12 output/backpressure/tag contract，且 replay counters 精确等于 188,148,490 correction ops 与各 width PWP populations；
- 如果 matcher/packer/enumerator 不在 top，结果只能叫 `vector-service-island throughput`。

## Synopsys 与 SRAM 公平口径

### logic-only A/B

baseline 与 combined candidate 必须使用同一：

- TSMC28 setup/min DB、`ssg0p9v125c`、period grid；
- ideal/unpropagated clock、ZeroWireload、compile recipe；
- setup/hold uncertainty、0.25 ns I/O delay、0.1 ns transition、0.01 output load、max fanout；
- synchronous reset 与 256-bit memory port cut；
- 1152-bit output endpoint、tag/backpressure/protocol state；
- achieved passing-grid Fmax 方法，禁止用单条 raw path delay 取倒数。

必须报告 cell area、FF、operator family、setup/hold、constraint violations 和 output manifest。M101 的 M99/M85 sweep 可以证明 serial metadata audit 优于 unrolled audit，但不能提供这里的 `f_bit_sparse` 或 combined candidate Fmax。

### SRAM/macro ledger

至少同时报告：

| 项目 | bit-sparse baseline | cap11 candidate |
|---|---:|---:|
| 双 weight buffer | 24,576 B | 24,576 B |
| 双 PWP+metadata buffer | 0 | 29,588 B |
| pattern table | 0 | 55,296 B |
| phase offset table | 0 | 6,916 B |
| listed response FIFO | 对称配置或 0 | 149 B |
| M88 listed subtotal | 24,576 B（baseline model） | 116,525 B |

32 B/cycle 比较是同带宽，不是同面积。candidate 额外容量、decoder、clocking、ECC 和端口能耗必须计入；若主张 equal-area，还必须让 baseline 使用同等硅面积预算做合理的 banking/replication，并把结果作为第二条基线。宏 inclusive 版本两边必须使用相同 memory compiler/corner/响应延迟，并闭合 address-to-SRAM-to-data path。

## 正确与错误的吞吐公式

port-cut service island 只允许：

```text
S_service = (1,114,383,288 / f_bit_sparse_weight_service)
            / (790,667,725 / f_combined_candidate_service)
```

完整 M88 module 只有在两边 complete top 对齐后才允许：

```text
S_module = (1,114,402,488 / f_bit_sparse_complete_top)
           / (790,706,475 / f_candidate_complete_top)
```

若部署在共同 clock domain，应使用两边都能满足的 `f_common`，频率项相消，只剩匹配的 cycle ratio。

明确禁止：

```text
1.409375695 x (f_M99 / f_M85)
```

因为 M85 与 M99 是同一个 PWP 功能的两种 metadata-audit 实现；M85 不是 bit-sparse baseline，M99 也不是包含 correction/fallback 的完整 candidate top。

## Findings

### P0

1. **M102-P0-01：物理 denominator 缺失。** 当前 `bit_sparse_cycles` 无对应同范围 RTL、VCS trace 或 Synopsys Fmax/area。
2. **M102-P0-02：物理 numerator scope 不匹配。** M101 的 M99 Fmax 只属于 PWP logic island，不能代表 M88 combined candidate cycles；因此现阶段不能形成 cycles/Fmax 的同分母吞吐。

### P1

1. **M102-P1-01：active-source enumeration 是隐含 port cut。** baseline 只收费 popcount 后的 op，scanner/compactor 没有周期、面积或 Fmax。
2. **M102-P1-02：combined weight+PWP service/arbiter 缺失。** correction 和 real escape fallback 尚无 executable path。
3. **M102-P1-03：同带宽但非同面积。** baseline 只有 24,576 B 双 weight buffer，candidate listed subtotal 116,525 B，且仍未含 ECC/queues/accumulators。
4. **M102-P1-04：M88 complete-top 组件不齐。** DMA/parser/row writer/double-buffer controller/matcher/packer 的物理临界频率尚未闭合；service-island Fmax 不能直接套完整 bounded cycles。

### P2

1. **M102-P2-01：service 与 bounded 周期非常接近但不同。** 不能在 numerator/denominator 一边用 service count、另一边用 startup/tail 后的 count。
2. **M102-P2-02：escape evidence 稀薄。** catalog 只有一个 escape entry，heldout 有 362 次使用；需要有针对性的真实 fallback stress。
3. **M102-P2-03：数据仍是 valid825-internal。** 即使物理 service throughput 闭合，也不自动获得 sequence-disjoint accuracy、full-network 或 DATE headline。

## GO / NO-GO

- **GO**：实现并冻结最小 M102 fixed8 weight-service baseline；复用 M82，减少不必要的新 RTL。
- **GO**：同时实现 combined candidate service top，并做相同 descriptor trace、相同 port cut、相同 Synopsys grid A/B。
- **GO**：先发表 `bandwidth-matched vector-service-island throughput`，宏和 equal-area 作为独立后续表格。
- **NO-GO**：当前任何 bit-sparse physical Fmax/area/energy 或 cycles/Fmax throughput claim。
- **NO-GO**：将 M88 `1.409375695x` 乘 M99/M85 frequency ratio。
- **NO-GO**：仅综合 baseline 后用 M99-alone 作为 numerator Fmax。

机器审计见 `m102_bit_sparse_baseline_preflight_audit.json`；机器评审见 `m102_bit_sparse_physical_baseline_preflight_independent_hammer_review.json`。
