# M231 independent hammer review

**Score: 76/100. P0: 2. P1: 7. P2: 4.**

M231 的 checkpoint 与 trace 算术成立，但当前 RTL 不能按“fail-closed”晋级，H67 的实际 producer→FC2 性能链也没有闭合。

独立脚本没有导入 M231 production analyzer，重新打开 H67 ep35 checkpoint、枚举 12 个 FFN `sn2` threshold、遍历 120 条 FC2 manifest record 并解析四档 DC 报告。唯一复用的是冻结 checkpoint 的 pickle import-path helper。结果确认：

- 原 H67 checkpoint SHA 为 `4f33e086...`；12/12 threshold 都是 scalar `float32` exact `1.0`，小端原始值均为 `0000803f`。
- PAFT 只能核对 M162 receipt：其记录 12/12 exact `1.0`，但 PAFT hardware accuracy 没有晋级，M193 BN recalibration 也未选中。
- 120 条 FC2 record 合计 5,580,000 tokens、3,502,080,000 bits、437,760,000 packed bytes、143,894,510 events；写一次再读一次的算术是 875,520,000 bytes，零 mismatch。
- 这个 875.52 MB 只表示“假设中间 binary feature map 被 packed 后写一次、读一次”的片上工作量，不是已测 SRAM/DRAM transaction，更不是 cycle、energy 或 speedup。

## Fresh VCS verdict

四档正常路径 fresh VCS 均通过，转置、pair/row tag、group order、ping-pong full 和 deterministic stalls 没有 mismatch：

| width | pairs | tokens | packets | header stalls | raw stalls | full hits |
|---:|---:|---:|---:|---:|---:|---:|
| 384 | 3 | 6 | 6 | 28 | 3 | 9 |
| 768 | 3 | 6 | 12 | 54 | 2 | 10 |
| 1536 | 3 | 6 | 24 | 100 | 7 | 14 |
| 3072 | 3 | 6 | 48 | 196 | 10 | 23 |

但独立并发攻击复现了真实 P0：当 bridge 正在给出一个合法 raw beat 时，同拍注入 wrong-tag event，组合逻辑同时给出：

```text
protocol_error = 1
raw_valid       = 1
raw_ready       = 1
raw_accept      = 1
```

VCS marker 是 `REPRODUCED_M231_P0_FAULT_CYCLE_RAW_ACCEPT`。原因是 `protocol_error` 包含当拍 `illegal_event`，而 `header_valid/raw_valid` 只由寄存的 `fault_q` 门控。内部 `if (!protocol_error)` 不推进，下游却观察到 transfer，破坏事务原子性。原 TB 只在 idle 注入 orphan event，因此没有暴露这个问题。

## Independent DC parse

| width | cell area (um2) | cells | sequential cells | storage bits | setup slack (ns) | hold slack (ns) |
|---:|---:|---:|---:|---:|---:|---:|
| 384 | 5,706.162 | 5,454 | 1,721 | 1,536 | +1.5881 | +0.0008 |
| 768 | 10,631.502 | 9,563 | 3,257 | 3,072 | +1.2559 | +0.0004 |
| 1536 | 20,955.942 | 20,171 | 6,328 | 6,144 | +1.4489 | +0.0002 |
| 3072 | 40,849.326 | 36,355 | 12,473 | 12,288 | +1.2549 | +0.0004 |

四档 recursive evidence manifest 全部通过。数字可信地描述同一 3 ns、TSMC 28 nm、ZeroWireload、0 macro DC screen；sequential cells 几乎就是 `4*INPUT_WIDTH` storage 加控制状态，说明当前实现是全 flop。它不是 SRAM buffer、布局后时序或 paper PPA。

## Layered verdict

| scope | verdict |
|---|---|
| H67 ep35 12 个 threshold exact 1.0 | **GO** |
| PAFT threshold | **GO only as M162 receipt；accuracy NO-GO** |
| 120-record bits/bytes/events | **GO** |
| 875.52 MB | **GO only as counterfactual packed on-chip write+read accounting** |
| 四档正常转置/顺序/背压 | **conditional GO；被 fault atomicity P0 阻断完整 RTL admission** |
| DC area/timing | **GO only as full-flop logic-only diagnostic** |
| M167→M231→M216/M218 executable chain | **NO-GO** |
| finite-buffer cycles / SRAM / energy / complete FFN / system | **NO-GO** |

## P0

1. **M231-P0-01 — fault-cycle output accept 泄漏。** wrong-tag event 与合法 raw beat 并发时，`protocol_error=1` 但 `raw_accept=1`。必须用 current-cycle fault 同拍隔离所有 output valid/accept，并加并发攻击 SVA/TB。
2. **M231-P0-02 — H67 producer-to-FC2 路径尚不可执行。** threshold=1 只证明 amplitude identity；M167 rank3 accuracy=false，PAFT hardware accuracy=false，dynamic-BN barrier 未消除，也没有 M167→M231→M216/M218 cross-module VCS/ordered-trace schedule。因此 875.52 MB 不能当已实现的 H67 收益。

## P1

1. 两槽有限缓冲只做 mean-rate screen；消费者比 producer 慢 2.44×–15.39× 预示大量 backpressure，不等于 cycle overlap 或 deadlock-free trace 证明。
2. PAFT checkpoint 未在本机独立打开，只能依赖 M162 exact receipt；且 accuracy 未晋级。
3. 875.52 MB 假设 packed feature map 写一次、读一次；没有 matched materializing RTL、SRAM bank/port/address transaction 或能耗换算。
4. DC 全 flop、0 macro、ZeroWireload，并用 `set_fix_hold`；最小 hold slack 只有 0.2–0.8 ps，不能推广到物理 SRAM/PPA。
5. 每档只有 3 个 deterministic pair；seed 不改变 data/stall。缺 active-fill/drain 错 tag/group/last、reset、counter rollover 和系统化 slot-turnover 攻击。
6. 原 contract/screen 仍为 `vcs=false`；应新建 superseding admission overlay 绑定修复后 RTL/VCS/DC/review，不要改写历史证据。
7. 没有 Formality、mapped-netlist VCS、PT STA、SAIF 或 PTPX。

## P2

1. M167 没有直接提供 M231 所需 pair header/group index/last；producer order adapter 尚未定义。
2. M231 把 `TAG_BITS-1` pair tag 加 row bit变成 M216 token tag；tag lifetime/唯一性/M218解释需在 wrapper 冻结。
3. trace population 只有 10 samples，缺跨 sequence/event-density stall 分布。
4. DC Tcl 沿用 M216 命名和通用 flow，功能无误但证据卫生可以更清晰。

## Minimum next milestone

做 **M231r2 atomic quarantine + executable M167/M216 trace bridge**：

1. current-cycle fault 同拍屏蔽 `header_valid/raw_valid` 和全部 accept；加入 `protocol_error -> no accept` SVA。
2. 四档覆盖 fill/header/raw 阶段的 wrong tag/group/last/orphan、reset 和双槽 turnover。
3. 实例化 typed M167 BACK adapter、M231、M216，并延伸到 M218 group service 或做 exact interface miter。
4. 用 ordered 120-record payload 执行 dynamic-BN barrier、两槽 ready/valid stalls，与 matched packed-SRAM materialization baseline 对比 cycles/transactions。
5. M167/PAFT accuracy 未晋级前保持 H67/complete-FFN/headline=false；修复后重跑 exact-SHA VCS/DC，再做 Formality 或 mapped-netlist equivalence。

本 review 只新增本目录，没有修改 production RTL/脚本/合同、论文或 `docs/359`。被审阅 RTL snapshot SHA 是 `2df1e2de...`；`docs/359` SHA 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
