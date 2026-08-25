# M133/M133r2 dual-row 512-bit elastic PWP stream independent hammer

## Verdict

**Active M133r2: 89/100；P0=0、P1=1、P2=3。Standalone 512-bit assembler 的数值、beat geometry、legal handshake、fail-closed 组合语义与 logic-only DC 通过；512-bit bank/macro 与 physical speedup 未实现。**

M133 r1 的 sealed commercial VCS 本身可复现，但独立交叉攻击发现了一个真实 P1：先采样一个 stalled output，再在下一拍送 malformed input，RTL 会按 fail-closed 语义立即隐藏 `output_valid`，而 r1 的无条件 `ap_output_stable_under_stall` 要求下一拍仍 valid，导致 assertion failure。冻结 r1 SVA exact SHA 为 `ab45fe7d...`，反例为 `started at 19500ps failed at 22500ps`。

M133r2 没有修改 RTL；它把属性修成“下一拍若 protocol fault 则允许且要求 quarantine，否则 held output 必须 valid/stable”，并把重叠攻击加入 production TB。r2 sealed VCS 与两路独立 VCS 均 assertion-clean，`cp_stall_to_fault_quarantine=1`；因此 r1 P1 已关闭。

## Independent beat geometry

96 lanes 的 packed signed payload 只需整数 ceiling division：

| Width | Payload bits | 512-bit accepted beats | Last-beat valid bits | Required zero padding | Signed range |
|---:|---:|---:|---:|---:|---:|
| 8 | 768 | 2 | 256 | 256 | -128..127 |
| 9 | 864 | 2 | 352 | 160 | -256..255 |
| 10 | 960 | 2 | 448 | 64 | -512..511 |
| 11 | 1,056 | 3 | 32 | 480 | -1,024..1,023 |

RTL 的 `beats_needed_q`、`expected_last`、three-beat buffer placement、per-lane sign extension 和 final-padding slices 与上表一致。生产 VCS 对每个 width 都命中 cover，101 个 numeric vectors 共检查 9,696 lanes；其中 5 个 vectors 使用 signed min/max 边界。

这证明的是 512-bit input stream 已到达模块之后的 assembler 服务拍数，不证明两个 256-bit SRAM rows 能在同一拍真实到达。

## Why accepted beats is 236

`236` 可以从 TB 控制流独立重建：

```text
first 64 round-robin vectors
  = 16 × (2 + 2 + 2 + 3) = 144 beats

mixed vectors 64..103
  = 36 numeric vectors + 4 escapes
  = (8×2 + 10×2 + 8×2 + 10×3) + 4 = 86 beats

long-stall width-11 vector = 3 beats
positive total             = 144 + 86 + 3 = 233 beats
negative-attack setup      = 1 + 2 = 3 accepted legal beats
directed counter total     = 233 + 3 = 236 beats
```

所以 `236` 是该 directed simulation 中的 input handshake 数，包含攻击发生前的 3 个合法 setup beats；不是 236 个 vectors、workload accesses、SRAM rows、cycles saved 或 speedup denominator。正向范围是 105 vectors = 101 numeric + 4 escape。

## Handshake and quarantine audit

### Legal traffic

- `beat_accept = beat_valid && beat_ready`，`output_accept = output_valid && output_ready`。
- valid=0 时 `beat_ready` 只由 capacity 决定，不依赖 idle payload；production TB 有显式检查。
- 输出只有一个 elastic entry。已有 output 在同拍 retire 时可接受下一 vector start；no-stall start II 精确为 2/2/2/3，63 次相邻 start 检查通过。
- legal output stall 下 tag、width、escape、1152-bit values 保持稳定；23-cycle long stall 与 pattern stalls 通过。

### Malformed request priority

- 当前请求若 metadata/order/last/padding 非法，`request_violation` 当拍进入 `quarantine`，使 `beat_ready=0`、`beat_accept=0`、`output_valid=0`、`output_accept=0`。
- fault 在时钟沿写入 `faulted_q`，直到 reset 前持续隔离。独立 R2 cross TB 在移除坏请求后又保持 3 cycles，sticky quarantine 通过。
- 这套语义明确选择“fault isolation 优先于之前 held output 的可见性”。因此 stall property 必须限定 legal traffic 或允许 consequent-edge `protocol_error`；r2 已这样修复。

## Commercial VCS evidence

| Run | Result | Key evidence |
|---|---|---|
| Sealed M133 r1 | Original directed PASS | 105 outputs、236 accepted beats、all listed covers；未覆盖 stall→fault composition |
| Frozen-r1 independent cross | Expected assertion FAIL | RTL quarantine 正确；r1 `ap_output_stable_under_stall` 与其冲突 |
| Sealed M133r2 | PASS, assertion-clean | production `stall_fault_overlap=1`，cover match=1 |
| Independent exact-SHA r2 production rerun | PASS, assertion-clean | frozen RTL/SVA/TB/contract SHA 独立预检 |
| Independent r2 cross | PASS, assertion-clean | 同一反例已关闭，另验证 3-cycle sticky fault |

R2 only changes SVA/TB；RTL SHA 始终为 `84f1b6f6...`。因此功能修复不会改变下面的 DC mapped logic。

## Synopsys logic-only DC

| Item | Result |
|---|---:|
| Tool/corner | DC Graphical V-2023.12-SP3, TSMC28 SSG 0.9 V / 125 C |
| Constraint | 3.000 ns, setup uncertainty 0.200 ns, hold uncertainty 0.050 ns |
| Clock/wire | ideal unpropagated / ZeroWireload |
| Cell area | 10,853.766052 um² |
| Combinational / sequential area | 5,275.493962 / 5,578.272090 um² |
| Cells / sequential cells | 12,667 / 2,767 |
| Setup worst slack | +1.1005 ns MET |
| Hold worst slack | +0.0001 ns MET |
| Constraint violations | 0 across max/min/cap/transition/fanout |
| Macro / net interconnect area | 0 / undefined |

2,767 sequential cells are consistent with the RTL's 1,536-bit assembly buffer, 1,152-bit output register and control/tag state。关键 setup path 是 `accepted_beats_q_reg[1] -> buffer_q_reg[824]`，data arrival 1.6940 ns。

这个结果只准入 standalone assembler at 3 ns。hold 只剩 0.1 ps，且是在 `set_fix_hold`、ideal clock、ZeroWireload 下，不是 post-layout margin；单个 3 ns point 也不能推出 Fmax。没有 Formality、PT/CTS/routing、SAIF/PTPX 或 macro-inclusive energy。

## The missing 512-bit hardware

当前 RTL port 是一个已经准备好的 `beat_data[511:0]`。模块没有 row address，也没有 16×32-bit bank request/response、bank mapping、collision arbitration、stall/replay、SRAM macro 或 macro-to-register path。因此名称中的 “dual-row” 只是 service-port hypothesis 的来源，不是双行 SRAM frontend implementation。

M132 的 `1.431769986×` 是 same-clock service-island cycle model；M133/M133r2 只证明 candidate assembler 能按模型消费 2/2/2/3 beats。要形成 physical throughput，仍需：

```text
physical gain = modeled cycle gain × (f512 / f256)
```

并且必须将 conflict stalls 纳入 candidate cycles。没有 matched 256-bit baseline 与真实 512-bit bank/macro，不能将 M132 ratio 或这里的 DC slack 称为 physical/system speedup。

## Scorecard

| Dimension | Score | Evidence |
|---|---:|---|
| Beat geometry and signed numeric | 20/20 | 独立 ceiling algebra、tail boundaries、9,696 lane checks exact。 |
| Elastic/fail-closed protocol | 18/20 | r1 反例已由 r2 和独立 VCS关闭；尚非 exhaustive formal。 |
| Commercial reproducibility | 15/15 | r1/r2 sealed receipts 与独立 exact-SHA r2 双测。 |
| Logic-only Synopsys | 14/15 | clean mapped 3 ns result；ideal clock/ZeroWireload。 |
| Physical supply realism | 7/20 | 512-bit port cut 存在；bank mapper、macro、conflicts、matched A/B 不存在。 |
| Claim discipline | 15/15 | contract/receipt 均禁止 physical/system/headline 上抬。 |
| **Total** | **89/100** | **强 standalone module milestone，不是完整 dual-row frontend/PPA。** |

## P0

**0 个。** 在冻结 directed scope 内没有 beat geometry、signed lane、handshake count、commercial VCS 或 logic-only DC 数值错误。

## P1

### P1-1 — 512-bit bank/macro supply path 未实现

这阻断 physical speedup 与 paper PPA。需要真实 16-word/two-row producer、address mapping、cycle-exact conflicts、arbiter/replay、matched 256/512 DC/PT/macro A/B。r1 的 stall/fault P1 已由 M133r2 关闭，不再列为 active P1。

## P2

### P2-1 — Dirty-padding negative 只攻击 width 8

RTL 对 9/10/11-bit padding slices 静态正确，positive vectors 也为零 padding，但 production negative 没有分别翻转 160/64/480-bit tail。应各加一次 exact boundary attack。

### P2-2 — Sticky fault 缺 production SVA

RTL `faulted_q` 与独立 3-cycle VCS 已证明 sticky behavior；production SVA/TB 仍没有“bad request 移除后多拍保持 fault，reset 后清除”的完整 temporal assertion。建议补一条 property 与 cover。

### P2-3 — r1 correction pin 了已演进的中间 audit SHA

`m133_r1_stall_fault_composition_correction_r1_20260824.json` 的 counterexample TB SHA 仍 exact，但其中 `audit_sha256=d192...` 对应本次 review 的中间版；同一路径随后追加 DC/r2 结果。功能和 r2 exact identity 不受影响，但 evidence lineage 应在最终 manifest 生成后用新的 identity supersession overlay 收口，不应原地修改已被 r2 pin 的 correction。

## Safe claim

> Exact-SHA commercial VCS and independent reruns verify an unchanged standalone 512-bit M133r2 stream assembler that reconstructs signed 96-lane vectors in 2/2/2/3 accepted beats for 8/9/10/11-bit payloads, sustains legal backpressure, and prioritizes same-cycle fail-closed quarantine for malformed requests. A TSMC28 3 ns ideal-clock ZeroWireload DC cut maps to 10,853.77 um² with +1.1005 ns setup and +0.0001 ns hold slack. The bank mapper, two-row SRAM supply, conflicts, macros, matched physical speedup and system speedup remain unadmitted.

## Prohibited claims

- 不得把 `beat_data[511:0]` 称为已经实现的 dual-row SRAM frontend；
- 不得把 2/2/2/3 accepted beats 直接写成 physical latency under conflicts；
- 不得把 236 当 vectors、SRAM accesses 或 cycles saved；
- 不得把 +1.1005 ns setup slack 推成 Fmax；
- 不得把 0.1 ps ideal-clock hold slack 当 post-layout margin；
- 不得将 M132 1.43177× 或 4.54146× 上抬为 physical/system/headline speedup；
- 不得声称 macro area/power/energy/FPS、paper PPA 或 DATE best-paper readiness。

## Artifacts

- `audit_m133.py` / `m133_independent_audit.json`：geometry、traffic、sealed VCS、R2、DC 与 scope 审计。
- `frozen_r1_m133_assertions.sv`：exact-SHA r1 SVA，保留已发现反例。
- `tb_m133_stall_fault_interaction.sv`：独立 stall→fault + 3-cycle sticky attack。
- `frozen_r1_stall_fault_cross_property/`：预期 r1 assertion failure。
- `independent_r2_production_rerun/`：exact-SHA production r2 rerun。
- `independent_r2_stall_fault_cross_property/`：同一交叉攻击在 r2 下 assertion-clean。
- `source_evidence.sha256`：production contracts/sources/sealed receipts/reports exact SHA。
- `manifest.sha256`：review 文本证据 exact SHA。

`docs/359_DATE终局冻结_20260813.md` 未修改，SHA256 保持 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
