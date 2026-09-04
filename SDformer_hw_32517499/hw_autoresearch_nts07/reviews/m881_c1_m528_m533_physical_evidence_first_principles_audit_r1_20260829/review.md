# M881｜C1/M528/M533 物理证据第一性原理审计

日期：2026-08-29  
模式：只读证据审计；未运行 VCS/DC/PT/FM/PTPX/SAIF/license/remote，未修改 RTL、既有封存或 `docs/359`。

## 裁决

**PASS_AUDIT；当前 C1 可以引用 CPU 同账本周期、R21 VCS 功能和 parent-scratch 组件宏模型，但尚无当前 M528 RTL 的 DC/STA/Formality/PPA。下一物理动作必须是绑定 R21 的 additive fresh source-only flow；M750/M756/M765 均不得原样执行。**

当前证据可支持三句严格限定的论文表述：

1. 四层 H67 bottleneck Conv、10 个冻结 sample、51.84M source-row 的 CPU 同账本结果为 `435,293,339` cycle；相对 M468 strong-zero 为 `1.746753430105x`，相对 same-coordinate bit 为 `1.741232213066x`。这不是 RTL cycle、PPA 或系统倍速。
2. 当前 RTL 身份 `m528_dead_write_only_1rw_product_capture_island_r2` 已由 Synopsys VCS R21 在 foundry `UNIT_DELAY` functional macro model 下通过，M879 fresh result hammer 为 PASS100。它证明功能/协议，不证明时序。
3. parent scratch 的真实生成宏组件为 `9 x 128x128b 1RW`，物理容量 `18,432 B`、冻结模型面积 `78,825.2454 um^2`；slow corner datasheet cycle/access 为 `0.616/0.4679 ns`。M623 可引用 dead-write suppression 对该九宏组件的模型能量 `3.3018888289 -> 2.0396326003 mJ/frozen sampled inference`（`-38.2283079%`），但不是 C1 总能量或 camera frame。

## 现有面积/时序证据分级

| 证据 | 数字 | 可否引用 | 严格标签 |
|---|---:|---|---|
| M475/M474 predecessor | `37,316.285232 um^2`; setup WNS `0.0000 ns`; hold WNS `+0.0101 ns`; 3 ns | 可作为祖先 DSE/logic-only 证据 | 不同 RTL；instrumented standard-cell-only；parent scratch 与 resident psum 均为 external cut；ideal clock、ZeroWireload、无 CTS/route |
| M477/M476r2 predecessor | `42,370.649130 um^2`; setup `0`; hold `+0.0101 ns` | **不可作为 PASS/PPA 引用** | max-transition、max-capacitance、max-fanout 均有违例；仅失败诊断 |
| 9 个 generated parent macros | `78,825.2454 um^2`; slow cycle/access `0.616/0.4679 ns` | 可作为组件宏 geometry/datasheet model | 非 integrated area/STA/PTPX；不得与 M475 直接相加冒充 M528 总面积 |
| 当前 M528/M533 R21 | 无 DC 面积/STA 数字 | 无当前 PPA 可引用 | 只有 VCS E3 functional PASS；UNIT_DELAY 不是 timing model |
| M750/M756/M765 | 无 result directory、无 attempt sentinel | 无结果可引用 | 全部 source/candidate-only；从未启动 DC |
| 当前 M528 Formality/PT/PTPX | 不存在 | 不可引用 | 没有 source package 或结果闭环 |

明确禁止把 `37,316.285232 + 78,825.2454 = 116,141.530632 um^2` 写成 M528 面积：两项来自不同 RTL 身份，且共享 SRAM、resident psum、weight store、route/clock 均未完整计入。

## DC/宏绑定失败链

1. **M750/r1：永久 NO-GO。** runner/release/final-review 形成未来 SHA 环，无有限 authoring 顺序；runner 从未执行，release 永久禁止。
2. **M756/r2：只修了 SHA DAG。** 它仍依赖未成立的 M746/r12 VCS 门，没有 release、没有 DC 结果，现已被后续 RTL/VCS 身份取代。
3. **M765/r3：最新静态成熟的 runner 骨架，但执行身份已陈旧。** M767 source/candidate hammer PASS100；然而 runner 硬绑定失败的 M758/r13 结果和未来 M766 review，因此不能绑定现有 R21 PASS，也没有 release/result/attempt。
4. **M863/R21：当前功能身份已闭合。** top/adapter 与 M765 所指物理源 SHA 未变化；因此 M765 的资源预检、one-shot、九宏计数、unresolved/inferred-array 拒绝和结果双封逻辑可迁移到 additive successor，但 immutable M765 runner 本身不可改、不可执行。

## 宏绑定的实际语义

- VCS R21 编译 checksum-pinned foundry slow behavioral Verilog，运行模式为 `UNIT_DELAY`；这是功能模型，不是 SDF/时序仿真。
- 计划中的 DC flow 不读取 behavioral `.v`，而是从 slow `.db` link `TS1N28HPCPHVTB128X128M4S`，在 elaboration 前后以及 mapped netlist 中都要求恰好 9 个 reference；这属于 **resolved library macro cells**，不是寄存器逻辑模型，也不是允许 unresolved 的 blackbox。
- Formality 的 binding plan 只规定把九个宏按 instance/cell 作为 cutpoint/blackbox 匹配；当前没有 Formality run。
- 现有 M750 Tcl 只对 standard-cell slow/fast 执行 `set_min_library`。私有资产中 fast macro DB 确实存在且已由 DC readback 验证，SHA 为 `8c163161...`；因此旧 Tcl 的 macro hold 只能叫 slow-view diagnostic，下一 fresh flow 必须显式加入 macro slow→fast min-library 对。

## `213,376 B` 与物理宏账一致性

结论分两层：

- **容量义务层：PASS。** M528 冻结账为 `213,376 B`，低于 `245,760 B`，余量 `32,384 B`。其中 parent scratch 的 logical `9,216 B` 已按真实 9 个 `128x128b` 宏、只用低 64 行，计为 physical capacity `18,432 B`；九宏面积单独计为 `78,825.2454 um^2`，没有换算成“免费容量”。
- **全存储物理实现层：OPEN。** 除 parent scratch 外仍有 `194,944 B` 的 psum/weight/directory/source/bitmap/FIFO 等义务。它们继承的是 simulator 的 64-row/144-bit granule `macro_rounded` 账，不等于现有 `128x128b` generated macro 的逐项实例映射；当前 DC top 还把 resident psum 作为输入端口，其他小数组可能综合成 standard cells。故 `213,376 B` 不能写成“全部 SRAM 已由 foundry 宏物理化”。

## 下一 fresh source-only 公平物理点

不应只重包 M765 单候选 flow；应冻结一个 **product-capture / strong-zero / same-coordinate-bit 三轴** 物理比较包，或先明确发布“candidate-only component DC”且禁止称公平比较。三轴包至少包含：

### 必须冻结的输入

- 当前 candidate top SHA `726039db...`、macro adapter SHA `8fd008a3...`。
- M863 R21 `RUN_COMPLETE.json` SHA `7b10955c...` 和 M879 review SHA `54daccf7...` 及双封。
- M528 CPU result SHA `778c8e1b...`、M528 hammer SHA `4f70610d...`；冻结三条 cycle：K/product `435,293,339`、zero `760,350,133`、bit `757,946,784`。
- 同一 10 sample/四 Conv/51.84M row、96 lane、B8、row64、128 B/cycle、3 ns、240 KiB ceiling、相同 Acc/IO cut/debug-counter policy。
- standard-cell slow/fast DB；macro slow DB `cd8c2050...` 与 macro fast DB `8c163161...`；foundry manifest `c070d542...`。
- 新增的 zero/bit RTL wrappers：当前盘上不存在可直接综合、与 M528 同协议/IO cut 的两条 baseline，不能用 CPU simulator 名字替代 RTL。

### 公平资源规则

- 三条设计使用完全相同 SDC、corner、input/output delay/load、clock uncertainty、compile/hold-fix 策略。
- product-capture 必须计入 9 个 parent macros；zero/bit 不应被强塞无用 scratch，也不得把 candidate scratch 隐去。分别报告 logic area、design-specific macro area、shared-memory subtotal和总面积。
- shared psum/weight/source/directory storage 要么三条都绑定同一物理宏/模型，要么三条都作为完全相同 external cut 并单列表外 subtotal；禁止一条 macro-inclusive、另一条 logic-only。
- 3 ns 若任一条不闭合，必须用各自 achieved period 计算 `cycle x period` latency；不得只放宽 candidate 时钟后仍沿用 `1.74x`。

### 准入门

1. 三条功能结果均有独立 VCS/miter 证据；当前 R21 只覆盖 candidate。
2. candidate elaboration/postcompile/netlist 必须恰好 9 个 resolved macro cell；禁止 inferred parent array、unresolved reference 或 behavioral `.v` 进入 DC。
3. slow setup + fast hold（包括 macro slow→fast pairing）均 MET；setup/hold/max-cap/max-transition/max-fanout 五类约束全净。
4. area 必须拆出 standard cell / parent macro / shared storage；当前 M475/M477 数字不得代入。
5. cycle gate 保持同账本 `>=1.70x` vs zero 和 bit；物理主表再要求 latency speedup `>=1.50x` 且 throughput/mm2 `>=1.15x`。否则 C1 仅保留 CPU-cycle/组件证据。
6. DC 后必须进行 RTL↔mapped-netlist Formality；九宏按固定 instance/cell cutpoint，adapter address/control/slice boundary 全比较。
7. 该 DC 即使 PASS 也只称 pre-macro/macro-linked DC candidate；PT/route/clock/SAIF/PTPX 未完成前 `paper_ppa_ready=false`。

## 优先修复

1. 基于 R21 新建 additive DC source-only identity；迁移 M765 fail-closed 骨架，但把 prerequisite 改为 M863+M879，且显式绑定 macro fast DB。
2. 若目标是最快获得当前 C1 物理锚点，可先做一次 candidate-only macro-linked DC；必须在收据里写 `fair_K_zero_bit_physical_comparison=false`。
3. 在论文需要公平 PPA 前，再补 zero/bit 同协议 wrappers 与三轴 DC；不要用 M475/M477 祖先面积充当 baseline。
4. DC 成功后先做 Formality，再做 PT/PTPX；当前没有任何可复用的 M528 Formality 结果。

## 证据完整性

本审计重新验证了 M528 CPU result、M528 hammer、M863 R21 result、M879 result hammer、SRAM mapping、M617 component model 和 M623 result hammer 的双封；`docs/359` SHA 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
