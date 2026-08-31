# M457：M455 M451-vs-M433 standalone DC 独立打铁

结论：**84/100。原始 standalone DC 证据 PASS，但 M451 性能主线、standalone Formality 和 standalone PrimeTime 全部 NO-GO。** 无 P0；4 个 P1、4 个 P2。M451 是功能上正确、可综合的负 Pareto 探针，应封存，不应继续占 DATE 新思队列。

## 独立复核结果

独立脚本未读取 M455 receipt，而是重验 M451/M452/M439/M455/M447/M449 内外封，重新解析候选与参考的原始 DC log、area/QoR/setup/hold/check/constraint 报告和 mapped netlist。

| 指标 | M433 sealed reference | M451 K1 fused candidate | M451/M433 |
|---|---:|---:|---:|
| Cell area | 8,351.405814 µm² | 12,952.043867 µm² | 1.5508818701× |
| Cells | 7,139 | 12,802 | 1.7932483541× |
| FF | 1,348 | 1,445 | 1.0719584570× |
| Logic levels | 52 | 42 | -10 |
| Setup worst slack | +0.8411 ns | +0.8828 ns | +0.0417 ns |
| Hold worst slack | +0.0251 ns | +0.0251 ns | equal |
| Macro / blackbox / latch / loop | 0 | 0 | — |
| Constraint violations | 0 | 0 | — |

候选增加 `4,600.638053 µm²`，其中 `4,380.642064 µm²` 是组合逻辑。13-bit × 96 lane 输出加一个 fused flag 恰好解释相对 M433 多出的 97 个 FF，成本主体确实是 96-lane signed pre-adder，而不是隐藏状态。

两次综合使用完全相同的 DC V-2023.12-SP3、Tcl SHA、3 ns SDC SHA、slow-max/fast-min library SHA、`ssg0p9v125c`、ideal clock、ZeroWireload 和 0 macro。M433 来自 07:55 的旧 sealed M439 run，M451 来自 09:08 的 M455 run；这足以支撑机制局部逻辑成本，但不是新 paired-run，更不是集成 iso-resource 对比。

## 性能方向为什么关闭

独立复算 K1 trace 机会为：

`517,041,352 / 430,154,216 = 1.201990664668971×`

它仍不是 RTL measured 或 resource normalized speedup。再除以实际 standalone area ratio：

`1.201990664668971 / 1.550881870126303 = 0.775036892120662×`

即 opportunity-throughput/area 比 M433 **低 22.4963%**。M449 所要求的“低增量成本 K1 探针”条件已被实测推翻。

同时，M451 fused cycle 需要 `160 B PWP + 96 B correction = 256 B`，M433 wide cycle 是 160 B。两个 top 都没有 SRAM macro、bank conflict、address generator、线网、互连或 `old_psum` accumulator。比较还有双向未摊销：M433 分母未包含被消除周期所对应的 separate correction resource；M451 也未包含非 fused correction 剩余路径和 old-PSUM/commit。因此 standalone 数字只能作为负向 fast-kill，不能包装成集成吞吐面积。

## Formality / PT 裁决

- **standalone Formality：NO-GO / defer。** DC 在六处 signed arithmetic assignment 上给出 VER-318。若本点仍有性能价值，本应消歧 signed type 后做 RTL→own mapped netlist Formality；但负 Pareto 已足以关停 DATE 主线，现在继续签核不会改善架构选择。
- **standalone PrimeTime：NO-GO / defer。** 两点 hold 都只有 +25.1 ps，来自 25 ps mapping guard；SDC 无 input slew/driving cell，且 ideal-clock/ZeroWireload。PT 只能进一步精化一个已淘汰点。
- 仅当新的 **integrated amortization screen** 同时纳入 baseline separate correction、candidate residual correction、old_psum、commit、matched 256 B port/SRAM/interconnect，并恢复 throughput/area ≥1 时，才重开“集成 top”的 Formality 和 PT；不重开 standalone M451。

允许表述：M451 standalone 3 ns DC 可综合，但面积是 M433 的 1.5509×；其未准入周期机会折算后的 standalone opportunity/area 为 0.7750×，因此作为 DATE 性能主线关停。

禁止表述：M451 达到 1.202× 硬件、Conv 或系统加速；42 层逻辑意味着更高吞吐；256 B 同拍输入是免费已有带宽；standalone Formality/PT 能把该负 Pareto 点重新准入。
