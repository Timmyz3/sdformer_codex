# M485 FC2 三点 Pareto predesign 独立打铁评审

## 裁决

**87/100。GO 一次 matched top-level 3 ns DC/STA 诊断；NO-GO 当前三点直接进入论文 PPA/加速主表。**

最值得收割的故事不是把 M342 的 `5.281374845x` 重新包装成“同带宽加速”，而是把三点连成一条资源曲线：

1. `K1 resident`（M216 `SOURCE_CAP=1` + M219）以一个 bank-word/cycle 工作，逻辑较小但闲置七个已有 weight bank；
2. `K8 bank-coissue`（M216 `SOURCE_CAP=8` + M218）在一次事务中消费至多八个 bank-word，并把它们折叠到一份 Acc24 context；
3. `K1x8 equal-peak-bandwidth`（M216 `SOURCE_CAP=8` dispatcher + 8×M219）达到相同八 bank-word/cycle，但用八套 service/partial-Acc24 state。

M349 已经证明第 2、3 点在 directed raw4→Acc24 边界上严格 `1.000x` cycles。若 matched DC 证明第 2 点以显著更小面积达到第 3 点的吞吐，这可以写成 **throughput-preserving bank-coissue and partial-state collapse**；它是面积/能效贡献，不是新的 cycle speedup。这个方向比继续捍卫 `5.281x` 更诚实，也更可能通过 DATE 审稿人的公平性复算。

## 现有三点究竟是否可比

| 点 | frontend / service | weight 峰值 | service capacity | Acc24 context bits | 现有周期证据 | 现有 DC 证据 |
|---|---|---:|---:|---:|---|---|
| K1 low-logic | M216 cap1 + 1×M219 | 1×128 bit/cyc，总共八个逻辑 bank 但每拍最多激活一个 | O8 / FIFO4 | 18,432 | M342 serialized-port sensitivity 的分母 | 只有拆分组件 |
| K8 bank-coissue | M216 cap8 + 1×M218 | 8×128 = 1,024 bit/cyc | O8 / FIFO4 | 18,432 | M342 相对 K1 `5.281374845x`；M349 相对 K1x8 `1.000x` | 只有拆分组件 |
| K1x8 equal-BW | M216 cap8 + 8×M219 + atomic join | 8×128 = 1,024 bit/cyc | O64 / FIFO32 | 147,456 | M349 相对 K8 `1.000x` | 无 top-level DC |

### 可比的部分

- 三者使用相同 raw bitmap、source/destination mapping、signed INT8 weight 和 Acc24 结果边界。
- K8 与 K1x8 在物理 weight-bank 层都只要求八个 1R bank endpoint，每个 bank 每拍最多一个 128-bit word；weight 容量原则上可以完全相同。
- M216 cap1/cap8 已有 matched DC，除被常量折叠的三位 source-count 外保留相同 descriptor queue 和双 D8 window。
- M218/M219 已有同一 28 nm、3 ns、ideal-clock、ZeroWireload、zero-macro DC；两者都保留一份 18,432-bit Acc24 context。

### 不可比、必须显式列出的部分

- M342 与 M349 是两个 testbench/allow-schedule。连同一个 K8 点，M342 的 B1/B2/B4/B8 cycles 为 `42/112/410/1027`，M349 为 `51/131/486/1231`。因此现有数字不能直接组成一张同一横轴的三点 Pareto；先要做同一 frozen replay。
- M349 K1x8 是故意偏强的 cycle baseline：O64/FIFO32、八份 context 和八套 identity/control，不是 same-resource baseline。`1.000x` 证明等峰值带宽时没有额外 cycle gain；它不能单独证明 area savings。
- 三组 DC 都未实例化 weight SRAM。所谓“相同八 bank 容量”仍只是合同不变量，未包含 macro area、port timing、decode/interconnect 和 energy。
- M218/M219 的 context 仍是带 mass reset 的 FF array。K1x8 的 147,456 bits 与 K8 的 18,432 bits 是真实架构差异，但现有 stdcell 面积不是可投稿 SRAM PPA。
- M349 的八路 epoch/slot/generation/response metadata、atomic result join 和更大 reset/clock tree 都是实际实现成本，不能从比较中删掉；debug conservation counters 则必须在三点都以同一规则关闭/裁掉。

## 现有面积能说明什么

精确拆分组件数字为：

- M216 frontend：K1 `20,436.696076 um2`，K8 `20,587.392080 um2`；
- M219 K1 service：`76,857.858437 um2`；
- M218 K8 service：`88,851.042296 um2`。

仅作 predesign 的 block-sum（不是 integrated DC）得到：

- K1：`97,294.554513 um2`；
- K8：`109,438.434376 um2`，约为 K1 的 `1.124816x`；
- K1x8：`635,450.259576 um2`（尚未加 atomic join），约为 K8 的 `5.806463x`。

这组 block-sum 给出了非常明确的 **DC 值得做** 信号，但没有准入任何面积倍率。原因是跨层优化、未使用 debug 裁剪、reset/hold repair、join 和大端口缓冲都可能显著改变 integrated area。尤其 M218 的 hold repair 约增加 `7.820%`，而 M219 约增加 `1.392%`，简单相加会把不同比例的 hold 税固化进结论。

若把 M342 synthetic `5.281374845x` 与上述 block-sum 相乘，可得到 `4.695325x` 的 throughput/logic-area **代数敏感性**；若把 M349 的 `1.000x` 与 K1x8/K8 block-sum 相乘，则是 `5.806463x`。两者现在都禁止写成 iso-area 或 measured throughput/mm2。前者混用了拆分 DC 与 bundled-port synthetic cycles；后者的面积分母尚未 top-level 综合且 baseline 过度配置。

当前能合法写的只有：

> M342 在单 bundled request port 下观察到 5.281x；将 baseline 扩到相同 1,024-bit/cycle 峰值后，M349 的周期比变为 1.000x。由此，下一项待验证的硬件价值是以 bank-coissue/shared-context 替代八套 scalar service 的面积和能量，而不是额外 cycle speedup。

## 综合阻塞审计

### 未发现 P0 级语法/多驱动阻塞

- M342/M349 exact-SHA VCS compile 均为 0 warning/error；RTL 中没有 `force/release`、delay、testbench task 或不可综合 assertion。
- M342、M218、M219 的 `initial $fatal` 只位于非法参数 generate 分支。M218/M219 已被同一 DC 版本成功综合；固定合法参数后该分支会被展开消除。
- M349 的 block-local signed sum、unpacked array ports 和 generate 结构均为合法 SystemVerilog；M216 中相同风格的 block-local declaration 已通过 DC。
- 没有看到同一 `logic` 被两个 procedural block 驱动的迹象；clean VCS elaboration 也未报多驱动。

### 有三项真实执行风险

1. **M349 规模。** 八个 M219 至少带来 147,456 个 Acc24 context FF；按已有单块报告外推，sequential cells 会超过 158k，再叠加 frontend、join、reset/hold buffer。DC 可能耗时/内存很高，但这是成本暴露，不是应通过删状态规避的“综合错误”。
2. **巨型边界。** M219 单块已有 1,100 个 ports；M349 的八套 metadata 和 1,024-bit weight response 会带来显著 top-level IO buffering。三点必须使用功能 wrapper 和 canonical eight-bank adapter；不得只给 K8 bundling 免费黑盒、却让 K1x8 的 metadata 留在 stdcell 边界。
3. **debug 与 mass reset。** 当前 debug counters 是可观察输出，默认会被保留；context data 又全部 reset。这些适合 diagnostic DC，不适合 paper macro PPA。综合 wrapper 应对三点同样开放功能端口、同样关闭 debug；后续 macro 版必须以 valid/epoch fence 替代 data mass reset。

另有 Formality P1：M218/M219 DC 已出现 signed conversion/part-select `VER-318` 和 sequential output inversion。matched DC 完成后必须使用各自 emitted SVF 做 RTL↔netlist Formality，不能把 VCS exact 直接当成 netlist 等价。

## 最小 M485 合同

### A. 身份与功能范围

- 固定三个 DUT：M342 `SOURCE_CAP=1`、M342 `SOURCE_CAP=8`、M349；固定本评审 JSON 中列出的 RTL SHA。
- 唯一性能边界：accepted header 至 accepted token_done，raw4→signed Acc24；BN2/SN2/requant、FC1、完整 FFN 和系统均为 false。
- 三点用同一 120-record frozen H67 FC2 cohort、同一顺序、同一 per-bank fixed-L4 响应规则、同一 raw/result/done stall edge ordinal。
- 每个 bank 每拍最多一个 128-bit request/response；weight capacity/address mapping 完全相同。K8 bundled response 只可在其 mask 的所有 bank word 就绪时返回。
- 必须逐 token 比较结果、request/response multiset、active-bank reads、weight bytes、cycle endpoints；覆盖 zero/full8、OOO、request/raw/result/done stall、midflight reset 和 protocol attack。

### B. matched 3 ns DC/STA

- TSMC 28 nm HPC+、同一 slow corner、3.000 ns、ideal clock、ZeroWireload、zero macro、同一 compile effort、同一 IO delay/load/max transition/max fanout/max capacitance；不得给任一点 false/multicycle path 优待。
- 使用三个 synthesis-only functional wrapper。debug counters 在三点都以相同规则不可观察并允许裁剪；protocol/numeric/stale/busy 等功能状态必须保留。
- 用 canonical eight-bank boundary 统一八个 1R×128-bit bank endpoint。weight macro 本体同时排除或同时计入；不得混用。
- 同时报最终和 hold-fix 前面积、combinational/sequential/buf-inv/delay cell、context bits、clock/reset loads、port count、setup/hold、critical path、warning census、check_design/check_timing 和五类 constraint。
- 先报告 3 ns QoR；Fmax sweep 必须是独立同规则实验。不得用一个点的 3 ns 面积配另一个点的 Fmax。

### C. GO/NO-GO 门

1. **统一 VCS 门：** zero mismatch/assertion failure；K8 对 K1x8 的每-record throughput ratio 不低于 `0.95x`，geomean 不低于 `0.98x`；K8 对 K1 的 geomean 至少 `3.0x`。后一个数字只能标为 bank-bandwidth Pareto，不得标 grouping-only speedup。
2. **DC clean 门：** 三点 setup/hold 均 `>=0`，postcompile check_design/check_timing clean，五类 constraint clean，无 unresolved reference、多驱动、latch 或 test-only cell。禁止放松 3 ns 过门。
3. **K1→K8 Pareto 门：** K8 functional-top area不超过 K1 的 `1.25x`，且 unified throughput 至少 `3.0x`；否则 K8 不能作为低成本 bank-utilization Pareto 点。
4. **K8→K1x8 iso-throughput-area 门：** K8 throughput 至少 K1x8 的 `0.95x`，Fmax 至少 `0.90x`，functional-top logic/register area不超过 K1x8 的 `0.50x`。通过后只准入“pre-macro logic/register area reduction”。
5. **paper 能效门：** 在相同 weight macros、显式 context storage 和同一 frozen SAIF 下，K8 energy/token 不超过 K1x8 的 `0.70x`，PTPX annotation coverage 至少 `95%`。未过此门不得写 energy efficiency。
6. **paper PPA 门：** context/weight macro 或经过审计的 CACTI/宏敏感性、Formality 0 unresolved/aborted、至少两条 DSEC sequence。否则 `paper_ppa_ready=false`、`headline=false`。

M485 的最小动作不是新写第四种 FC2 算术 RTL，而是：**统一三点 replay → 三个 matched wrapper DC → 只在面积门通过后做 macro/SAIF/PTPX。** 若 K8 相对 K1x8 的 integrated area reduction 小于 2×，或同频吞吐落后超过 5%，立即 NO-GO，不再扩结构。

## 论文口径建议

可以成为一个硬件贡献的写法：

> A bank-coissued signed-accumulation service converts eight independently addressable weight-bank words into one tagged transaction and one shared Acc24 context. Compared with a throughput-matched eight-lane scalar construction, it preserves raw4-to-Acc24 throughput while eliminating replicated partial-state/control; a serialized resident K1 point exposes the bandwidth/area Pareto endpoint.

禁止的写法：

- “K8 在同带宽下加速 5.281×”（M349 已证伪）；
- “K8 比 K1x8 同资源”（O8/FIFO4 对 O64/FIFO32，context 1× 对 8×）；
- 用 block-sum `5.806×` 写 measured area efficiency；
- 把 standalone FC2 数字写成完整 FFN、网络或系统倍速；
- 把 zero-macro、ideal-clock、ZeroWireload 数字称为 28 nm paper PPA。

## 评分

| 维度 | 分数 | 判断 |
|---|---:|---|
| 身份与既有证据可追溯 | 20/20 | exact-SHA VCS、两组独立 hammer 和拆分 DC 完整 |
| 公平性诊断 | 18/20 | equal-BW 负结果已纠错；仍缺同一三点 frozen replay |
| Pareto/创新潜力 | 18/20 | shared-context bank-coissue 有清晰面积/能效假设 |
| 物理可执行性 | 15/20 | 无明显综合语法阻塞；macro、reset、巨型 M349 仍高风险 |
| 论文准入完整度 | 16/20 | 合同和红线明确；integrated DC/Formality/PTPX/多序列均缺 |
| **总分** | **87/100** | **GO diagnostic DC，NO-GO paper claim** |

本评审只读既有 RTL/VCS/DC/trace evidence，没有运行 DC，没有修改现有 RTL、合同或 `docs/359_DATE终局冻结_20260813.md`。后者 SHA256 复核仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
