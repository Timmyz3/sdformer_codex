# M388：M387/M384 controller DC 独立打铁

结论为 **92/100，P0=0、P1=0、P2=6**。M387 r1b 的 exact-input、evidence manifest、seal、receipt、reports 与 mapped netlist 一致，因此放行 RTL→netlist Formality 和明确标注为 prelayout 的 PT-STA；没有真实 SAIF/VCD/FSDB，activity-backed PTPX 继续 NO-GO。

3.0 ns logic-only 点为 7,588.223873 um²、7,746 cells、1,170 sequential cells、68 logic levels，setup/hold 分别为 +0.6302/+0.0251 ns；max/min delay、max capacitance、max transition、max fanout 五类约束均无违例，macro/black box 为 0。最差 setup path 独立重数为 68 级，与 QoR 一致。

这个点不能只报 `MET`。最终网表包含 1,233 个显式 DEL cell、2,057.705994 um²，占总 cell area 27.117%。DC log 中 hold-only 前的 rounded point 是 5,194.2 um²/5,642 leaf cells；最终约增加 2,394.023873 um²（46.09%）和 2,104 cells（37.29%）。这是 ideal-clock、ZeroWireload 下强 hold-fix inflation，不能外推为物理 PPA。

存储同样是主要来源：512b centers 映射为 512 flops/1,161.216 um²；原始 384b FIFO 因 72 个常量位被优化，映射为 312 flops/628.992 um²；其余 state 为 346 flops/784.727914 um²。descriptor/PWP SRAM、q32 matcher 和 PWP compute 均不在此 cut。

警告均未形成 P0/P1：VER-104 仅是冻结 `FIFO_DEPTH=8` 下未展开的 bad-parameter `$fatal`；VER-318 三处只涉及正数常量或 0..31 loop index；TIM-216 仅为已 false-path 的异步 `reset_n`；TIM-134 仅为 ideal `clk_core`；全部 LINT-29/31/52 与 123 个 UCN-1 均被核对为 top-level shared-output boundary。它们仍列为 P2，后续 signoff 不能隐藏。

首次 r1 目录虽然 payload manifest 可校验，但同时存在 `RUN_FAILED_OR_INCOMPLETE.txt` 且 `runner_sha256.txt` 为空，故明确不可引用；只认 r1b。

冻结边界保持：`physical_descriptor_sram=false`、`physical_pwp_sram=false`、`physical_timing=false`、`system_speedup=false`、`paper_ppa_ready=false`、`date_headline=false`。
