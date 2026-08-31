# M219 cropped-K1 DC independent hammer review

结论：**90/100，P0=0。** M219 native-cropped K1 service island 的 exact-SHA、
3 ns、ideal-clock、ZeroWireload、zero-macro DC 可以封存；与 M218 K8 service
island 的 matched logic/register 面积敏感性也可以引用。完整 FC2/FFN、macro-aware
PPA/energy、物理实现、系统和 headline 继续 NO-GO。

独立校验通过 M219 sealed run 的全部 evidence manifest、`dc.rc=0`、五类
constraint clean、postcompile `check_design/check_timing`、setup/hold、mapped netlist、
resource report 和 exact 输入 SHA。M219 原始报告重算为：

- 91,887 cells，19,800 sequential cells，76,857.858437 um2；
- 36 logic levels，2.15 ns critical path；
- setup `+0.3764 ns`、hold `0.0000 ns`；
- 0 macro，ideal clock，ZeroWireload，无 mapped multiplier。

同一 TSMC-28nm、3 ns flow 下，M218 K8 为 88,851.042296 um2。因而 K8
service logic/register 比 native-cropped K1 多 **11,993.183859 um2 / 15.604369%**；
反向说，K1 比 K8 小 13.498079%。这是一组 scope-matched DC sensitivity：两者均
保留 O8/FIFO4 设计目标、18,432-bit Acc24 context、epoch16/gen32、flush1024 和
debug conservation counters。外部八个等容量 weight bank 只是合同不变量，并没有
作为 macro 实例进入任一面积。

寄存器差异可完全解释：两者各有 18,432 context FF；M218/M219 的 response-skid
family 为 1,039/135 FF、FIFO family 为 547/175 FF、scoreboard family 为
696/656 FF，合计恰好减少 1,316 FF。M219 的 context 已占 19,800 个 sequential
cell 的 **93.091%**，所以 native-cropped 之后最贵的仍是 context，而不是单路
INT8 update datapath。

最终面积差并不全是“八路算术”。M218 相对 M219 的 11,993.184 um2 增量中，
2,653.056 um2 是 sequential，5,700.492 um2 是 buf/inv，剩余
3,639.636 um2 是其他 combinational。按 DC 日志中 min-path 修复前的四舍五入面积，
M218/M219 约为 82,407.0/75,802.7 um2，K8 增量只有约 8.712%；hold repair 分别
把面积推高约 7.820%/1.392%。因此 15.604% 可以作为这个 exact DC 点的最终面积
敏感性，但不能写成本征 K8 datapath cost，更不能外推到 routed/macro PPA。

冻结 H67 premodel 的 standalone service cycle 是 K1 2,552,566,588、K8
515,449,096，即 4.952121573x。只把该 premodel 周期比与本次两个 logic-only DC
面积相乘，可重算出条件性 service throughput/logic-area sensitivity：

`4.952121573 * 76,857.858437 / 88,851.042296 = 4.283680292x`。

这个 **4.283680292x 只允许以 conditional / algebraic sensitivity 留档**。它不是
RTL-measured throughput/area，也不是 fair total-accelerator throughput/area：周期来自
固定延迟 premodel，面积不含 weight/context SRAM、bank adapter、M216 frontend、
clock tree 和 route，更没有 power/energy。M220 小矩阵 cross-module miter 也不能把
该冻结 full-trace 周期自动升级成 achieved RTL 性能。

物理风险仍然明显。M219 clock 有 19,800 loads，reset tree 报告 19,795 loads；
DC 对高扇出采用 1,000 的 timing substitute，clock 仍是 ideal。hold worst slack
仍为 0.0000 ns，且有 1,935 个显式 delay cells。DC 还报告 8 个 `VER-318`，并启用
sequential output inversion；必须使用 emitted SVF 跑 Formality。真实 context SRAM
应使用 valid/epoch ownership fence，不能依赖清零 18,432 个 data bit。

允许引用：M219 exact logic/pre-macro DC 数字、M218/M219 matched final-area ratio、
共同 context 与被裁寄存器的组成。只可带完整限定语记录 4.283680292x 条件性代数
敏感性。禁止引用：完整 FC2/FFN、paper-ready PPA、macro/energy、routed timing、
physical/system speedup 或任何 DATE headline 对比。

`docs/359_DATE终局冻结_20260813.md` SHA256 仍为
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
