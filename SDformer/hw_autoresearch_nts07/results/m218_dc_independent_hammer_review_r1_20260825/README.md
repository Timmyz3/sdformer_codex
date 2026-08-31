# M218 service-only DC independent hammer review

结论：**89/100，P0=0。** M218 service-only logic/pre-macro DC 可以封存；
macro adapter、native-cropped K1 sensitivity 和 M216+M218 connected cycle miter
均可进入下一阶段。完整 FC2/FFN、物理、系统与 headline 继续 NO-GO。

独立检查通过 sealed run 的全部 evidence manifest、`dc.rc=0`、五类 constraint
clean、postcompile `check_design/check_timing`、setup/hold report、mapped netlist 和
resource report。没有 multiplier、macro 或 timing violation。报告数字独立解析为：

- 113,012 cells，21,116 sequential cells；
- 88,851.042296 um2 cell area，51 logic levels；
- 1.84 ns critical path，setup `+0.6872 ns`、hold `0.0000 ns`；
- 0 macro，ideal clock，ZeroWireload。

最大的新增事实不是“时序很好”，而是 service state 的真实成本。18,432 个 Acc24
context FF 占全部 sequential cell 的 **87.289%**；noncombinational area 为
42,569.856685 um2，占总 cell area **47.911%**。1,024-bit weight skid 也确实被
寄存，整个 response-skid family 为 1,039 FF。它证明 M218 没有继续免费使用
6,144-bit response，但还没有证明 SRAM macro 的 port、latency、energy 或输出时序。

物理风险也已经显性化。mapped netlist 的 clock 有 21,116 loads，reset tree 报告
21,115 loads；DC 在 timing 中把高扇出按 1,000 处理，而且 clock 仍是 ideal。
hold-only phase 将 rounded area 从约 82,407.0 推到 88,851.042296 um2，增加约
6,444.0 um2 / 7.820%。最终有 2,141 个显式 delay cell；全部 buffer/inverter
面积为 12,586.391915 um2，占 14.166%，但 worst hold 仍只有 0.0000 ns。因此
setup 正余量不能外推成 routed closure。

Macro adapter 应优先解决 context 数据的 mass reset：真实 SRAM 不应依赖清零
18,432 个 data bit，应使用 valid/epoch/ownership fence。随后必须把有限 bank port、
response latency/backpressure、slot/generation/tag 身份接入 connected cycle miter。
native-cropped K1 可作为面积/能耗 sensitivity，但 scope-matched K1/K8 仍要保留，
否则会丢失 SOURCE_CAP 因果对照。

另有 P1：DC 有九项 signed conversion/part-select `VER-318` warning，并启用了
sequential output inversion；VCS/RTL 独立评审已通过，但仍应使用 emitted SVF 跑
Formality。当前只允许引用 service-only logic/pre-macro 面积与时序，不允许声称
complete FC2/FFN、paper-ready PPA、physical/system speedup 或 headline。

`docs/359_DATE终局冻结_20260813.md` SHA256 仍为
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
