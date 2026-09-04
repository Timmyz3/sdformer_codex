# M444：M441 独立打铁评审

## 结论

评分 **91/100**，`P0=0 / P1=3 / P2=3`。

`GO` 仅限两个窄口径：

1. M433 standalone dual-coread RTL 到它自己的 M439 mapped netlist，在主输入不加约束的条件下，2701/2701 compare points 的顺序等价。
2. 同一 standalone mapped logic 在 3.0 ns、slow setup / fast hold、prelayout 条件下的时序可行性。

`NO-GO`：serial-vs-dual 功能等价、reset recovery/removal 签核、post-route、SRAM/互连、power/energy、完整 Conv、系统倍速以及 paper PPA/headline。

本评审没有读取或采用候选 receipt 中的派生数字；数字全部从原始 Formality/PT 日志、报告、Tcl、runner、合同、网表与 SHA 账本重新计算。

## 身份与封存

- 候选仅为 `dc_handoff/runs/m441_m433_to_m439_formality_ptsta_r1d_20260826`。
- 候选 manifest 33 项全部复核，inner manifest SHA256 为 `9dac4ccd...a7771f3`，outer seal 文件 SHA256 为 `77bee08d...9598618`。
- 16 项 exact-SHA 输入全部与当前文件一致；runner SHA256 为 `99102094...3dddb8`。
- `r1` 与 `r1c` 分别留下 exit 41 / exit 1 的 `FAILED_OR_INCOMPLETE_DO_NOT_CITE`，候选目录没有引用它们，隔离成立。
- 另有旧的 `r1b` 带 `RUN_COMPLETE`；它不是本评审候选。公开 artifact 前应补显式 `SUPERSEDED_DO_NOT_CITE` 或在唯一候选注册表中撤销，避免误引。
- `docs/359_DATE终局冻结_20260813.md` 未修改，SHA256 仍为 `dedde7ce...a7bdfc4`。

## Formality 独立复算

工具为 Synopsys Formality V-2023.12-SP3。reference 是 M433 RTL，implementation 是 **M433 自己的** M439 mapped netlist；Tcl 没有 `set_constant`、case assumption、cut 或 `set_dont_verify`，SVF 也没有约束或黑盒 guidance。

| 项目 | 原始报告复算 |
|---|---:|
| verify return | 1 |
| passing compare points | 2701 |
| ports / DFF | 1353 / 1348 |
| failing / aborted / unverified | 0 / 0 / 0 |
| unmatched / unread | 0 / 0 |
| BBPin / BBNet / Loop / Cut / LAT | 0 / 0 / 0 / 0 / 0 |

因此允许的精确措辞是：

> Unconstrained-primary-input sequential equivalence passes for all 2701 compare points between the M433 RTL and its own M439 mapped netlist.

它不是 M405 serial 与 M433 dual 的功能等价；两者 transaction schedule 不同，M441 也从未读取 M405。

Formality 有默认 nettype 警告，以及两次“375 个无 power-down function 的未链接 power cells”警告。逻辑 compare-point 中 blackbox 为 0，因此不推翻顺序等价；但它进一步说明本里程碑不包含 UPF/低功耗或 power 签核。

## PrimeTime 独立复算

| 项目 | 结果 |
|---|---:|
| clock | 3.0 ns, ideal |
| slow setup corner | `ssg0p9v125c` |
| setup WNS | +0.841061 ns MET |
| fast hold corner | `ffg1p05vm40c` |
| hold WNS | +0.017869 ns MET |
| setup / hold coverage | 1348/1348 / 1348/1348 |
| output setup / hold coverage | 1353/1353 / 1353/1353 |
| constraint violations | 0 |

slow-to-fast `set_min_library` 与显式 OCV max/min operating condition 都在 Tcl 和 report 中一致。报告为 ZeroWireload；Tcl 没有 `read_parasitics`，没有 SPEF，没有 propagated clock，也没有 power/SAIF 命令。链接库仅为两个标准单元角，层次单元为 0，物理 SRAM/macro 与片上互连不在设计范围。

reset 不签核：`reset_n` 没有 clock-relative input delay，并在冻结 SDC 中被 `set_false_path -from`。recovery 与 removal 各 1348 项全部 untested；min-pulse-width 10784 项只有 5392 项 met、5392 项 untested。因此“setup/hold MET”不能扩写为 reset timing signoff。

PT 仅有两次 SDC 2.1 版本匹配警告，无 Error/Fatal；建议后续清理版本声明，但不改变本次窄口径时序结论。

## 问题分级

### P1

1. reset recovery/removal 为 0% tested，异步清零 min-pulse-width 仅 50%；必须增加 reset 时钟/约束并单独签核。
2. ideal clock + ZeroWireload + no SPEF + 0 macro，不可升级为 post-route、SRAM/互连或 paper PPA。
3. 没有 SAIF/PTPX/power；也没有完整 Conv 或系统集成，因此不接纳 energy、full-Conv 或 system headline。

### P2

1. 旧 `r1b` 仍带 `RUN_COMPLETE`，需增加 superseded 标记以维护唯一候选语义。
2. Formality 的 power-cell 警告应在未来 UPF/power 流程解释或清理。
3. PT 的两条 SDC 2.1 版本匹配警告应清理。

## 下一步准入门

若要把 M433 从 standalone logic proof 推向论文 PPA，至少需要：reset recovery/removal/min-pulse 完整约束；带实际布线/时钟树与 SPEF 的 post-route STA；双端口/双 bank SRAM macro 和互连建模；映射门级 SAIF 注释率合格后的 PTPX；最后再把该点放回完整 Conv 周期、带宽和能耗 Pareto。未完成这些门之前，M441 只能作为 M433 的形式等价与 prelayout logic timing 收据。
