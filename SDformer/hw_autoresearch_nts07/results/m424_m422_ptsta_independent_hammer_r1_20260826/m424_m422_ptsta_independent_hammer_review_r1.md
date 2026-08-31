# M424：M422 PrimeTime 独立打铁评审

结论：**92/100，P0/P1/P2 = 0/1/2；接受 M422 的 3 ns pre-layout data-path STA，准入 real-SAIF PTPX。** 这不是 post-route/SPEF timing、reset recovery/removal signoff、SRAM-inclusive PPA 或论文能耗数字。

独立 PrimeTime W-2024.09-SP3 从冻结 M416 mapped netlist/SDC 重跑，未调用 M422 Tcl 或 parser。结果复现为：slow setup `+0.759794 ns`、fast-min-library hold `+0.017835 ns`，与 M422 四位小数 `+0.7598/+0.0178 ns` 一致；setup/hold 各 `4100/4100`、out_setup/out_hold 各 `1268/1268`，均 0 violated、0 untested。link 成功，constraint violator 为 0，唯一无 input-delay 端口是显式 false-path 的 `reset_n`。关键 setup/hold 路径身份也与 M422 一致。

物理边界被独立确认：ideal 3 ns clock（未 propagated）、`ZeroWireload`、无 `read_parasitics`、20,803 个 flat leaf standard cells、0 hierarchical cell/0 physical macro。它只能写成 logic-only pre-layout timing diagnostic。

## 必须修窄的两处口径

1. `coverage_gate_pass=true` 只覆盖四类数据路径检查，不能理解成所有 timing checks 完成。全表是 `27,136/51,736 = 52%` met：recovery 和 removal 各 4,100 项因异步 reset 无 startpoint clock 而全未测；min-pulse 32,800 项中 16,400 项因条件化异步 `CDN` 弧无 clock 而未测。不能写 recovery/removal signoff 或 all-check coverage。
2. M421 的修正已被 M422 正确继承：可写 1,268 port + 4,100 observable DFF compare points 的 observational equivalence；320 个 unread debug-counter DFF 不是 internal-state equivalence points。禁止写 all internal state equivalence。

## 打铁发现

- **P1：runner Error/Fatal guard 不是 fail-closed。** `! grep -Eq ...` 位于 Bash 取反上下文，`set -e` 不会使匹配结果退出。M422 原始日志实际 0 Error/Fatal，且独立 PT 干净，因此当前数值不被推翻；以后 runner 必须改为显式 `if grep ...; then exit; fi`。M424 独立 runner 已使用修正形式。
- **P2：coverage 字段命名过宽。** 论文表只写 `data-path setup/hold coverage`，并并列列出三类 untested。
- **P2：两次 SDC 2.1 version mismatch warning。** 两次运行完全相同，未影响 link、约束、路径、slack 或 coverage；下次重新导出 SDC 时清理。

## 下一门

准入 real-SAIF PTPX，但必须冻结真实 workload 仿真区间与层次，生成非空 SAIF，在 exact M416 mapped hierarchy 上读取并报告 annotation coverage（当前目标至少 95%）。在 SAIF 和 coverage 通过前不得报 power/energy；即使通过，仍保留 no-SPEF、ideal-clock、0 macro/0 SRAM 的边界，也不得写 system speedup 或 paper-ready PPA。

`docs/359` 未修改，SHA-256：`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
