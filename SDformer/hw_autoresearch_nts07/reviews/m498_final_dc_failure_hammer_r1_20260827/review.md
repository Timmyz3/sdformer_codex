# M498 最终 DC 失败独立 receipt-blind hammer（r1）

日期：2026-08-27  
裁定：`KILL_CLOSE_M479_FAMILY_PHYSICAL_LINE`  
分数：90/100（证据纪律），0/100（M498 DC 准入）

## 1. 裁定

M498 r1 不能引用为通过的 DC/PPA 点，也不允许再开 r1b 定向电气修复。理由不是只有“三条小网没修好”，而是同时触发了合同中的两个独立永久关闭条件：

1. 五类约束中只有 max-delay 和 min-delay 干净；max-capacitance、max-transition、max-fanout 三类失败。runner 因此以 code 33 fail closed。
2. 映射网表仅有 108 个 `BUFFD1BWP35P140`，低于 runner 的 204 个硬门；预编译时存在的 204 个显式物理 buffer 层次在 compile 期间全部被 `OPT-776` ungroup，映射后不能证明“12 个 branch + 192 个 BUFFD1 leaf”原拓扑存活。

合同 `failure_rule` 明确写明这是 M479-family 的 final retry，任何 design-rule violation、lost buffer tree 或少于要求的证据都永久关闭该线，且 constraints may not be relaxed。因此追加 r1b 会直接违反已冻结的实验规则；本审查不提供可执行修复 TCL。

## 2. 身份与输入复核

- DC runner 状态：`dc.rc=0`，但封装 runner 退出码为 33，`RUN_FAILED_OR_INCOMPLETE.txt` 正确标记 `DO_NOT_CITE`。
- `dc.log` 中 current design 始终为 `m498_segmented_enable_backpressure_safe_parent_queue_pipeline`，未出现 forbidden M479/M476r2 top；mapped Verilog 也只有 M498 core/top 两个 module。
- RTL、wrapper、filelist、SDC、TCL、contract、VCS seal、DC executable、slow/fast library 与 `input_sha256.txt` 一致；runner 中的 exact-SHA 期望也一致。
- 先决 VCS seal 独立 `sha256sum -c` 全通过；receipt 为 `PASS_M498_SEGMENTED_ENABLE_EXACT_VCS`。full regression 与 stale-RAW targeted test 均通过，但该 receipt 已明确 `explicit_physical_tree_after_dc=false`。
- `docs/359_DATE终局冻结_20260813.md` SHA256 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

## 3. 通过与失败的硬指标

| 检查项 | 实测 | 门槛 | 裁定 |
|---|---:|---:|---|
| Cell area | 42,370.649130 um² | <=44,779.2 um² | PASS（余量 2,408.551 um² / 5.379%） |
| 相对 M475 area | 1.13546x | <=1.20x | PASS |
| Sequential cells | 5,508 | exactly 5,508 | PASS |
| Macro/black box | 0 | 0（logic-only） | PASS |
| Setup WNS | +0.0000 ns | >=0 | PASS，但无工程余量 |
| Hold WNS | +0.0101 ns | >=0 | PASS |
| max-delay | clean | clean | PASS |
| min-delay | clean | clean | PASS |
| max-capacitance | 2 violations | clean | FAIL |
| max-transition | 1 violating net | clean | FAIL |
| max-fanout | 3 violating nets | clean | FAIL |
| mapped BUFFD1 | 108 | >=204 | FAIL |

`report_qor` 同样给出 `Nets With Violations=3`、max trans/cap/fanout 分别为 1/2/3，和独立解析一致。

## 4. 三个违规网的定位

| 网 | 映射驱动路径 | fanout | cap (required/actual/slack) | transition (required/actual/slack) | 说明 |
|---|---|---:|---|---|---|
| `u_core/n17470` | `n15349 -> U19407 CKND0 -> n17470` | 80（slack -48） | 0.0446/0.0776/-0.0330 | 0.5280/0.7441/-0.2161 | 同时违反三类；直接驱动 75 个 enable FF pins，另驱动 delay/inverter loads |
| `u_core/n16011` | `n15349 -> n1 -> n17465 -> U10959 CKND0 -> n16011` | 61（slack -29） | clean in all-violators report | clean in all-violators report | 直接驱动 31 个 enable FF pins及大量 inverter loads |
| `u_core/n1` | `n15349 -> U19536 CKND0 -> n1` | 57（slack -25） | 0.0446/0.0479/-0.0033 | clean in all-violators report | 直接驱动 41 个 enable FF pins、组合 loads，并继续派生 n17465/n16011 |

这三条网来自同一 `n15349` enable 根的重新综合分发链，并非三个互不相关的普通数据网。最坏网 `n17470` 已经跨多个 lane 的 row/psum enable；这与源 RTL 声称的每叶最多驱动 13/20 state bits 不一致，说明工具在全局优化/hold 修复中重构了物理分段。

## 5. 拓扑复核

- precompile reference report 中可枚举 `m498_physical_enable_buffer_0..203`，即源结构确实展开了 204 个物理 buffer 实例。
- compile log 对 12 个 branch 与 192 个 leaf wrapper 全部报告 `Ungrouping ... before Pass 1 (OPT-776)`。
- mapped netlist 中 `BUFFD1BWP35P140=108`；即使 `BUFFD0BWP35P140=117`，也不能把两者相加冒充 204 个原显式 BUFFD1，因为这些 cell 已无原实例身份，且网表仍存在 80/61/57 fanout 的 enable 网。
- enable-flop 网表解析得到 701 个带 E pin 的 cell，分布在 30 条 enable 网；其中 `n17470/n1/n16011` 分别直接承担 75/41/31 个 E pin。这是映射后分段树未按合同形态存活的直接证据。
- `check_design_postcompile.rpt` 和 `check_timing_postcompile.rpt` 的孤立 `1` 只是命令返回值输出，不替代五类 `report_constraint`；不得据此写“clean”。

## 6. 为什么不允许 r1b

单纯在三条违规网上插 buffer 在工程上可能减少 fanout/cap/transition，但在本项目合同下不构成合法的“一次定向修复”：

1. 还需把 mapped BUFFD1 从 108 恢复到至少 204，并证明 branch/leaf connectivity；这已超出三网 ECO，是 synthesis hierarchy/topology 策略变更。
2. setup WNS 只有打印精度下的 +0.0000 ns；任何新增 cell/重映射都可能立即破坏 setup。hold 仅 +0.0101 ns，也不足以在没有新 STA 的情况下保证安全。
3. 若用 `set_dont_touch_network`、禁止 ungroup、手工 `insert_buffer` 或改变 compile sequence，都会改变被冻结的 TCL/input SHA；必须成为新实验合同，而原合同明令不再 retry。
4. 放松 max-fanout、transition/capacitance 或 hold uncertainty 绝对禁止。

因此硬停止门是现在：M479/M498 dual-slot stdcell enable-tree 物理线关闭。后续只能把 M498 当作功能/微结构证据，不能当 DC/PPA、性能准入或 DATE headline。若未来另立完全不同的 memory-backed/1RW architecture（例如 M504），必须使用新身份、新合同、新 VCS/SVA、新 DC 门，不能命名为 M498 r1b 或继承本点的 PPA。

## 7. 可引用边界

- 可引用：M498 exact RTL 在 VCS directed/full regression 中保持零周期语义；这只是功能证据。
- 不可引用：42,370.65 um² 作为 clean PPA；M498 3 ns timing clean；显式 12×16 buffer tree survives DC；Conv performance admitted；paper-PPA-ready；system speedup；DATE headline。
- 可作为消融/工程教训：logic-only mapping 在 3 ns 达到 area/seq/setup/hold 门，但分段 enable tree 经综合重构后留下 3 个电气违规网，故 fail closed。

## 8. 审查边界

本 hammer 仅只读审查既有 M498 runner/contract/RTL/VCS/DC 证据；未运行 VCS、DC、PT、Formality 或 GPU；未修改 docs/359。
