# GateStack DC 约束、开放综合与 LEC 交付签核

> 2026-07-18更新：开放综合脚本现可用`CSR_FORMAT_FADC24`和`ENABLE_RESIDENCY`冻结真实候选参数。最新Typed Adaptive+IPD-only residency配置为5249个Yosys design cells、14个`$mem_v2`，RTL-to-structure LEC为4832/4832；Adaptive无驻留为4958/11、LEC 4762/4762；静态IPD+驻留为4191/13、LEC 4832/4832。它们只作开放结构代理，目标库DC/PPA结论仍为空。

## 一、当前结论

GateStack 单 context 执行顶层已经具备独立 DC filelist、500 MHz 探索 SDC、可选 SAIF 注释入口、DC fail-fast、Formality入口、Yosys保留memory结构综合和开放LEC。

当前状态是“DC交付输入准备完成，开放结构等价通过”，不是“DC/PPA完成”。本机没有 Synopsys 工具、标准单元库、PVT 和 SRAM macro。

## 二、交付文件

| 文件 | 用途 |
|---|---|
| `rtl_hitflow/filelist_single_context_execution.f` | 28 个综合 RTL 文件 |
| `dc_handoff/constraints/gatestack_single_context_500mhz.sdc` | `clk_core` 单时钟探索约束 |
| `dc_handoff/run_dc.sh` | GateStack DC入口与工具/库fail-fast |
| `dc_handoff/run_formality.sh` | DC mapped netlist正式等价入口 |
| `dc_handoff/scripts/audit_gatestack_sdc.py` | 中文静态约束/环境审计 |
| `dc_handoff/scripts/run_gatestack_yosys_structure.sh` | 保留逻辑memory的开放结构综合 |
| `dc_handoff/scripts/run_gatestack_yosys_lec.sh` | RTL到开放结构网表等价 |

## 三、约束审计

静态审计覆盖 `create_clock`、setup/hold uncertainty、全部数据输入延迟/转换、全部输出延迟/负载、最大扇出、同步复位和 filelist 完整性，全部通过。

`rst_core` 是同步复位，未设置 false path。`500 MHz / 0.2 ns setup uncertainty / 0.05 ns hold uncertainty` 只是探索口径，必须由工艺目标复核。

## 四、开放综合结果

| 指标 | 结果 |
|---|---:|
| 层次结构单元 | 3,982 |
| 逻辑 memory | 12 |
| `$mul` IR | 43 |
| Yosys problem | 0 |
| LEC compare point | 4,559 |
| proven / unproven | 4,559 / 0 |

18 条结构综合 warning 来自三路 replay mux 的小型端口数组被转换为寄存器，主 head-slot、descriptor cache 和 AccTile memory 仍保留为 12 个 `$mem_v2`。43 个 `$mul` 包含 32 个真实 gate×weight lane 和 11 个变量×常数地址表达式；DC 前不能把后者宣传为已消除。

## 五、为什么现在仍不能报 PPA

缺失项包括：

1. 目标 `.db`、operating condition 和 wire-load/physical guidance；
2. 104×64 head-slot、1920×24 descriptor cache、162×1024 AccTile 的 SRAM macro 或 wrapper；
3. RTL VCD 到 mapped hierarchy 的 SAIF 转换与注释覆盖率；
4. DC `check_timing`、unconstrained path、WNS/TNS、area、power、fanout/transition 报告；
5. DC SVF 与 mapped netlist 的 Formality 结果；
6. DFT、UPF、clock-gating cell 和物理拥塞评估。

## 六、正式调用

```bash
DESIGN_NAME=gatestack_single_context_execution_top \
LIB_DB=/path/to/ss_corner.db \
OPERATING_CONDITION=ss_0p9v_125c \
dc_handoff/run_dc.sh

DESIGN_NAME=gatestack_single_context_execution_top \
LIB_DB=/path/to/ss_corner.db \
DC_RUN_DIR=dc_handoff/runs/gatestack_single_context_execution_top \
dc_handoff/run_formality.sh
```

开放结构候选参数调用：

```bash
CSR_FORMAT_FADC24=2 ENABLE_RESIDENCY=1 \
  bash dc_handoff/scripts/run_gatestack_yosys_structure.sh
CSR_FORMAT_FADC24=2 ENABLE_RESIDENCY=1 \
  bash dc_handoff/scripts/run_gatestack_yosys_lec.sh
```

如提供 SAIF，还必须同时给出与 DC current design 一致的 `SAIF_INSTANCE`。当前 1.1 GiB VCD 只证明活动非空和层次热点，不是功耗签核。
