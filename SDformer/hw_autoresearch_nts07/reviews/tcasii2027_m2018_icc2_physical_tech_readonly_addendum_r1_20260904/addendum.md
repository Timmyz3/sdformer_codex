# TCAS-II 2027 M2018 ICC2 物理工艺只读补充审计 r1

日期：2026-09-04（Asia/Shanghai）  
对象：M2018 ordinary-LRU4 与 TSBG-B4 两轴 28-nm macro-free matched P&R/CTS/route 可行性  
模式：只读文件系统审计；没有执行 ICC2、ICC2 Library Manager、StarRC、DC、PT、PTPX、Formality、VCS/simv、GPU 或 license query；没有修改原审计、既有源码、结果、论文或 `docs/359`。

本补充不覆盖原审计，而是修正其中“本地物理 tech 似乎缺失”的先验：本机确实存在足够强的 TSMC 28-nm 标准单元物理/时序/RC 源，可启动一次受控的 ICC2 library-import preflight；但不存在可直接使用的现代 NDM，且尚未由工具验证旧 Milkyway 库能否无损转换。因此裁决是 **GO_CONDITIONAL**，不是已证明可 signoff。

## 一、结论

### GO：允许做什么

允许为 M2018 ordinary-LRU4 与 TSBG-B4 各做一次完全 matched 的 **macro-free logic-island placement/CTS/route/RC/hold-repair**：

- 两轴使用相同 3-ns 时钟、floorplan、利用率、pin placement、routing layers、CTS cell set、MCMM corner、hold-repair 策略和外部 288-KiB SRAM 接口模型；
- 目标是替换当前 `ZeroWireload + ideal clock` 下约 73k 条 hold violation，得到 routed setup/hold、逻辑面积、buffer/clock-tree 税、拥塞/线长及 routed logic power；
- 该结果可把 TCAS-II 的 C2/TSBG 从“prelayout setup-only”提升为“matched post-route logic island”。

### NO-GO：暂时不能声称什么

在真正执行并通过 library checks/P&R gates 之前，不得声称：

- 已有 ready-to-use ICC2 NDM 或已完成 Milkyway-to-NDM 转换；
- LEF+Liberty 单独足以建库；现有 cell LEF 没有 routing `LAYER`、`VIA` 或 `SITE` technology definitions；
- macro-inclusive P&R、全芯片 signoff、SRAM integrated timing/power、EM/IR、LVS 或 tapeout readiness；
- routed timing/area/power 已经成立。

最准确的论文标签是：**macro-free matched post-route logic-island result with an identical external/common 288-KiB SRAM model**。

## 二、本地 ICC2/PDK 证据

| 类别 | 只读核验结果 | 意义与边界 |
|---|---:|---|
| ICC2 | `icc2_shell` 与 `icc2_lm_shell` 存在且 executable | 仅检查文件元数据，未启动工具或查 license |
| StarRC | `StarXtract`、`grdgenxo` 存在 | 同上，未执行 |
| 预建 NDM/NLIB | `0 / 0` | 没有直接可开的现代 reference library |
| 目标单元时序 | 目标 family 内 96 `.db` + 96 `.lib`；所选 TT/SS/FF `.lib` 各 1,000 cells | 足以给 Library Manager/ICC2 提供逻辑/时序视图；实际 link/check 未运行 |
| cell LEF | 1,044 `MACRO`；0 technology `LAYER`、0 `VIA`、0 `SITE` definitions；1,044 个 `SITE core` references | 仅有 cell abstracts，不能单独提供 P&R routing technology |
| GDS | `tcbn28hpcplusbwp35p140.gds`，14,187,464 B | 标准单元版图源存在；字符串覆盖不等于 stream-in/DRC 验证 |
| Milkyway | frame-only 库有 1,044 个 `FRAM/*:1`；`lib_1` 明示 TSMC N28 P&R tech | 是最可信的 Synopsys physical source；仍需 ICC2 实际导入验证 |
| 物理/CTS cells | antenna、boundary、decap、fill、tap、tie、buffer/inverter/clock-buffer 均存在 | 支撑常规 floorplan/CTS/hold repair 的 cell inventory |
| RC | 1P9M 6X1Z1U typical `.nxtgrd` + ICC layer map + ITF 都存在 | ICC2 文档明确允许 `read_parasitic_tech -tlup` 读 common NXTGRD；未执行 sanity check |
| Virtuoso techfile | 对应 1P9M 6X1Z1U OA `.tf` 存在 | 只用于 stack 交叉核验；禁止直接当作 ICC2 Synopsys technology file |

本地 ICC2 `create_lib` 文档明确：`-ref_libs` 可以接收 physical libraries、LEF 和 Milkyway libraries，并依据 `link_library` 自动构建 cell libraries。故最小可行路线是 **旧 Milkyway frame library + TT/SS/FF DB 自动导入**，不是“cell LEF + DB 直接开跑”。

本地 ICC2 `read_parasitic_tech` 文档也明确：`-tlup` 参数既可读 TLUPlus，也可直接读 common NXTGRD，并支持 `-layermap` 与 `-sanity_check advanced`。因此不必先自行从 ITF 生成 TLUPlus；优先使用 foundry NXTGRD 与配套 ICC map，减少一个派生工件。

## 三、M2018 网表的物理完整性预检

冻结输入为 M2029 两轴 mapped netlist/SDC。只读解析得到：

| 轴 | leaf cells | sequential cells | DC cell area | setup WNS | hold WNS | hold violations | public ports |
|---|---:|---:|---:|---:|---:|---:|---:|
| ordinary-LRU4 | 266,394 | 74,460 | 249,710.452 um2 | +0.0264 ns | -0.0164 ns | 73,372 | 4,551 |
| TSBG-B4 | 266,459 | 74,460 | 249,739.810 um2 | +0.0688 ns | -0.0164 ns | 73,362 | 4,551 |

两份网表合计使用 94 种标准单元 master；静态覆盖如下：

- cell LEF：94/94；
- TT Liberty：94/94；
- SS Liberty：94/94；
- FF Liberty：94/94；
- Milkyway FRAM：94/94；
- GDS 文件字符串可见性：94/94。

两份 mapped netlist 都只有一个顶层 module，未实例化 SRAM macro 或未知子模块；共同 288-KiB weight SRAM 通过外部端口建模。因此从网表结构看，macro-free P&R 是可行的。GDS 字符串检查只说明 cell name 存在，不替代 ICC2 `check_lib`、stream-in、DRC 或 logical/physical cell matching。

物理风险也很清楚：266k leaf cells、74k flops 和 4,551 个 boundary ports 不是一个轻量小岛。若两轴各自自由优化 pin placement，TSBG/ordinary 的 wire/timing/power 比较会失去公平性。必须冻结同一 core boundary、同一 pin order/grouping 与相同 route constraints。

当前 mapped SDC 中保留 `set_wire_load_model ZeroWireload`。物理 run 不得把它当 routed RC；应过滤这一命令，以 NXTGRD/route extraction 为准。时钟 `clk_core`、3-ns period、setup uncertainty 0.2 ns、hold uncertainty 0.05 ns 和 0.25-ns IO delay 可作为 matched 约束起点，但必须在 CTS 后使用 propagated clock。

## 四、最小 source-only 路线

以下是执行合同，不是已执行脚本。

### Gate 0：冻结输入身份

先封存两轴 mapped `.v/.sdc`、TT/SS/FF `.db`、Milkyway frame library、cell LEF/GDS、NXTGRD 与 ICC map 的 SHA。任何一个输入变化都必须新建 run identity。

### Gate 1：只建 reference/design library，先不 place

后续获得 EDA 授权后，在 ICC2/Library Manager 内：

```tcl
set_app_var link_library [list $TT_DB $SS_DB $FF_DB]
set_app_var lib.configuration.local_output_dir $LOCAL_LIB_CACHE
create_lib -ref_libs [list $MW_FRAME_LIB] $DESIGN_LIB
```

若 legacy Milkyway 自动导入不能正确生成 technology/reference library，立即 STOP。不得把 Virtuoso `.tf` 塞给 ICC2“试出来”。正确补救是使用 ICC2 Library Manager 的受支持 Milkyway conversion/import 路径，或从该 Milkyway 库导出/取得受支持的 standalone Synopsys technology library，再通过 `-technology` 或 `-use_technology_lib` 建库。

Gate 1 必须通过：

- `check_lib` 无 unresolved/error；
- 94/94 mapped masters 的 logical/physical views 匹配；
- `core` site、M1--M9/AP routing layers、VIA1--VIA8/RV 与 antenna/fill/tap/tie/CTS cells 可用；
- DB 与 physical cell names/units/orientations 一致。

### Gate 2：导入 parasitic technology

```tcl
read_parasitic_tech \
  -tlup $NXTGRD_1P9M_6X1Z1U_TYP \
  -layermap $ICC_LAYER_MAP_1P9M_6X1Z1U \
  -name n28_1p9m_6x1z1u_typ \
  -sanity_check advanced
```

任何 conducting/via layer 缺失、map mismatch 或 advanced sanity failure 都 fail closed。ITF 只作 foundry identity 交叉核验；本轮不再生成新的 TLUPlus。

### Gate 3：做两轴 matched P&R

- 读取 M2029 exact-SHA mapped netlist并 link；读取去掉 `ZeroWireload` 的 physical-safe SDC。
- 两轴共享同一 die/core、aspect ratio、target utilization、row/site、placement blockage、pin group/order、routing layers、CTS cells/targets 和 optimization switches。
- 4,551 个 ports 采用确定性的 bus-group pin placement；禁止 ordinary 与 TSBG 各自独立优化 pins。
- MCMM 至少用 SSG 0.9 V 125 C 做 setup、FFG 1.05 V -40 C 做 hold；TT 0.9 V 25 C 用于 routed power。
- 同顺序运行 placement optimization、CTS/clock optimization、routing/route optimization 和 hold repair；CTS 后时钟必须 propagated。

### Gate 4：准入与论文输出

两轴都需满足：

1. no unresolved cells/ports；library/parasitic checks PASS；
2. routed connectivity/DRC 无 fatal；
3. setup WNS 与 hold WNS 都 `>= 0`；
4. 物理优化后的网表重新做等价检查；
5. 同表报告 routed standard-cell area、hold/setup buffers、clock-tree cells/area、wirelength、congestion、setup/hold、logic power；
6. common 288-KiB SRAM 容量、面积、leakage 两轴完全相同；仅 dynamic bank activation 按各自真实读请求分列建模。

若 Gate 1 失败，本轮结论降为 NO-GO_PNR_TECH_IMPORT；若 Gate 1/2 通过而两轴 routing/hold 未闭合，则可保留为 physical-feasibility negative result，但不得替换现有 DC 数字。若全部通过，可将 C2/TSBG 主表升级为 matched post-route logic-island 结果。

## 五、对 TCAS-II 主线的直接影响

这项物理收口比新增稀疏 idea 更能提高 TCAS-II 匹配度：

- C1 保留“容量约束的 single-port product capture”；
- C2 保留“typed signed K8 的等带宽面积效率”；
- TSBG 作为 C2 的 storage-aware scheduling specialization；
- 新增物理层证据回答 reviewer 最可能追问的两个问题：`4.55x throughput/mm2 是否能在真实布线/时钟树后保住？`、`TSBG 的几乎零逻辑面积税是否会被 hold/route/power 反转？`

它不改变算法/checkpoint，也不替换 C1/C2/C3。它只把 C2/TSBG 从逻辑综合证据推进到 TCAS-II 更看重的 physical-design evidence。即使 P&R 成功，仍应避免宣称 full accelerator：共同 SRAM 尚未物理集成，full-network/system metrics 仍需独立证据。

## 六、残余风险与置信度

| 风险 | 严重度 | 置信度 | 处置 |
|---|---|---|---|
| ICC2 2023 是否能无损导入 2010/2013-era Milkyway 库 | High | Medium | Gate 1 先行；失败立即停止，不做虚假 techfile 替代 |
| 4,551 pins 与 266k cells 的 routing/congestion | High | High | 两轴相同 floorplan/pins；先小步 place/congestion gate，再完整 route |
| common 288-KiB SRAM 未集成 | High | High | 论文始终标 external/common model；面积/leakage同量加入两轴 |
| 只有 typical NXTGRD 被本次选中 | Medium | High | routed feasibility先用 typical；最终 setup/hold corner parasitic 需补 max/min RC 或明确限制 |
| GDS 字符串覆盖不是版图验证 | Medium | High | 必须由 `check_lib`/stream/DRC 替代，不能直接引用为 signoff |

## 最终裁决

**GO_CONDITIONAL_M2018_MATCHED_MACRO_FREE_ICC2_PNR**：本机已有足够的 source material 支撑一次受控的 Library Manager/ICC2 import preflight，并且 M2018 的 94/94 cell-view 静态覆盖与 macro-free 网表结构支持两轴 matched P&R。第一步必须是 Milkyway/DB/NXTGRD 的 library-import 与 sanity gate，而不是直接 route。只有实际 Gate 1--4 全部通过后，才能把 C2/TSBG 升级为 routed hold-clean/power evidence；否则保持现有 prelayout/hold-open 边界。

