# DATE硬件Design Compiler与PrimeTime PX交付包

## 当前结论

本目录提供当前Motion/Local5论文切片、旧H67/H68增量注意力行核及GateStack单context执行顶层的DC读取、约束、综合、工件审计、Formality和PrimeTime PX交接脚本。
H68在部署时与TTX同构，`h68_castling_deploy_top` 不包含训练期Castling矩阵分支；H67增加
时间配对K和Motion-XOR计数。两者共享Q1.7 9-bit gate、整数Shiftmax、占用分数类扫描和
稀疏gated-K输出协议。

H67/H68逐位RTL Shiftmax valid825已经完成：H67 AEE为`1.462688`，相对原部署变化
`+0.000055`；H68 AEE为`1.472654`，变化`+0.001167`。当前LUT、舍入和center省略顺序可作为
本轮DC输入冻结。开放工具回归覆盖8/162 token、H67/H68、断言和行级映射网表回灌。

本机没有 `dc_shell`、目标 `.db/.lib` 和SRAM宏，因此当前不能宣称获得真实WNS、面积、
功耗或达到流片签核。`500 MHz / 500000 um2 / 100 mW` 是架构探索口径，不是用户或工艺
签核预算。当前顶层是attention row子系统，不是包含投影、ATLIF、残差、DMA和片上网络的
完整SDformer加速器。

## 当前双线冻结顶层

| 顶层 | 冻结内容 | 证据边界 |
|---|---|---|
| `h67_fixed2s_mssb5_dc_top` | MSSB5 score-front + Fixed2S，`QUOTIENT_ENABLE=0` | Motion T450 attention-row slice |
| `h67_rqtb2s_mssb5_dc_top` | 相同前后端，仅`QUOTIENT_ENABLE=1` | Motion T450 attention-row slice |
| `local5_unified_out2_dc_top` | Q-silent+ident-K+overlap、FCSR、TCFM5、`OUT_DIM=2` | score→Acc32 tile，不是encoder |

Motion两个顶层有相同端口、FIFO、MSSB5前端、K存储、SCS/Shiftmax和输出接口，适合隔离时间商逻辑的PPA增量。Local5的PPA对象明确冻结为手写cross-r1 FCSR（`RELATION_SCHED_MODE=0`）和TCFM5；拓扑编译生成器仅作为同一退休合同的验证证据，当前不属于该综合顶层。配置身份由`scripts/audit_date_dual_handoff.py`及SHA清单审计。

## 文件

- `constraints/h67_h68_500mhz.sdc`：单时钟探索SDC，完整覆盖输入/输出时序和负载。
- `scripts/run_dc.tcl`：双顶层DC综合、网表、DDC、SDC、时序、面积和功耗报告流程。
- `run_dc.sh`：检查工具和工艺库后调用DC；缺少条件时明确失败，不生成伪结果。
- `scripts/audit_dc_artifacts.py`：检查DC网表、SVF和报告工件是否完整，不替代QoR签核。
- `run_formality.sh`：使用DC生成的SVF、映射网表和同一工艺库执行正式等价验证。
- `run_ptsta.sh`：读取DC网表/SDC并可选读取布局布线SPEF，执行独立PrimeTime STA。
- `run_ptpx.sh`：要求真实trace SAIF，并可选读取布局布线SPEF后执行PrimeTime PX；缺任一关键输入即失败。
- `run_local5_activity.sh`：用冻结Local5 DC wrapper和真实向量产生限定测量窗口VCD及活动合同。
- `run_motion_activity.sh`：分别用冻结Fixed2S/RQTB2S wrapper和真实T450行向量产生VCD及活动合同。
- `scripts/compare_motion_activity_contracts.py`：逐行核对两份Motion活动合同的trace、equal和emitted身份。
- `filelists/date_motion_2s.f`、`filelists/date_local5_out2.f`：当前双线冻结源文件集。
- `constraints/date_dual_core.sdc`：默认3ns探索约束，可用`CLOCK_PERIOD_NS`覆盖；不是签核约束。
- `scripts/run_yosys_generic.sh`：无工艺库的结构综合与通用门级网表。
- `scripts/run_yosys_lec.sh`：实验性RTL与通用综合网表顺序等价检查；当前全顶层运行超时，
  不能替代Formality。
- `run_open_checks.sh`：当前机器可执行的开放工具回归；默认不把未关闭的Yosys LEC记为通过。

## 正式DC调用

```bash
cd hw_autoresearch_nts07
python3 dc_handoff/scripts/audit_date_dual_handoff.py \
  --root . \
  --output dc_handoff/runs/date_dual_handoff_audit.json

DESIGN_NAME=h67_fixed2s_mssb5_dc_top \
LIB_DB=/path/to/ss_corner.db \
OPERATING_CONDITION=ss_0p9v_125c \
CLOCK_PERIOD_NS=3.0 \
dc_handoff/run_dc.sh

DESIGN_NAME=h67_rqtb2s_mssb5_dc_top \
LIB_DB=/path/to/ss_corner.db \
OPERATING_CONDITION=ss_0p9v_125c \
CLOCK_PERIOD_NS=3.0 \
dc_handoff/run_dc.sh

DESIGN_NAME=local5_unified_out2_dc_top \
LIB_DB=/path/to/ss_corner.db \
OPERATING_CONDITION=ss_0p9v_125c \
CLOCK_PERIOD_NS=3.0 \
dc_handoff/run_dc.sh

DESIGN_NAME=h67_attention_top \
LIB_DB=/path/to/ss_corner.db \
OPERATING_CONDITION=ss_0p9v_125c \
dc_handoff/run_dc.sh

DESIGN_NAME=h67_attention_top \
LIB_DB=/path/to/ss_corner.db \
DC_RUN_DIR=dc_handoff/runs/h67_attention_top \
dc_handoff/run_formality.sh

DESIGN_NAME=h68_castling_deploy_top \
LIB_DB=/path/to/ss_corner.db \
OPERATING_CONDITION=ss_0p9v_125c \
dc_handoff/run_dc.sh

DESIGN_NAME=gatestack_single_context_execution_top \
LIB_DB=/path/to/ss_corner.db \
OPERATING_CONDITION=ss_0p9v_125c \
dc_handoff/run_dc.sh
```

双线每个DC顶层完成后，使用相同`.db`运行Formality：

```bash
DESIGN_NAME=h67_rqtb2s_mssb5_dc_top \
LIB_DB=/path/to/ss_corner.db \
DC_RUN_DIR=dc_handoff/runs/h67_rqtb2s_mssb5_dc_top \
dc_handoff/run_formality.sh
```

Formality脚本将`verify`返回值写入`reports/formality_status.txt`；compare失败会使shell返回非零，不能仅凭报告文件存在判为通过。

PrimeTime STA需要先对综合网表做pre-layout检查；获得OpenROAD或商业P&R的SPEF后，必须用相同角重新运行post-route检查：

```bash
DESIGN_NAME=local5_unified_out2_dc_top \
LIB_DB=/path/to/ss_corner.db \
OPERATING_CONDITION=ss_0p9v_125c \
CORNER_ROLE=setup \
DC_RUN_DIR=dc_handoff/runs/local5_unified_out2_dc_top \
NETLIST_FILE=/path/to/pnr/local5_unified_out2.v \
SPEF_FILE=/path/to/local5_unified_out2.spef \
dc_handoff/run_ptsta.sh
```

只要提供`SPEF_FILE`，`NETLIST_FILE`就成为必填项，且必须是产生该SPEF的同一P&R网表，不能把DC综合网表与后布局SPEF混用。setup/hold应使用各自库与`OPERATING_CONDITION`分开运行；单库同时报告max/min不构成MMMC签核。

真实trace SAIF必须来自与综合网表顶层对应的RTL回放，并记录trace SHA、仿真器、层次前缀和窗口集合。PTPX调用示例：

```bash
DESIGN_NAME=h67_rqtb2s_mssb5_dc_top \
LIB_DB=/path/to/ss_corner.db \
DC_RUN_DIR=dc_handoff/runs/h67_rqtb2s_mssb5_dc_top \
SAIF_FILE=/path/to/h67_rqtb2s_real_trace.saif \
SAIF_INSTANCE=TOP/tb_h67_motion_dc_activity/g_rqtb/dut \
SAIF_MANIFEST=/path/to/h67_rqtb2s_real_trace_manifest.json \
OPERATING_CONDITION=tt_0p9v_25c \
CORNER_ROLE=power \
MIN_SAIF_COVERAGE_PCT=95.0 \
NETLIST_FILE=/path/to/pnr/h67_rqtb2s.v \
SPEF_FILE=/path/to/h67_rqtb2s.spef \
dc_handoff/run_ptpx.sh
```

`SAIF_MANIFEST`必须绑定设计名、SAIF SHA、原始trace SHA、仿真器、经VCD头实际核对的`strip_path`、`busy_cycles`、完整测量周期、测量窗口类型和trace范围。PTPX只接受工作负载特定的论文功耗合同：Motion使用`paper_power_compute`，必须是公平LFSR下完整138行并命中`112589/94891`、slot `62100/34099`和equal `28001`；Local5当前只接受连续的100-group `paper_power_with_io`，busy `155791`且至少30组非平凡。单组烟测、旧无反压Motion VCD和多段busy-only Local5 VCD不能进入论文功耗。解析器直接积分VCD中的`dump_active`，要求持续时间等于日志测量周期，论文合同还要求单一连续区间。PTPX后审计默认要求至少95%的可解析`SAIF annotation coverage`，并要求`ptpx_unannotated.rpt`明确给出零个未注释对象；格式未知或非零均失败。其他coverage字段不会被误读。没有SPEF时只能得到pre-layout/vector-based power；没有真实SAIF时DC不运行`report_power`，不得用默认翻转率替代论文功耗结果。`report_power`必须与实际周期结合换算每个head-row/window能量，不能只比较瞬时mW。

同样地，若DC探索运行设置`SAIF_FILE`，脚本也强制要求`SAIF_INSTANCE`与`SAIF_MANIFEST`；`PPA_ADMISSION=1`时还拒绝非代表性烟测。但DC阶段没有替代PTPX注释覆盖率门槛，因此该`report_power`仍不是论文功耗终值。未提供SAIF时只写`power_scope.rpt: NO_SAIF_POWER_NOT_RUN`，不生成默认翻转率功耗文件。

Local5 wrapper 活动文件可直接复现：

```bash
VECTOR_DIR=tb_qfit/vectors/local5_joint_ep29_score_projection_realw_sample100_population_v1_20260813 \
RUN_GROUPS=100 DUMP_START_GROUP=0 DUMP_GROUPS=100 \
DUMP_SCOPE=full ACTIVITY_PURPOSE=paper_power_with_io \
OUTPUT_DIR=dc_handoff/runs/local5_dc_activity_full_population100 \
dc_handoff/run_local5_activity.sh
```

该脚本的VCD层次前缀冻结为`TOP/tb_qfit_local5_score_projection_postg0/g_dc_wrapper/dut`。`DUMP_SCOPE=busy`只覆盖projection执行，`DUMP_SCOPE=full`覆盖权重装载、计算和Acc32读回；两者必须分列。全静默group0只允许`ACTIVITY_PURPOSE=identity_smoke`。到有Synopsys工具的服务器后，先用`vcd2saif`转换，再执行：

```bash
python3 dc_handoff/scripts/make_saif_manifest.py \
  --activity-contract dc_handoff/runs/local5_dc_activity_full_population100/activity_contract.json \
  --saif /path/to/local5_unified_out2_dc_top.saif \
  --output /path/to/local5_unified_out2_dc_top_saif_manifest.json
```

Motion必须对Fixed2S与RQTB2S分别运行`run_motion_activity.sh`，不能复用Local5或双DUT testbench的SAIF。论文功耗活动必须运行同一138行集合、相同公平LFSR的`paper_power_compute`；当前准入合同已经逐项锁定`112589/94891`、slot和equal账本。旧无反压活动目录仅保留为`identity_smoke`。

`scripts/audit_synopsys_postrun.py`检查工件齐全、工具显式`Error/FATAL`、PTPX活动注释覆盖率下限和零未注释对象。它仍不会自动把WNS/TNS、未约束路径、功耗或违例判为签核，以上项目必须逐份报告审阅。每次DC/Formality/PT运行还会归档RTL/库/网表/SDC/SPEF/SAIF SHA、git提交和角点信息。

## 宏库与论文PPA准入

`MACRO_DBS`接受冒号分隔的SRAM/RF `.db`并加入DC/PT/Formality链接库。普通探索运行可不提供；若设置`PPA_ADMISSION=1`，DC脚本会强制要求`OPERATING_CONDITION`、`MACRO_DBS`和逗号分隔的`EXPECTED_MACRO_REFS`，并在`report_reference`中逐一核验宏实例。当前冻结顶层仍使用推断数组，尚未完成目标SRAM wrapper映射，因此即使逻辑DC通过也只能称为pre-macro逻辑PPA，不能作为最终ASIC PPA。

GateStack使用 `constraints/gatestack_single_context_500mhz.sdc` 和
`rtl_hitflow/filelist_single_context_execution.f`。若已有与综合网表层次匹配的SAIF，可额外设置
`SAIF_FILE` 与 `SAIF_INSTANCE`；当前仓库的1.1 GiB RTL VCD只是活动审计输入，尚未转换为可签核SAIF。

## DC前仍需冻结

1. 决定active-entry行缓冲使用触发器还是SRAM宏；当前异步读数组会被综合为触发器/多路器。
   35/3项直方图明确按小型寄存器bank实现，不假设单口同步SRAM可直接替换。
2. 提供目标工艺、PVT、SRAM、集成时钟门控单元、DFT和真实trace SAIF/VCD。
3. 在DC中确认无约束路径、WNS/TNS、面积、功耗、最大扇出和映射单元满足正式预算。
4. 运行Formality并获得所有compare point等价；当前Yosys全顶层顺序LEC超时，不得标记关闭。
5. 若论文声称全encoder PPA，必须把投影、93个实际ATLIF调用点、三条skip/最终skip、残差、
   state SRAM和数据搬运纳入更高层模型；本子系统面积不能冒充整网面积。

## 接口注意

- `out_gate_q8` 是历史端口名，实际编码为9-bit无符号Q1.7，`1.0=128`、`2.0=256`。
- K为零的token不产生输出beat，但其score仍进入Shiftmax分母。全折叠行只发`done`，下游必须
  预清零或按`out_token_idx`散写，不能只等待`out_last`。
- H67默认对35类占用位图做两拍查找/乘加流水；H68只有3类，编译期特化为单拍。该选择不是
  运行时混合数据路。
