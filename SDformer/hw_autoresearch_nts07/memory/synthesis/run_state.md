run_id:      synthesis_20260718_typed_adaptive_open_structure
design_name: gatestack_single_context_execution_top
tool:        yosys（DC/Formality交付准备；当前环境无dc_shell/fm_shell/目标库）
start_time:  2026-07-18T18:00:00Z
last_stage:  adaptive_residency_parameterized_open_structure_lec_complete_dc_blocked

result:
  constraint_audit: PASS
  yosys_structure_static_ipd_res1: PASS_4191_cells_13_mem_v2
  yosys_structure_adaptive_res0: PASS_4958_cells_11_mem_v2
  yosys_structure_adaptive_res1: PASS_5249_cells_14_mem_v2
  yosys_lec_adaptive_res0: PASS_4762_proven_0_unproven
  yosys_lec_adaptive_res1: PASS_4832_proven_0_unproven
  dc: BLOCKED_no_dc_shell_no_target_library
  formality: BLOCKED_no_fm_shell_no_dc_mapped_netlist
  saif_power: BLOCKED_no_vcd2saif_no_target_library

constraints:
  clock_mhz: 500
  area_um2: 500000（探索预算，未按工艺冻结）
  power_mw: 100（探索预算，未按工艺冻结）
  wns_ns_target: 0
  tns_ns_target: 0
  fanout_max: 32
  clock_uncertainty_ps: 200

# 2026-07-20 HATF同步Bias后开放映射更新

- Nangate45未约束逻辑映射：OUT_TILE32/64/96/128 area为90159.902/146983.886/203975.716/261017.820。
- 对应真实RTL加速1.000/1.861/2.634/3.281，面积归一吞吐1.000/1.142/1.164/1.133。
- 每个候选仍有3个未映射mem_v2；无SDC/STA/SAIF/SRAM macro，证据等级仅为开放目标库逻辑映射代理。

# 2026-07-20 BSF 开放映射消融

- 固定HATF96，仅切换BIAS_STATIONARY_ENABLE：baseline/BSF logic area为204042.748/225972.852，cells为148243/160151。
- BSF周期加速1.067x但逻辑面积增加10.748%，面积归一吞吐0.963x；flop-based实现不晋级默认配置。
- 结果保留3个未映射mem_v2，不含SDC、STA、SAIF或SRAM macro，不能作为目标PPA。

# 2026-07-20 Central96与3xIndependent32同库映射

- 固定96 product lane、同一19-file RTL集合和Nangate45开放库：Central96 area203921.452/cells148134/mem_v2=3；Independent area270665.640/cells190696/mem_v2=9。
- Central开放逻辑面积减少24.659%、cells减少22.319%；仅为logic proxy。
- mem_v2未计面积且宽度/用途不同；无SDC、STA、SAIF、macro和布局布线，不得写成总面积或EDP结果。

# 2026-07-20 DCTF前端开放映射

- Nangate45无约束代理：完整term adapter area5731.236/cells3886/mem_v2=1；Q2 fabric area4475.184/cells2722/mem_v2=0。
- 两者logic area合计10206.420；adapter token buffer memory面积未计。
- 该结果不含bank executor/Acc/SRAM/SDC/STA/SAIF，只能作为前端开销筛选。

# 2026-07-20 DCTF32 Bank Executor开放映射

- 同源Nangate45无约束映射中，对齐后的32-lane product engine为area 20040.706/cells 15409，DCTF32 executor为area 20367.886/cells 15643。
- executor净增area 327.180（1.633%）、234 cells（1.519%），两者`mem_v2=0`且`check -assert`通过。
- 差值混合command协议、epoch隔离、term内product驻留、Acc路由、完成控制和顶层可观察性优化，不得解释为纯路由面积。
- 无SDC、STA、SAIF、SRAM macro、DC和布局布线，证据等级仍为开放库logic proxy。

# 2026-07-20 DCTF96三Bank Term Datapath开放映射

- Q2/TOKENS162/OUT_TILE32完整flatten top为Nangate45 area 74071.690、55216 cells、5个未映射`mem_v2`。
- 既有3*executor+adapter+Q2 fabric算术和为area 71310.078、53537 cells、1个`mem_v2`。
- 集成净差为area +2761.612（+3.873%）、cells +1679（+3.136%）；差值混合tracker、地址合同、跨层优化和可观察输出裁剪，不能称纯协调器面积。
- mapped网表非空、0 process，除`mem_v2`外无未映射美元单元；无SDC/STA/SAIF/SRAM/DC/P&R。

# 2026-07-20 DCTF96完整Projection开放映射启动

- 完整bank-local projection已通过generic hierarchy/check/stat：3996 wires、158508 wire bits、11 memories、499254 memory bits、3848 cells、0 process。
- Nangate45完整flatten独立映射已启动；只作为无SDC逻辑代理，不与含decoder的Independent wrapper直接比较。

# 2026-07-20 DCTF96完整Projection开放映射完成

- Q2/TOKENS162/每bank OUT_TILE32完整flatten为area 182719.124、135045 cells、11个`$mem_v2`、0 process。
- 映射网表19566055字节，除`$mem_v2`外无未映射美元单元；汇总器单元测试2项PASS。
- 顶层从term/event开始，不含decoder；无SDC/STA/SAIF/SRAM/DC/P&R，不能与含decoder的Independent wrapper直接宣称面积优势。
