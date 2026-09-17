# T39 综合：与 T25 treek4 同器件同口径，隔离"T39 两处注入"的面积/时序代价。
# 基线 = t25_k4_gate/treek4_util.rpt（5,030 LUT / 1,756 FF / WNS +1.053ns）。
# 用法：vivado -mode batch -source t39_sglr_rtl/synth_t39.tcl
# 注意：模块名两者相同（cert_gate_bitl_synth），靠文件区分；两个工程分目录。
set ROOT [file normalize [file join [file dirname [info script]] ..]]
cd $ROOT
set part xczu5ev-sfvc784-2-e

proc syn_one {src tag} {
    set part xczu5ev-sfvc784-2-e
    create_project -force t39_$tag ./t39_sglr_rtl/objsyn_$tag -part $part
    read_verilog -sv $src
    read_xdc t24_fpga/clk4ns.xdc
    synth_design -top cert_gate_bitl_synth -mode out_of_context -part $part
    report_utilization -file t39_sglr_rtl/${tag}_util.rpt
    report_timing_summary -delay_type max -file t39_sglr_rtl/${tag}_timing.rpt
    close_project
}

# 1) SGLR + sop 粗区间（T39 设计点）
syn_one t39_sglr_rtl/cert_gate_sglr.sv pp
# 2) 仅 SGLR（消融，隔离 sop 粗区间的面积代价）
syn_one t39_sglr_rtl/cert_gate_sglr_np.sv np
puts "T39_SYNTH_DONE"
