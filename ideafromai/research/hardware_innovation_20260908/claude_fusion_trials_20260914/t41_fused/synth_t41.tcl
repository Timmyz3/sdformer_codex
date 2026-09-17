# T41 综合：融合基线（送 10 判决位）与 T39 同一器件/同一流程，隔离"把 A 折进生产者"的面积代价。
# 基线：T39 变体 A（+SGLR+sop 粗区间）= 8,387 LUT / 1,751 FF / 0 DSP / +0.985ns
#       T39 变体 B（+SGLR）= 5,042 LUT / 1,756 FF / 0 DSP / +1.053ns
#       T25 k4 = 5,030 LUT / 1,756 FF / 0 DSP / +1.053ns
# 用法：vivado -mode batch -source t41_fused/synth_t41.tcl
set ROOT [file normalize [file join [file dirname [info script]] ..]]
cd $ROOT
set part xczu5ev-sfvc784-2-e

create_project -force t41_fused/objsyn_fused ./t41_fused/objsyn_fused -part $part
read_verilog -sv t41_fused/t41_fused_gate.sv
read_xdc t24_fpga/clk4ns.xdc
synth_design -top t41_fused_gate -mode out_of_context -part $part
report_utilization -file t41_fused/fused_util.rpt
report_timing_summary -delay_type max -file t41_fused/fused_timing.rpt
close_project
puts "T41_SYNTH_DONE"
