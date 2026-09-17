# T47 综合：把「零插值行走器 Z」与「相位分解行走器 P」放进同一器件、同一流程、同一 4ns 预算，
# 隔离出两者唯一可能不同的东西 —— 控制路径（地址生成 / 有效判定 / 抽头槽位）。
# 算术部分两者逐字相同（同一个 MAC、同一个二值×int8 select），所以面积差就是控制路径差。
#
# 用法：/opt/vivado/2025.1/Vivado/bin/vivado -mode batch -source t47_fused/t47_synth.tcl
set ROOT [file normalize [file join [file dirname [info script]] ..]]
cd $ROOT
set part xczu5ev-sfvc784-2-e

proc syn_one {tag top} {
    global part
    create_project -force t47_$tag ./t47_fused/objsyn_$tag -part $part
    read_verilog -sv t47_fused/t47_deconv_walker.sv
    read_xdc t47_fused/t47_if.xdc
    synth_design -top $top -mode out_of_context -part $part
    report_utilization -file t47_fused/${tag}_util.rpt
    report_timing_summary -delay_type max -file t47_fused/${tag}_timing.rpt
    close_project
}

syn_one z t47_deconv_z
syn_one p t47_deconv_p
puts "T47_SYNTH_DONE"
