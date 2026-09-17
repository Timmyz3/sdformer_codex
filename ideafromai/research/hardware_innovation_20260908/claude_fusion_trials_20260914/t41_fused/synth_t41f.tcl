# T41 补充综合：把融合基线放进与 T25/T39 **可比的时序口径**，并给出 LUT-only 面积上界。
#
# 为什么需要这一步：
#   1) t41_fused_gate 全是 input→FF 的组合路径，没有任何 reg→reg 路径，所以
#      report_timing_summary 给出 WNS = NA（TNS Total Endpoints 为空），无法与
#      T25 (WNS +1.053, vtop_reg→dec_raw_reg) / T39 (+0.985, vtop_reg→dec_raw_reg) 并列。
#      改用 t41_if.xdc 后，本模块的组合逻辑与 T25/T39 的 reg→reg 路径拿到**同一条 4ns
#      预算**（launch 0 / capture 4），WNS 才可直接比较。
#   2) T41 的 40 个乘法被 Vivado 推成 DSP48E2（241 LUT / 40 DSP）；T25/T39 的加法树是
#      LUT 实现（0 DSP）。两种实现的面积不可直接相减，故补一次 `-max_dsp 0` 强制 LUT，
#      给出"若不许用 DSP"的面积上界。
#
# 用法：vivado -mode batch -source t41_fused/synth_t41f.tcl
set ROOT [file normalize [file join [file dirname [info script]] ..]]
cd $ROOT
set part xczu5ev-sfvc784-2-e

proc syn_one {tag args} {
    global part
    create_project -force t41_$tag ./t41_fused/objsyn_$tag -part $part
    read_verilog -sv t41_fused/t41_fused_gate.sv
    read_xdc t41_fused/t41_if.xdc
    synth_design -top t41_fused_gate -mode out_of_context -part $part {*}$args
    report_utilization -file t41_fused/${tag}_util.rpt
    report_timing_summary -delay_type max -file t41_fused/${tag}_timing.rpt
    close_project
}

# 1) 带 DSP（Vivado 默认推断），加接口延时后取真实组合路径
syn_one dsp
# 2) 禁 DSP，纯 LUT 加法树 —— 与 T25/T39 的 0 DSP 口径对齐
syn_one lut -max_dsp 0
puts "T41F_SYNTH_DONE"
