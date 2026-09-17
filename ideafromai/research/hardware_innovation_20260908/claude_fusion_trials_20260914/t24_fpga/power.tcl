# T24 二期：实现 + read_vcd（真实开关活动）+ report_power
# 用法：cd claude_fusion_trials_20260914 && vivado -mode batch -source t24_fpga/power.tcl
set ROOT [file normalize [file join [file dirname [info script]] ..]]
cd $ROOT
set part xczu5ev-sfvc784-2-e

proc power_one {top vcd} {
    set part xczu5ev-sfvc784-2-e
    create_project -force t24pw_$top ./t24_fpga/objpw_$top -part $part
    read_verilog -sv t15_synth/$top.sv
    read_xdc t24_fpga/clk4ns.xdc
    synth_design -top $top -mode out_of_context -part $part
    opt_design
    place_design
    phys_opt_design
    route_design
    report_utilization -file t24_fpga/${top}_impl_util.rpt
    report_timing_summary -delay_type max -file t24_fpga/${top}_impl_timing.rpt
    read_vcd -strip_path TOP $vcd
    report_power -file t24_fpga/${top}_power.rpt
    close_project
}

power_one cert_gate_bitl_synth t24_fpga/cert_s0_fix.vcd
power_one fx_gate_synth t24_fpga/fx_s0_fix.vcd
puts "T24_POWER_DONE"
