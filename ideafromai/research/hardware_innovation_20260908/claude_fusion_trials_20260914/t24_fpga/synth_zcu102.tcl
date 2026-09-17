# T24：C1 门核 + FX 基线 + plane_ser 的 Kria KV260（xczu5ev-sfvc784-2-e，
# FireFly-S/T 同器件）out-of-context 综合。口径对齐 FireFly 系（Vivado 报告）；
# 目标时钟 250 MHz（4 ns），实际 Fmax 由 report_timing_summary 给出。
# 注：本机 Vivado Standard 版不支持 xczu9eg（ZCU102）；KV260 为 FireFly 系
# 对比锚点器件，且 eventflow_zcu102（M803）后续可在远端 Vivado 上做 ZCU102 口径。
# 用法：cd claude_fusion_trials_20260914 && vivado -mode batch -source t24_fpga/synth_zcu102.tcl

set ROOT [file normalize [file join [file dirname [info script]] ..]]
cd $ROOT

proc run_one {top} {
    set part xczu5ev-sfvc784-2-e
    create_project -force t24_$top ./t24_fpga/obj_$top -part $part
    read_verilog -sv t15_synth/$top.sv
    read_xdc t24_fpga/clk4ns.xdc
    synth_design -top $top -mode out_of_context -part $part
    report_utilization -file t24_fpga/${top}_util.rpt
    report_timing_summary -delay_type max -file t24_fpga/${top}_timing.rpt
    close_project
}

run_one cert_gate_bitl_synth
run_one fx_gate_synth
run_one plane_ser
puts "T24_SYNTH_ALL_DONE"
