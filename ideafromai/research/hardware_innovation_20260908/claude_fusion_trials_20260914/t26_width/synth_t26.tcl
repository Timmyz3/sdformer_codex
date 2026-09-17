# T26 综合：区间数据通路位宽扫描（k4 掩码加法树，同器件同口径）
# 用法：cd claude_fusion_trials_20260914 && vivado -mode batch -source t26_width/synth_t26.tcl
set ROOT [file normalize [file join [file dirname [info script]] ..]]
cd $ROOT
set part xczu5ev-sfvc784-2-e

proc syn_one {top src tag} {
    set part xczu5ev-sfvc784-2-e
    create_project -force t26_$tag ./t26_width/objsyn_$tag -part $part
    read_verilog -sv $src
    read_xdc t24_fpga/clk4ns.xdc
    synth_design -top $top -mode out_of_context -part $part
    report_utilization -file t26_width/${tag}_util.rpt
    report_timing_summary -delay_type max -file t26_width/${tag}_timing.rpt
    close_project
}

foreach W {48 40 38 32} {
    syn_one cert_gate_bitl_synth t26_width/w${W}_k4/cert_gate_bitl_synth.sv w${W}
}
puts "T26_SYNTH_DONE"
