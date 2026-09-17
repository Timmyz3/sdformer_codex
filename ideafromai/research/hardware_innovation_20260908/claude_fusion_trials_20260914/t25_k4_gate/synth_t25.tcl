# T25 综合：ROM 版 vs 加法树 full vs 加法树 k4（同器件同口径，隔离"ROM→树"与"A 稀疏"两步）
# 用法：cd claude_fusion_trials_20260914 && vivado -mode batch -source t25_k4_gate/synth_t25.tcl
# 注意 cwd 必须是仓库根（ROM 版模块的 $readmemh 路径相对根）。
set ROOT [file normalize [file join [file dirname [info script]] ..]]
cd $ROOT
set part xczu5ev-sfvc784-2-e

proc syn_one {top src tag} {
    set part xczu5ev-sfvc784-2-e
    create_project -force t25_$tag ./t25_k4_gate/objsyn_$tag -part $part
    read_verilog -sv $src
    read_xdc t24_fpga/clk4ns.xdc
    synth_design -top $top -mode out_of_context -part $part
    report_utilization -file t25_k4_gate/${tag}_util.rpt
    report_timing_summary -delay_type max -file t25_k4_gate/${tag}_timing.rpt
    close_project
}

# 1) ROM 基线（t15 综合版，12.8kb 子集和 ROM）
syn_one cert_gate_bitl_synth t15_synth/cert_gate_bitl_synth.sv rom
# 2) 加法树，稠密 A（10 项/行）——同常数，隔离 ROM→树
syn_one cert_gate_bitl_synth t25_k4_gate/full/cert_gate_bitl_synth.sv treefull
# 3) 加法树，k=4 掩码 A（4 项/行）——T25 设计点
syn_one cert_gate_bitl_synth t25_k4_gate/k4/cert_gate_bitl_synth.sv treek4
puts "T25_SYNTH_DONE"
