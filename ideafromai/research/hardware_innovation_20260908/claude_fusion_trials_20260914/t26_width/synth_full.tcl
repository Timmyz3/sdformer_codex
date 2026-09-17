set ROOT [file normalize [file join [file dirname [info script]] ..]]
cd $ROOT
set part xczu5ev-sfvc784-2-e
create_project -force t26_w38full ./t26_width/objsyn_w38full -part $part
read_verilog -sv t26_width/w38_full/cert_gate_bitl_synth.sv
read_xdc t24_fpga/clk4ns.xdc
synth_design -top cert_gate_bitl_synth -mode out_of_context -part $part
report_utilization -file t26_width/w38full_util.rpt
report_timing_summary -delay_type max -file t26_width/w38full_timing.rpt
close_project
puts "T26_FULL_DONE"
