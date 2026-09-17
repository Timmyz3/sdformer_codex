if {[catch {
 set here $::env(OP_TIMING_DIR)
 set arm $::env(OP_TIMING_ARM)
 set out "$here/dc_$arm"
 file mkdir "$out/reports"
 file mkdir "$out/netlist"
 define_design_lib WORK -path "$out/WORK"
 set slow $::env(OP_TIMING_LIB)
 set_app_var target_library [list $slow]
 set_app_var link_library [list * $slow]
 set_app_var search_path [list $here [file dirname $slow]]
 set_app_var verilogout_no_tri true
 analyze -format sverilog [list "$here/operator_slice.sv"]
 elaborate operator_slice -parameters "NORMALIZED=>$arm"
 link
 uniquify
 set_operating_conditions ssg0p9v125c
 create_clock -name core_clk -period 3.000 -waveform {0 1.5} [get_ports clk_core]
 set_clock_uncertainty -setup 0.200 [get_clocks core_clk]
 set_clock_uncertainty -hold 0.050 [get_clocks core_clk]
 set data_inputs [remove_from_collection [all_inputs] [get_ports clk_core]]
 set_input_delay 0.250 -clock core_clk $data_inputs
 set_input_transition 0.100 $data_inputs
 set_output_delay 0.250 -clock core_clk [all_outputs]
 set_load 0.010 [all_outputs]
 set_max_fanout 32 [current_design]
 set_fix_multiple_port_nets -all -buffer_constants
 set_wire_load_model -name ZeroWireload [current_design]
 ungroup -all -flatten
 compile_ultra
 update_timing
 redirect "$out/reports/qor.rpt" {report_qor}
 redirect "$out/reports/area.rpt" {report_area -hierarchy}
 redirect "$out/reports/timing_all.rpt" {report_timing -delay_type max -max_paths 20 -significant_digits 5 -input_pins -nets}
 redirect "$out/reports/check_design.rpt" {check_design}
 redirect "$out/reports/constraints.rpt" {report_constraint -all_violators}
 redirect "$out/reports/resources.rpt" {report_resources}
 set decisions [get_pins -hierarchical {*gate_out_reg*/D *locked_out_reg*/D *lower_hit_out_reg*/D *upper_hit_out_reg*/D}]
 if {[sizeof_collection $decisions]>0} {
   redirect "$out/reports/timing_decision.rpt" {report_timing -to $decisions -delay_type max -max_paths 10 -significant_digits 5 -input_pins -nets}
   foreach stem {prefix_r lut_lo lut_hi tau_r p_r n_r original normalized} {
    set starts [get_cells -hierarchical -filter "full_name =~ *${stem}* && is_sequential == true"]
    if {[sizeof_collection $starts]>0} {
     redirect "$out/reports/timing_${stem}_to_decision.rpt" {report_timing -from $starts -to $decisions -delay_type max -max_paths 3 -significant_digits 5 -input_pins -nets}
    }
   }
 }
 change_names -rules verilog -hierarchy
 write_file -format verilog -hierarchy -output "$out/netlist/operator_mapped.v"
 write_file -format ddc -hierarchy -output "$out/netlist/operator.ddc"
 write_sdc "$out/netlist/operator.sdc"
 set fp [open "$out/reports/identity.rpt" w]
 puts $fp "design=[get_object_name [current_design]]\nNORMALIZED=$arm\nclock_ns=3\nsetup_uncertainty_ns=0.2\nin_out_delay_ns=0.25\ninput_transition_ns=0.1\noutput_load=0.01\nwireload=ZeroWireload\ncorner=ssg0p9v125c\ncompiled_once=true"
 close $fp
} error]} {
 puts stderr "OP_TIMING_FAILED: $error"
 exit 1
}
exit

