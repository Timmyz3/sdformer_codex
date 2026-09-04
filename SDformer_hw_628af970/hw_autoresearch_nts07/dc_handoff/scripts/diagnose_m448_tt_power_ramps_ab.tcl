set design_name m405_q32_elastic_selected_slice
set tt_lib_db [file normalize $::env(TT_LIB_DB)]
set ss_lib_db [file normalize $::env(SS_LIB_DB)]
set mapped_netlist [file normalize $::env(MAPPED_NETLIST)]
set mapped_sdc [file normalize $::env(MAPPED_SDC)]
set gate_saif [file normalize $::env(GATE_SAIF_FILE)]
set output_dir [file normalize $::env(OUTPUT_DIR)]

file mkdir $output_dir
set_app_var search_path [list [file dirname $tt_lib_db] [file dirname $ss_lib_db]]
set_app_var link_path [list "*" $tt_lib_db $ss_lib_db]
read_verilog $mapped_netlist
current_design $design_name
link_design $design_name
read_sdc $mapped_sdc
set_operating_conditions tt0p9v25c \
    -library tcbn28hpcplusbwp35p140tt0p9v25c
set_wire_load_model -name ZeroWireload \
    -library tcbn28hpcplusbwp35p140tt0p9v25c

set data_inputs [remove_from_collection [all_inputs] [get_ports {clk_core reset_n}]]
if {$::env(APPLY_INPUT_TRANSITION) eq "true"} {
    set_input_transition $::env(INPUT_TRANSITION_NS) $data_inputs
}

set power_enable_analysis true
set_app_var power_analysis_mode averaged
read_saif -strip_path tb_m425_h67_balanced_selected_slice_direct_saif/dut/u_gate \
    $gate_saif
set_app_var timing_save_pin_arrival_and_slack true
update_timing -full
check_power -verbose -significant_digits 8 \
    > "$output_dir/check_power_verbose.rpt"
report_port -verbose $data_inputs > "$output_dir/data_input_ports.rpt"
report_timing -from $data_inputs -max_paths 20 -nworst 1 \
    -transition_time -capacitance -significant_digits 6 \
    > "$output_dir/input_timing_sample.rpt"

set fp [open "$output_dir/DIAGNOSTIC_COMPLETE.txt" w]
puts $fp "apply_input_transition=$::env(APPLY_INPUT_TRANSITION)"
puts $fp "input_transition_ns=$::env(INPUT_TRANSITION_NS)"
puts $fp "data_input_count=[sizeof_collection $data_inputs]"
puts $fp "update_power_called=false"
puts $fp "report_power_called=false"
close $fp
quit
