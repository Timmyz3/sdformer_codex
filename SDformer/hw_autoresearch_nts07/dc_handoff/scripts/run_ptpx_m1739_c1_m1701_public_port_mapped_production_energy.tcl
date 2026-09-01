set design_name m935_m912_three_stage_exact_parent_match_product_capture_island
set macro_cell TS1N28HPCPHVTB128X128M4S
set expected_macro_count 9
set std_tt_db [file normalize $::env(M1739_STD_TT_DB)]
set std_ss_db [file normalize $::env(M1739_STD_SS_DB)]
set macro_slow_db [file normalize $::env(M1739_MACRO_SLOW_DB)]
set mapped_netlist [file normalize $::env(M1739_MAPPED_NETLIST)]
set mapped_sdc [file normalize $::env(M1739_MAPPED_SDC)]
set gate_saif [file normalize $::env(M1739_GATE_SAIF)]
set output_dir [file normalize $::env(M1739_OUTPUT_DIR)]

file mkdir $output_dir
file mkdir "$output_dir/reports"
set_app_var search_path [list [file dirname $std_tt_db] \
    [file dirname $std_ss_db] [file dirname $macro_slow_db]]
set_app_var link_path [list "*" $std_tt_db $std_ss_db $macro_slow_db]

read_verilog $mapped_netlist
current_design $design_name
link_design $design_name
if {[sizeof_collection [get_cells -hierarchical -filter "is_black_box==true"]] != 0} {
    error "M1739_FAIL_BLACK_BOX_AFTER_LINK"
}
read_sdc $mapped_sdc
set_operating_conditions tt0p9v25c \
    -library tcbn28hpcplusbwp35p140tt0p9v25c
set_wire_load_model -name ZeroWireload \
    -library tcbn28hpcplusbwp35p140tt0p9v25c
set nonclock_inputs [remove_from_collection [all_inputs] [get_ports clk_core]]
set_input_transition 0.100 $nonclock_inputs

set macro_cells [get_cells -hierarchical -filter "ref_name == $macro_cell"]
set macro_count [sizeof_collection $macro_cells]
if {$macro_count != $expected_macro_count} {
    error "M1739_FAIL_MACRO_COUNT_$macro_count"
}

set power_enable_analysis true
set_app_var power_analysis_mode averaged
set_app_var timing_save_pin_arrival_and_slack true
proc m1739_read_text {path} {
    set fp [open $path r]
    set value [read $fp]
    close $fp
    return $value
}
redirect -file "$output_dir/reports/saif_annotation_summary.rpt" {
    read_saif -strip_path $::env(M1739_SAIF_INSTANCE) \
        -report_inconsistent_annotation \
        "$output_dir/reports/inconsistent_annotation.rpt" $gate_saif
}
redirect -file "$output_dir/reports/switching_coverage.rpt" {
    report_switching_activity -coverage -include_mapping_types
}
report_switching_activity -list_annotated -include_mapping_types \
    > "$output_dir/reports/switching_annotated.rpt"
report_switching_activity -list_not_annotated -include_mapping_types \
    > "$output_dir/reports/switching_unannotated.rpt"

set saif_text [m1739_read_text "$output_dir/reports/saif_annotation_summary.rpt"]
if {![regexp {Total number of nets = ([0-9]+)} $saif_text -> total_nets]} {
    error "M1739_FAIL_CANNOT_PARSE_TOTAL_NETS"
}
if {![regexp {Number of annotated nets = ([0-9]+) \(([0-9.]+)%\)} \
        $saif_text -> annotated_nets annotated_percent]} {
    error "M1739_FAIL_CANNOT_PARSE_ANNOTATED_NETS"
}
if {![regexp {Total number of leaf cells = ([0-9]+)} \
        $saif_text -> total_leaf_cells]} {
    error "M1739_FAIL_CANNOT_PARSE_TOTAL_LEAF_CELLS"
}
if {![regexp {Number of fully annotated leaf cells = ([0-9]+) \(([0-9.]+)%\)} \
        $saif_text -> annotated_leaf_cells annotated_leaf_percent]} {
    error "M1739_FAIL_CANNOT_PARSE_ANNOTATED_LEAF_CELLS"
}
if {$total_nets <= 0 || $annotated_nets != $total_nets \
        || $annotated_percent != 100.0} {
    error "M1739_FAIL_EXACT_NET_ANNOTATION_GATE"
}
if {$total_leaf_cells <= 0 || $annotated_leaf_cells != $total_leaf_cells \
        || $annotated_leaf_percent != 100.0} {
    error "M1739_FAIL_EXACT_LEAF_ANNOTATION_GATE"
}

check_timing -verbose > "$output_dir/reports/check_timing.rpt"
update_timing -full
check_power -verbose -significant_digits 8 \
    > "$output_dir/reports/check_power.rpt"
update_power
report_power -unit mW -nosplit -significant_digits 8 \
    > "$output_dir/reports/ptpx_top_including_macro_liberty.rpt"
report_power -hierarchy -area -unit mW -nosplit -significant_digits 8 \
    > "$output_dir/reports/ptpx_hierarchy_including_macro_liberty.rpt"
report_power $macro_cells -unit mW -nosplit -significant_digits 8 \
    > "$output_dir/reports/ptpx_nine_parent_macros_liberty_diagnostic.rpt"
report_timing -delay_type max -max_paths 20 -nworst 2 \
    > "$output_dir/reports/timing_setup_slow.rpt"
report_timing -delay_type min -max_paths 20 -nworst 2 \
    > "$output_dir/reports/timing_hold_fast_diagnostic.rpt"

set scope_fp [open "$output_dir/reports/scope_and_boundary.rpt" w]
puts $scope_fp "milestone=M1739"
puts $scope_fp "design=$design_name"
puts $scope_fp "analysis=averaged_prelayout_mapped_gate_activity"
puts $scope_fp "clock_period_ns=3.000"
puts $scope_fp "measurement_cycles=$::env(M1739_MEASUREMENT_CYCLES)"
puts $scope_fp "saif_duration_ns=$::env(M1739_SAIF_DURATION_NS)"
puts $scope_fp "saif_scope=$::env(M1739_SAIF_INSTANCE)"
puts $scope_fp "public_port_only_testbench=true"
puts $scope_fp "testbench_force_or_release=false"
puts $scope_fp "clock_network=ideal_no_cts"
puts $scope_fp "wireload=ZeroWireload"
puts $scope_fp "spef=false"
puts $scope_fp "macro_count=$macro_count"
puts $scope_fp "top_report_includes_macro_liberty=true"
puts $scope_fp "paper_logic_power_requires_top_minus_macro_rederivation=true"
puts $scope_fp "parent_sram_paper_energy_uses_external_access_model=true"
puts $scope_fp "macro_liberty_power_is_diagnostic_and_must_not_be_double_counted=true"
puts $scope_fp "not_total_c1=true"
puts $scope_fp "not_energy_per_frame=true"
close $scope_fp

set marker [open "$output_dir/PTPX_INTERNAL_COMPLETE.txt" w]
puts $marker "PASS_M1739_C1_M1701_PUBLIC_PORT_MAPPED_PTPX_TOOL_COMPLETE"
puts $marker "macro_count=$macro_count"
puts $marker "public_port_only=true"
puts $marker "exact_net_annotation=$annotated_nets/$total_nets=100.0%"
puts $marker "exact_leaf_annotation=$annotated_leaf_cells/$total_leaf_cells=100.0%"
puts $marker "claim_boundary=logic_only_plus_separate_nine_parent_sram_model_pending_checker"
close $marker
quit
