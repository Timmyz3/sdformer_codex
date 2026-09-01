set design_name $::env(DESIGN_NAME)
set tt_lib_db [file normalize $::env(TT_LIB_DB)]
set sdc_lib_db [file normalize $::env(SDC_LIB_DB)]
set mapped_netlist [file normalize $::env(MAPPED_NETLIST)]
set mapped_sdc [file normalize $::env(MAPPED_SDC)]
set gate_saif [file normalize $::env(GATE_SAIF_FILE)]
set output_dir [file normalize $::env(OUTPUT_DIR)]

proc m1684_read_text {path} {
    set fp [open $path r]
    set value [read $fp]
    close $fp
    return $value
}

file mkdir "$output_dir/reports"
set_app_var search_path [list [file dirname $tt_lib_db] [file dirname $sdc_lib_db]]
set_app_var link_path [list "*" $tt_lib_db $sdc_lib_db]
read_verilog $mapped_netlist
current_design $design_name
link_design $design_name
if {[sizeof_collection [get_cells -hierarchical -filter "is_black_box==true"]] != 0} {
    error "M1684_FAIL_BLACK_BOX_AFTER_LINK"
}
read_sdc $mapped_sdc
set_operating_conditions tt0p9v25c \
    -library tcbn28hpcplusbwp35p140tt0p9v25c
set_wire_load_model -name ZeroWireload \
    -library tcbn28hpcplusbwp35p140tt0p9v25c
set nonclock_inputs [remove_from_collection [all_inputs] [get_ports clk_core]]
set_input_transition 0.100 $nonclock_inputs
set power_enable_analysis true
set_app_var power_analysis_mode averaged
set_app_var timing_save_pin_arrival_and_slack true

set scope_fp [open "$output_dir/reports/ptpx_scope.rpt" w]
puts $scope_fp "milestone=M1684"
puts $scope_fp "design=$design_name"
puts $scope_fp "analysis=averaged_prelayout_standard_cell_power"
puts $scope_fp "power_corner=tt0p9v25c"
puts $scope_fp "voltage_v=0.9"
puts $scope_fp "temperature_c=25"
puts $scope_fp "clock_period_ns=3.0"
puts $scope_fp "saif_duration_ns=$::env(SAIF_DURATION_NS)"
puts $scope_fp "measurement_cycles=$::env(MEASUREMENT_CYCLES)"
puts $scope_fp "accepted_sources=$::env(ACCEPTED_SOURCES)"
puts $scope_fp "saif_scope=$::env(SAIF_INSTANCE)"
puts $scope_fp "saif_scope_is_gate_only=true"
puts $scope_fp "primary_input_transition_ns=0.100"
puts $scope_fp "clock_network=ideal_no_cts"
puts $scope_fp "wireload=ZeroWireload"
puts $scope_fp "spef=false"
puts $scope_fp "macro_count=0"
close $scope_fp

redirect -file "$output_dir/reports/saif_annotation_summary.rpt" {
    read_saif -strip_path $::env(SAIF_INSTANCE) \
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
report_switching_activity > "$output_dir/reports/switching_summary.rpt"

set saif_text [m1684_read_text "$output_dir/reports/saif_annotation_summary.rpt"]
if {![regexp {Total number of nets = ([0-9]+)} $saif_text -> total_nets]} {
    error "M1684_FAIL_CANNOT_PARSE_TOTAL_NETS"
}
if {![regexp {Number of annotated nets = ([0-9]+) \(([0-9.]+)%\)} \
        $saif_text -> annotated_nets annotated_percent]} {
    error "M1684_FAIL_CANNOT_PARSE_ANNOTATED_NETS"
}
if {![regexp {Total number of leaf cells = ([0-9]+)} \
        $saif_text -> total_leaf_cells]} {
    error "M1684_FAIL_CANNOT_PARSE_TOTAL_LEAF_CELLS"
}
if {![regexp {Number of fully annotated leaf cells = ([0-9]+) \(([0-9.]+)%\)} \
        $saif_text -> annotated_leaf_cells annotated_leaf_percent]} {
    error "M1684_FAIL_CANNOT_PARSE_ANNOTATED_LEAF_CELLS"
}
if {$total_nets <= 0 || $annotated_nets != $total_nets \
        || $annotated_percent != 100.0} {
    error "M1684_FAIL_EXACT_NET_ANNOTATION_GATE"
}
if {$total_leaf_cells <= 0 || $annotated_leaf_cells != $total_leaf_cells \
        || $annotated_leaf_percent != 100.0} {
    error "M1684_FAIL_EXACT_LEAF_ANNOTATION_GATE"
}

check_timing -verbose > "$output_dir/reports/ptpx_check_timing.rpt"
update_timing -full
check_power -verbose -significant_digits 8 \
    > "$output_dir/reports/ptpx_check_power_pre_update.rpt"
set check_text [m1684_read_text "$output_dir/reports/ptpx_check_power_pre_update.rpt"]
if {![regexp {check_power succeeded\.} $check_text]} {
    error "M1684_FAIL_CHECK_POWER"
}
update_power
report_power -unit mW -nosplit -significant_digits 8 \
    > "$output_dir/reports/ptpx_power.rpt"
report_power -hierarchy -area -unit mW -nosplit -significant_digits 8 \
    > "$output_dir/reports/ptpx_power_hierarchy.rpt"
report_power -verbose -unit mW -nosplit -significant_digits 8 \
    > "$output_dir/reports/ptpx_power_verbose.rpt"
report_timing -delay_type max -max_paths 20 -nworst 2 \
    > "$output_dir/reports/ptpx_timing_setup.rpt"
report_clock > "$output_dir/reports/ptpx_clock.rpt"

set power_text [m1684_read_text "$output_dir/reports/ptpx_power.rpt"]
foreach field {"Net Switching Power" "Cell Internal Power" \
        "Cell Leakage Power" "Total Power"} {
    set pattern [format {%s[[:space:]]*=[[:space:]]*([0-9.eE+-]+)} $field]
    if {[regexp -all $pattern $power_text] != 1} {
        error "M1684_FAIL_NONUNIQUE_POWER_FIELD_$field"
    }
}

set marker [open "$output_dir/PTPX_INTERNAL_COMPLETE.txt" w]
puts $marker "PASS_M1684_C2_M1609_FRESH_MAPPED_PRODUCTION_PTPX"
puts $marker "axis=$::env(AXIS)"
puts $marker "case=$::env(CASE_ID)"
puts $marker "exact_net_annotation=$annotated_nets/$total_nets=100.0%"
puts $marker "exact_leaf_annotation=$annotated_leaf_cells/$total_leaf_cells=100.0%"
puts $marker "fault_binary_clean=$::env(FAULT_BINARY_CLEAN)"
puts $marker "registered_fault_public_zero=$::env(REGISTERED_FAULT_PUBLIC_ZERO)"
close $marker
quit
