set design_name m518_matched_fixed_t10_atlif
set tt_lib_db [file normalize $::env(M1790_TT_LIB_DB)]
set mapped_netlist [file normalize $::env(M1790_MAPPED_NETLIST)]
set mapped_sdc [file normalize $::env(M1790_MAPPED_SDC)]
set gate_saif [file normalize $::env(M1790_GATE_SAIF)]
set output_dir [file normalize $::env(M1790_OUTPUT_DIR)]

proc m1790_read_text {path} {
    set fp [open $path r]
    set value [read $fp]
    close $fp
    return $value
}

file mkdir "$output_dir/reports"
set_app_var search_path [list [file dirname $tt_lib_db]]
set_app_var link_path [list "*" $tt_lib_db]
read_verilog $mapped_netlist
current_design $design_name
link_design $design_name
set black_boxes [get_cells -hierarchical -filter "is_black_box==true"]
set macro_cells [get_cells -hierarchical -filter "is_memory_cell==true"]
if {[sizeof_collection $black_boxes] != 0} {
    error "M1790_FAIL_BLACK_BOX_AFTER_LINK"
}
if {[sizeof_collection $macro_cells] != 0} {
    error "M1790_FAIL_NONZERO_MACRO_COUNT"
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

redirect -file "$output_dir/reports/saif_annotation_summary.rpt" {
    read_saif -strip_path $::env(M1790_SAIF_INSTANCE) \
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

set saif_text [m1790_read_text "$output_dir/reports/saif_annotation_summary.rpt"]
if {![regexp {Total number of nets = ([0-9]+)} $saif_text -> total_nets]} {
    error "M1790_FAIL_CANNOT_PARSE_TOTAL_NETS"
}
if {![regexp {Number of annotated nets = ([0-9]+) \(([0-9.]+)%\)} \
        $saif_text -> annotated_nets annotated_percent]} {
    error "M1790_FAIL_CANNOT_PARSE_ANNOTATED_NETS"
}
if {![regexp {Total number of leaf cells = ([0-9]+)} \
        $saif_text -> total_leaf_cells]} {
    error "M1790_FAIL_CANNOT_PARSE_TOTAL_LEAF_CELLS"
}
if {![regexp {Number of fully annotated leaf cells = ([0-9]+) \(([0-9.]+)%\)} \
        $saif_text -> annotated_leaf_cells annotated_leaf_percent]} {
    error "M1790_FAIL_CANNOT_PARSE_ANNOTATED_LEAF_CELLS"
}
if {$total_nets <= 0 || $annotated_nets != $total_nets \
        || $annotated_percent != 100.0} {
    error "M1790_FAIL_EXACT_NET_ANNOTATION_GATE"
}
if {$total_leaf_cells <= 0 || $annotated_leaf_cells != $total_leaf_cells \
        || $annotated_leaf_percent != 100.0} {
    error "M1790_FAIL_EXACT_LEAF_ANNOTATION_GATE"
}

check_timing -verbose > "$output_dir/reports/check_timing.rpt"
update_timing -full
check_power -verbose -significant_digits 8 \
    > "$output_dir/reports/check_power.rpt"
set check_text [m1790_read_text "$output_dir/reports/check_power.rpt"]
if {![regexp {check_power succeeded\.} $check_text]} {
    error "M1790_FAIL_CHECK_POWER"
}
update_power
report_power -unit mW -nosplit -significant_digits 8 \
    > "$output_dir/reports/ptpx_whole_mapped_c3_logic.rpt"
report_power -hierarchy -area -unit mW -nosplit -significant_digits 8 \
    > "$output_dir/reports/ptpx_hierarchy_diagnostic.rpt"
report_power -verbose -unit mW -nosplit -significant_digits 8 \
    > "$output_dir/reports/ptpx_verbose.rpt"

set power_text [m1790_read_text "$output_dir/reports/ptpx_whole_mapped_c3_logic.rpt"]
foreach field {"Net Switching Power" "Cell Internal Power" \
        "Cell Leakage Power" "Total Power"} {
    set pattern [format {%s[[:space:]]*=[[:space:]]*([0-9.eE+-]+)} $field]
    if {[regexp -all $pattern $power_text] != 1} {
        error "M1790_FAIL_NONUNIQUE_POWER_FIELD_$field"
    }
}

set scope_fp [open "$output_dir/reports/scope_and_boundary.rpt" w]
puts $scope_fp "milestone=M1790"
puts $scope_fp "design=$design_name"
puts $scope_fp "analysis=averaged_prelayout_mapped_gate_activity"
puts $scope_fp "power_corner=TT_0p9V_25C"
puts $scope_fp "clock_period_ns=3.000"
puts $scope_fp "measurement_cycles=$::env(M1790_MEASUREMENT_CYCLES)"
puts $scope_fp "saif_duration_ns=$::env(M1790_SAIF_DURATION_NS)"
puts $scope_fp "saif_scope=$::env(M1790_SAIF_INSTANCE)"
puts $scope_fp "public_port_only_testbench=true"
puts $scope_fp "hierarchical_drive_or_read=false"
puts $scope_fp "clock_network=ideal_no_cts"
puts $scope_fp "wireload=ZeroWireload"
puts $scope_fp "spef=false"
puts $scope_fp "macro_count=0"
puts $scope_fp "component_only=true"
puts $scope_fp "not_speedup=true"
puts $scope_fp "not_system_or_frame_energy=true"
puts $scope_fp "not_silicon_or_signoff=true"
close $scope_fp

set marker [open "$output_dir/PTPX_INTERNAL_COMPLETE.txt" w]
puts $marker "PASS_M1790_C3_M1454_FIXED_T10_MAPPED_COMPONENT_PTPX_TOOL_COMPLETE"
puts $marker "macro_count=0"
puts $marker "exact_net_annotation=$annotated_nets/$total_nets=100.0%"
puts $marker "exact_leaf_annotation=$annotated_leaf_cells/$total_leaf_cells=100.0%"
puts $marker "claim_boundary=prelayout_logic_only_component_energy"
close $marker
quit
