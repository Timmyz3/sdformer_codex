# Matched averaged PTPX for one M2018 schedule axis.  The gate SAIF contains
# only the post-load execute window of the pre-registered ep34 G48 slot 0.
set design_name $::env(M2054_DESIGN_NAME)
set tt_lib_db [file normalize $::env(M2054_TT_LIB_DB)]
set mapped_netlist [file normalize $::env(M2054_MAPPED_NETLIST)]
set mapped_sdc [file normalize $::env(M2054_MAPPED_SDC)]
set gate_saif [file normalize $::env(M2054_GATE_SAIF)]
set output_dir [file normalize $::env(M2054_OUTPUT_DIR)]

proc m2054_read_text {path} {
    set fp [open $path r]
    set value [read $fp]
    close $fp
    return $value
}

if {$design_name ne "m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend_SCHEDULE_MODE0"
        && $design_name ne "m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend_SCHEDULE_MODE1"} {
    error "M2054_FAIL_DERIVED_TOP"
}
file mkdir "$output_dir/reports"
set_app_var search_path [list [file dirname $tt_lib_db]]
set_app_var link_path [list "*" $tt_lib_db]
read_verilog $mapped_netlist
current_design $design_name
link_design $design_name
if {[sizeof_collection [get_cells -hierarchical -filter "is_black_box==true"]] != 0} {
    error "M2054_FAIL_BLACK_BOX_AFTER_LINK"
}
if {[sizeof_collection [get_cells -hierarchical -filter "is_memory_cell==true"]] != 0} {
    error "M2054_FAIL_NONZERO_MACRO_COUNT"
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
    read_saif -strip_path $::env(M2054_SAIF_INSTANCE) \
        -report_inconsistent_annotation \
        "$output_dir/reports/inconsistent_annotation.rpt" $gate_saif
}
redirect -file "$output_dir/reports/switching_coverage.rpt" {
    report_switching_activity -coverage -include_mapping_types
}
report_switching_activity -list_not_annotated -include_mapping_types \
    > "$output_dir/reports/switching_unannotated.rpt"
report_switching_activity > "$output_dir/reports/switching_summary.rpt"

set saif_text [m2054_read_text "$output_dir/reports/saif_annotation_summary.rpt"]
if {![regexp {Total number of nets = ([0-9]+)} $saif_text -> total_nets]
        || ![regexp {Number of annotated nets = ([0-9]+) \(([0-9.]+)%\)} \
            $saif_text -> annotated_nets annotated_percent]
        || ![regexp {Total number of leaf cells = ([0-9]+)} \
            $saif_text -> total_leaf_cells]
        || ![regexp {Number of fully annotated leaf cells = ([0-9]+) \(([0-9.]+)%\)} \
            $saif_text -> annotated_leaf_cells annotated_leaf_percent]} {
    error "M2054_FAIL_ANNOTATION_PARSE"
}
if {$total_nets <= 0 || $annotated_nets != $total_nets
        || $annotated_percent != 100.0 || $total_leaf_cells <= 0
        || $annotated_leaf_cells != $total_leaf_cells
        || $annotated_leaf_percent != 100.0} {
    error "M2054_FAIL_EXACT_ANNOTATION_GATE"
}

check_timing -verbose > "$output_dir/reports/check_timing.rpt"
update_timing -full
check_power -verbose -significant_digits 8 \
    > "$output_dir/reports/check_power.rpt"
set check_text [m2054_read_text "$output_dir/reports/check_power.rpt"]
if {![regexp {check_power succeeded\.} $check_text]} {
    error "M2054_FAIL_CHECK_POWER"
}
update_power
report_power -unit mW -nosplit -significant_digits 8 \
    > "$output_dir/reports/power.rpt"
report_power -hierarchy -area -unit mW -nosplit -significant_digits 8 \
    > "$output_dir/reports/power_hierarchy.rpt"
report_timing -delay_type max -max_paths 20 -nworst 2 \
    > "$output_dir/reports/timing_setup_tt.rpt"

set power_text [m2054_read_text "$output_dir/reports/power.rpt"]
foreach field {"Net Switching Power" "Cell Internal Power" \
        "Cell Leakage Power" "Total Power"} {
    set pattern [format {%s[[:space:]]*=[[:space:]]*([0-9.eE+-]+)} $field]
    if {[regexp -all $pattern $power_text] != 1} {
        error "M2054_FAIL_NONUNIQUE_POWER_FIELD_$field"
    }
}

set scope_fp [open "$output_dir/reports/scope_and_boundary.rpt" w]
puts $scope_fp "milestone=M2054"
puts $scope_fp "design=$design_name"
puts $scope_fp "axis=$::env(M2054_AXIS)"
puts $scope_fp "analysis=averaged_prelayout_standard_cell_power"
puts $scope_fp "power_corner=tt0p9v25c"
puts $scope_fp "clock_period_ns=3.0"
puts $scope_fp "measurement_cycles=$::env(M2054_MEASUREMENT_CYCLES)"
puts $scope_fp "descriptor_preload_cycles_excluded=383"
puts $scope_fp "workload=ep34_full40_fixture_slot0_layer28_tokens0to3"
puts $scope_fp "saif_scope=$::env(M2054_SAIF_INSTANCE)"
puts $scope_fp "clock_network=ideal_no_cts"
puts $scope_fp "wireload=ZeroWireload"
puts $scope_fp "spef=false"
puts $scope_fp "macro_count=0"
puts $scope_fp "external_weight_sram_excluded=true"
close $scope_fp

set marker [open "$output_dir/PTPX_INTERNAL_COMPLETE.txt" w]
puts $marker "PASS_M2054_M2018_TSBG_MATCHED_MAPPED_PTPX_PENDING_RESULT_HAMMER"
puts $marker "axis=$::env(M2054_AXIS)"
puts $marker "exact_net_annotation=$annotated_nets/$total_nets=100.0%"
puts $marker "exact_leaf_annotation=$annotated_leaf_cells/$total_leaf_cells=100.0%"
puts $marker "macro_count=0"
close $marker
quit
