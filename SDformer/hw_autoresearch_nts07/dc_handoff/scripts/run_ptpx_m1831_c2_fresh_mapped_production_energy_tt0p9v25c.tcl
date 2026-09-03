# M1831 source draft: averaged, prelayout, standard-cell-only PTPX for one
# M1809-derived axis and one checked DUT-only SAIF coordinate.
set design_name $::env(M1831_DESIGN_NAME)
set tt_lib_db [file normalize $::env(M1831_TT_LIB_DB)]
set mapped_netlist [file normalize $::env(M1831_MAPPED_NETLIST)]
set mapped_sdc [file normalize $::env(M1831_MAPPED_SDC)]
set gate_saif [file normalize $::env(M1831_GATE_SAIF)]
set output_dir [file normalize $::env(M1831_OUTPUT_DIR)]

proc m1831_read_text {path} {
    set fp [open $path r]
    set value [read $fp]
    close $fp
    return $value
}

if {$design_name ne "m1809_c2_registered_fault_matched_k8_k1x8_raw4_acc24_ARCH_MODE0"
        && $design_name ne "m1809_c2_registered_fault_matched_k8_k1x8_raw4_acc24_ARCH_MODE1"} {
    error "M1831_FAIL_DERIVED_TOP"
}
file mkdir "$output_dir/reports"
set_app_var search_path [list [file dirname $tt_lib_db]]
set_app_var link_path [list "*" $tt_lib_db]
read_verilog $mapped_netlist
current_design $design_name
link_design $design_name
if {[sizeof_collection [get_cells -hierarchical -filter "is_black_box==true"]] != 0} {
    error "M1831_FAIL_BLACK_BOX_AFTER_LINK"
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

set scope_fp [open "$output_dir/reports/runtime_scope.rpt" w]
puts $scope_fp "milestone=M1831_SOURCE_DRAFT"
puts $scope_fp "design=$design_name"
puts $scope_fp "axis=$::env(M1831_AXIS)"
puts $scope_fp "case=$::env(M1831_CASE_ID)"
puts $scope_fp "analysis=averaged_prelayout_standard_cell_power"
puts $scope_fp "power_corner=tt0p9v25c"
puts $scope_fp "clock_period_ns=3.0"
puts $scope_fp "measurement_cycles=$::env(M1831_MEASUREMENT_CYCLES)"
puts $scope_fp "accepted_sources=$::env(M1831_ACCEPTED_SOURCES)"
puts $scope_fp "saif_scope=$::env(M1831_SAIF_INSTANCE)"
puts $scope_fp "saif_scope_is_exact_derived_mapped_top=true"
puts $scope_fp "clock_network=ideal_no_cts"
puts $scope_fp "wireload=ZeroWireload"
puts $scope_fp "spef=false"
puts $scope_fp "macro_count=0"
puts $scope_fp "external_weight_sram_excluded=true"
close $scope_fp

redirect -file "$output_dir/reports/saif_annotation_summary.rpt" {
    read_saif -strip_path $::env(M1831_SAIF_INSTANCE) \
        -report_inconsistent_annotation \
        "$output_dir/reports/inconsistent_annotation.rpt" $gate_saif
}
redirect -file "$output_dir/reports/switching_coverage.rpt" {
    report_switching_activity -coverage -include_mapping_types
}
report_switching_activity -list_not_annotated -include_mapping_types \
    > "$output_dir/reports/switching_unannotated.rpt"
report_switching_activity > "$output_dir/reports/switching_summary.rpt"

set saif_text [m1831_read_text "$output_dir/reports/saif_annotation_summary.rpt"]
if {![regexp {Total number of nets = ([0-9]+)} $saif_text -> total_nets]
        || ![regexp {Number of annotated nets = ([0-9]+) \(([0-9.]+)%\)} \
            $saif_text -> annotated_nets annotated_percent]
        || ![regexp {Total number of leaf cells = ([0-9]+)} \
            $saif_text -> total_leaf_cells]
        || ![regexp {Number of fully annotated leaf cells = ([0-9]+) \(([0-9.]+)%\)} \
            $saif_text -> annotated_leaf_cells annotated_leaf_percent]} {
    error "M1831_FAIL_ANNOTATION_PARSE"
}
if {$total_nets <= 0 || $annotated_nets != $total_nets
        || $annotated_percent != 100.0
        || $total_leaf_cells <= 0
        || $annotated_leaf_cells != $total_leaf_cells
        || $annotated_leaf_percent != 100.0} {
    error "M1831_FAIL_EXACT_ANNOTATION_GATE"
}

check_timing -verbose > "$output_dir/reports/check_timing.rpt"
update_timing -full
check_power -verbose -significant_digits 8 \
    > "$output_dir/reports/check_power.rpt"
set check_text [m1831_read_text "$output_dir/reports/check_power.rpt"]
if {![regexp {check_power succeeded\.} $check_text]} {
    error "M1831_FAIL_CHECK_POWER"
}
update_power
report_power -unit mW -nosplit -significant_digits 8 \
    > "$output_dir/reports/power.rpt"
report_power -hierarchy -area -unit mW -nosplit -significant_digits 8 \
    > "$output_dir/reports/power_hierarchy.rpt"
report_timing -delay_type max -max_paths 20 -nworst 2 \
    > "$output_dir/reports/timing_setup_tt.rpt"

set power_text [m1831_read_text "$output_dir/reports/power.rpt"]
foreach field {"Net Switching Power" "Cell Internal Power" \
        "Cell Leakage Power" "Total Power"} {
    set pattern [format {%s[[:space:]]*=[[:space:]]*([0-9.eE+-]+)} $field]
    if {[regexp -all $pattern $power_text] != 1} {
        error "M1831_FAIL_NONUNIQUE_POWER_FIELD_$field"
    }
}

set marker [open "$output_dir/PTPX_INTERNAL_COMPLETE.txt" w]
puts $marker "PASS_M1831_C2_FRESH_MAPPED_PRODUCTION_PTPX_PENDING_RESULT_HAMMER"
puts $marker "axis=$::env(M1831_AXIS)"
puts $marker "case=$::env(M1831_CASE_ID)"
puts $marker "exact_net_annotation=$annotated_nets/$total_nets=100.0%"
puts $marker "exact_leaf_annotation=$annotated_leaf_cells/$total_leaf_cells=100.0%"
puts $marker "macro_count=0"
close $marker
quit
