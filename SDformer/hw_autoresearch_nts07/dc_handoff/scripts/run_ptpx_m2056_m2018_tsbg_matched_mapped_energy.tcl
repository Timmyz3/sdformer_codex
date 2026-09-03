# M2056 matched averaged PTPX for one pre-registered mapped schedule axis.
# Axis is the only semantic selector.  Design, strip scope, and measurement
# cycles are derived here and cannot be supplied independently by a runner.
set axis $::env(M2056_AXIS)
set tt_lib_db [file normalize $::env(M2056_TT_LIB_DB)]
set ssg_lib_db [file normalize $::env(M2056_SSG_LIB_DB)]
set mapped_netlist [file normalize $::env(M2056_MAPPED_NETLIST)]
set mapped_sdc [file normalize $::env(M2056_MAPPED_SDC)]
set gate_saif [file normalize $::env(M2056_GATE_SAIF)]
set output_dir [file normalize $::env(M2056_OUTPUT_DIR)]

if {$axis eq "ordinary_lru4"} {
    set design_name m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend_SCHEDULE_MODE0
    set saif_scope tb_m2056_m2018_tsbg_matched_mapped_energy.core.dut_base.g_mapped.mapped_implementation
    set measurement_cycles 20292
} elseif {$axis eq "tsbg_b4"} {
    set design_name m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend_SCHEDULE_MODE1
    set saif_scope tb_m2056_m2018_tsbg_matched_mapped_energy.core.dut_tsbg.g_mapped.mapped_implementation
    set measurement_cycles 7569
} else {
    error "M2056_FAIL_AXIS_MUST_BE_ordinary_lru4_OR_tsbg_b4"
}

proc m2056_read_text {path} {
    set fp [open $path r]
    set value [read $fp]
    close $fp
    return $value
}

proc m2056_read_saif_header {path} {
    set fp [open $path r]
    set text ""
    set lines 0
    while {[gets $fp line] >= 0 && $lines < 256} {
        append text $line "\n"
        incr lines
        if {[regexp {\(DURATION[[:space:]]+[^\)]+\)} $text]
                && [regexp {\(TIMESCALE[[:space:]]+[^\)]+\)} $text]} {
            break
        }
    }
    close $fp
    return $text
}

foreach required_file [list $tt_lib_db $ssg_lib_db $mapped_netlist \
        $mapped_sdc $gate_saif] {
    if {![file isfile $required_file]} {
        error "M2056_FAIL_MISSING_INPUT_$required_file"
    }
}
file mkdir "$output_dir/reports"

# M2029's mapped SDC names the SSG library before this power run overrides the
# analysis point to TT.  Load both exact libraries before read_sdc so that the
# SDC cannot pass only because of ambient session state.
set_app_var search_path [list [file dirname $tt_lib_db] \
    [file dirname $ssg_lib_db]]
read_db $ssg_lib_db
read_db $tt_lib_db
set_app_var link_path [list "*" $tt_lib_db $ssg_lib_db]
read_verilog $mapped_netlist
current_design $design_name
link_design $design_name
if {[sizeof_collection [get_cells -hierarchical \
        -filter "is_black_box==true"]] != 0} {
    error "M2056_FAIL_BLACK_BOX_AFTER_LINK"
}
if {[sizeof_collection [get_cells -hierarchical \
        -filter "is_memory_cell==true"]] != 0} {
    error "M2056_FAIL_NONZERO_MACRO_COUNT"
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

# Reconcile the gate-SAIF time window with the frozen cycle denominator before
# it can affect power.  VCS may encode the same time using ns, ps, or fs.
set saif_header [m2056_read_saif_header $gate_saif]
if {![regexp {\(TIMESCALE[[:space:]]+([0-9.eE+-]+)[[:space:]]+([a-zA-Z]+)\)} \
        $saif_header -> saif_timescale_value saif_timescale_unit]
        || ![regexp {\(DURATION[[:space:]]+([0-9.eE+-]+)\)} \
            $saif_header -> saif_duration]} {
    error "M2056_FAIL_SAIF_HEADER_PARSE"
}
array set ns_per_unit {s 1.0e9 ms 1.0e6 us 1.0e3 ns 1.0 ps 1.0e-3 fs 1.0e-6}
if {![info exists ns_per_unit($saif_timescale_unit)]} {
    error "M2056_FAIL_UNSUPPORTED_SAIF_TIMESCALE_$saif_timescale_unit"
}
set saif_duration_ns [expr {double($saif_duration) \
    * double($saif_timescale_value) * $ns_per_unit($saif_timescale_unit)}]
set expected_duration_ns [expr {double($measurement_cycles) * 3.0}]
if {[expr {abs($saif_duration_ns - $expected_duration_ns)}] > 1.0e-6} {
    error "M2056_FAIL_SAIF_DURATION_EXPECTED_${expected_duration_ns}ns_GOT_${saif_duration_ns}ns"
}

redirect -file "$output_dir/reports/saif_annotation_summary.rpt" {
    read_saif -strip_path $saif_scope -report_inconsistent_annotation \
        "$output_dir/reports/inconsistent_annotation.rpt" $gate_saif
}
redirect -file "$output_dir/reports/switching_coverage.rpt" {
    report_switching_activity -coverage -include_mapping_types
}
report_switching_activity -list_not_annotated -include_mapping_types \
    > "$output_dir/reports/switching_unannotated.rpt"
report_switching_activity > "$output_dir/reports/switching_summary.rpt"

set annotation_text [m2056_read_text \
    "$output_dir/reports/saif_annotation_summary.rpt"]
if {![regexp {Total number of nets = ([0-9]+)} \
        $annotation_text -> total_nets]
        || ![regexp {Number of annotated nets = ([0-9]+) \(([0-9.]+)%\)} \
            $annotation_text -> annotated_nets annotated_percent]
        || ![regexp {Total number of leaf cells = ([0-9]+)} \
            $annotation_text -> total_leaf_cells]
        || ![regexp {Number of fully annotated leaf cells = ([0-9]+) \(([0-9.]+)%\)} \
            $annotation_text -> annotated_leaf_cells annotated_leaf_percent]} {
    error "M2056_FAIL_ANNOTATION_PARSE"
}
if {$total_nets <= 0 || $annotated_nets != $total_nets
        || $annotated_percent != 100.0 || $total_leaf_cells <= 0
        || $annotated_leaf_cells != $total_leaf_cells
        || $annotated_leaf_percent != 100.0} {
    error "M2056_FAIL_EXACT_ANNOTATION_GATE"
}

check_timing -verbose > "$output_dir/reports/check_timing.rpt"
update_timing -full
check_power -verbose -significant_digits 8 \
    > "$output_dir/reports/check_power.rpt"
set check_text [m2056_read_text "$output_dir/reports/check_power.rpt"]
if {![regexp {check_power succeeded\.} $check_text]} {
    error "M2056_FAIL_CHECK_POWER"
}
update_power
report_power -unit mW -nosplit -significant_digits 8 \
    > "$output_dir/reports/power.rpt"
report_power -hierarchy -area -unit mW -nosplit -significant_digits 8 \
    > "$output_dir/reports/power_hierarchy.rpt"
report_timing -delay_type max -max_paths 20 -nworst 2 \
    > "$output_dir/reports/timing_setup_tt.rpt"

set power_text [m2056_read_text "$output_dir/reports/power.rpt"]
foreach field {"Net Switching Power" "Cell Internal Power" \
        "Cell Leakage Power" "Total Power"} {
    set pattern [format {%s[[:space:]]*=[[:space:]]*([0-9.eE+-]+)} $field]
    if {[regexp -all $pattern $power_text] != 1} {
        error "M2056_FAIL_NONUNIQUE_POWER_FIELD_$field"
    }
}

set scope_fp [open "$output_dir/reports/scope_and_boundary.rpt" w]
puts $scope_fp "milestone=M2056"
puts $scope_fp "design=$design_name"
puts $scope_fp "axis=$axis"
puts $scope_fp "analysis=averaged_prelayout_standard_cell_power"
puts $scope_fp "power_corner=tt0p9v25c"
puts $scope_fp "clock_period_ns=3.0"
puts $scope_fp "measurement_cycles=$measurement_cycles"
puts $scope_fp "measurement_duration_ns=$saif_duration_ns"
puts $scope_fp "saif_timescale=$saif_timescale_value $saif_timescale_unit"
puts $scope_fp "saif_duration_raw=$saif_duration"
puts $scope_fp "descriptor_preload_cycles_excluded=383"
puts $scope_fp "workload=ep34_full40_global_slot42_sample0_layer28_fc1_token0_g48"
puts $scope_fp "m2047_semantic_anchor_slot=0"
puts $scope_fp "saif_scope=$saif_scope"
puts $scope_fp "clock_network=ideal_no_cts"
puts $scope_fp "wireload=ZeroWireload"
puts $scope_fp "spef=false"
puts $scope_fp "macro_count=0"
puts $scope_fp "external_weight_sram_excluded=true"
close $scope_fp

set marker [open "$output_dir/PTPX_INTERNAL_COMPLETE.txt" w]
puts $marker "PASS_M2056_M2018_TSBG_MATCHED_MAPPED_PTPX_PENDING_RESULT_HAMMER"
puts $marker "axis=$axis"
puts $marker "measurement_cycles=$measurement_cycles"
puts $marker "measurement_duration_ns=$saif_duration_ns"
puts $marker "exact_net_annotation=$annotated_nets/$total_nets=100.0%"
puts $marker "exact_leaf_annotation=$annotated_leaf_cells/$total_leaf_cells=100.0%"
puts $marker "macro_count=0"
close $marker
quit
