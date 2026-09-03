# UNSEALED M2061 DRAFT.  No execution is authorized before the M2058 failure
# hammer and a future M2061 source contract/review.  Axis alone derives the
# design, settled-negedge SAIF scope, and fixed measurement denominator.  The
# predecessor cell model has no UNIT_DELAY behavior and no SDF is annotated;
# this remains mapped zero-delay functional activity, not a delay repair.
set axis $::env(M2061_AXIS)
set tt_lib_db [file normalize $::env(M2061_TT_LIB_DB)]
set ssg_lib_db [file normalize $::env(M2061_SSG_LIB_DB)]
set mapped_netlist [file normalize $::env(M2061_MAPPED_NETLIST)]
set mapped_sdc [file normalize $::env(M2061_MAPPED_SDC)]
set gate_saif [file normalize $::env(M2061_GATE_SAIF)]
set output_dir [file normalize $::env(M2061_OUTPUT_DIR)]

if {$axis eq "ordinary_lru4"} {
    set design_name m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend_SCHEDULE_MODE0
    set saif_scope tb_m2061_m2018_tsbg_matched_mapped_energy.core.dut_base.g_mapped.mapped_implementation
    set measurement_cycles 20292
} elseif {$axis eq "tsbg_b4"} {
    set design_name m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend_SCHEDULE_MODE1
    set saif_scope tb_m2061_m2018_tsbg_matched_mapped_energy.core.dut_tsbg.g_mapped.mapped_implementation
    set measurement_cycles 7569
} else {
    error "M2061_FAIL_AXIS"
}

proc m2061_read_text {path} {
    set fp [open $path r]
    set value [read $fp]
    close $fp
    return $value
}

proc m2061_read_saif_header {path} {
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

foreach path [list $tt_lib_db $ssg_lib_db $mapped_netlist $mapped_sdc $gate_saif] {
    if {![file isfile $path]} { error "M2061_FAIL_MISSING_INPUT_$path" }
}
file mkdir "$output_dir/reports"
set_app_var search_path [list [file dirname $tt_lib_db] [file dirname $ssg_lib_db]]
read_db $ssg_lib_db
read_db $tt_lib_db
set_app_var link_path [list "*" $tt_lib_db $ssg_lib_db]
read_verilog $mapped_netlist
current_design $design_name
link_design $design_name
if {[sizeof_collection [get_cells -hierarchical -filter "is_black_box==true"]] != 0} {
    error "M2061_FAIL_BLACK_BOX_AFTER_LINK"
}
if {[sizeof_collection [get_cells -hierarchical -filter "is_memory_cell==true"]] != 0} {
    error "M2061_FAIL_NONZERO_MACRO_COUNT"
}
read_sdc $mapped_sdc
set_operating_conditions tt0p9v25c -library tcbn28hpcplusbwp35p140tt0p9v25c
set_wire_load_model -name ZeroWireload -library tcbn28hpcplusbwp35p140tt0p9v25c
set nonclock_inputs [remove_from_collection [all_inputs] [get_ports clk_core]]
set_input_transition 0.100 $nonclock_inputs
set power_enable_analysis true
set_app_var power_analysis_mode averaged
set_app_var timing_save_pin_arrival_and_slack true

set header [m2061_read_saif_header $gate_saif]
if {![regexp {\(TIMESCALE[[:space:]]+([0-9.eE+-]+)[[:space:]]+([a-zA-Z]+)\)} \
        $header -> scale_value scale_unit]
        || ![regexp {\(DURATION[[:space:]]+([0-9.eE+-]+)\)} \
            $header -> duration_raw]} {
    error "M2061_FAIL_SAIF_HEADER_PARSE"
}
array set ns_per_unit {s 1.0e9 ms 1.0e6 us 1.0e3 ns 1.0 ps 1.0e-3 fs 1.0e-6}
if {![info exists ns_per_unit($scale_unit)]} { error "M2061_FAIL_SAIF_UNIT" }
set duration_ns [expr {double($duration_raw) * double($scale_value) \
    * $ns_per_unit($scale_unit)}]
set expected_duration_ns [expr {double($measurement_cycles) * 3.0}]
if {[expr {abs($duration_ns - $expected_duration_ns)}] > 1.0e-6} {
    error "M2061_FAIL_SAIF_DURATION"
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

set annotation [m2061_read_text "$output_dir/reports/saif_annotation_summary.rpt"]
if {![regexp {Total number of nets = ([0-9]+)} $annotation -> total_nets]
        || ![regexp {Number of annotated nets = ([0-9]+) \(([0-9.]+)%\)} \
            $annotation -> annotated_nets net_percent]
        || ![regexp {Total number of leaf cells = ([0-9]+)} \
            $annotation -> total_leaf]
        || ![regexp {Number of fully annotated leaf cells = ([0-9]+) \(([0-9.]+)%\)} \
            $annotation -> annotated_leaf leaf_percent]} {
    error "M2061_FAIL_ANNOTATION_PARSE"
}
if {$total_nets <= 0 || $annotated_nets != $total_nets
        || $net_percent != 100.0 || $total_leaf <= 0
        || $annotated_leaf != $total_leaf || $leaf_percent != 100.0} {
    error "M2061_FAIL_EXACT_ANNOTATION"
}

check_timing -verbose > "$output_dir/reports/check_timing.rpt"
update_timing -full
check_power -verbose -significant_digits 8 > "$output_dir/reports/check_power.rpt"
if {![regexp {check_power succeeded\.} \
        [m2061_read_text "$output_dir/reports/check_power.rpt"]]} {
    error "M2061_FAIL_CHECK_POWER"
}
update_power
report_power -unit mW -nosplit -significant_digits 8 > "$output_dir/reports/power.rpt"
report_power -hierarchy -area -unit mW -nosplit -significant_digits 8 \
    > "$output_dir/reports/power_hierarchy.rpt"
report_timing -delay_type max -max_paths 20 -nworst 2 \
    > "$output_dir/reports/timing_setup_tt.rpt"
set power_text [m2061_read_text "$output_dir/reports/power.rpt"]
foreach field {"Net Switching Power" "Cell Internal Power" \
        "Cell Leakage Power" "Total Power"} {
    set pattern [format {%s[[:space:]]*=[[:space:]]*([0-9.eE+-]+)} $field]
    if {[regexp -all $pattern $power_text] != 1} {
        error "M2061_FAIL_NONUNIQUE_POWER_FIELD_$field"
    }
}

set fp [open "$output_dir/reports/scope_and_boundary.rpt" w]
puts $fp "milestone=M2061"
puts $fp "axis=$axis"
puts $fp "design=$design_name"
puts $fp "sampling=settled_negedge_valid_gated_sideband_checker"
puts $fp "mapped_simulation=zero_delay_functional_no_SDF"
puts $fp "unit_delay_fix_claimed=false"
puts $fp "window_alignment=first_settled_execute_negedge_to_settled_completion_negedge"
puts $fp "first_half_cycle_transition_excluded=true"
puts $fp "analysis=averaged_prelayout_standard_cell_power"
puts $fp "power_corner=tt0p9v25c"
puts $fp "clock_period_ns=3.0"
puts $fp "measurement_cycles=$measurement_cycles"
puts $fp "measurement_duration_ns=$duration_ns"
puts $fp "descriptor_preload_cycles_excluded=383"
puts $fp "workload=ep34_full40_global_slot42_sample0_layer28_fc1_token0_g48"
puts $fp "saif_scope=$saif_scope"
puts $fp "clock_network=ideal_no_cts"
puts $fp "wireload=ZeroWireload"
puts $fp "spef=false"
puts $fp "macro_count=0"
puts $fp "external_weight_sram_excluded=true"
close $fp
set marker [open "$output_dir/PTPX_INTERNAL_COMPLETE.txt" w]
puts $marker "PASS_M2061_M2018_TSBG_SETTLED_MAPPED_PTPX_PENDING_RESULT_HAMMER"
puts $marker "axis=$axis"
puts $marker "measurement_cycles=$measurement_cycles"
puts $marker "exact_net_annotation=$annotated_nets/$total_nets=100.0%"
puts $marker "exact_leaf_annotation=$annotated_leaf/$total_leaf=100.0%"
close $marker
quit
