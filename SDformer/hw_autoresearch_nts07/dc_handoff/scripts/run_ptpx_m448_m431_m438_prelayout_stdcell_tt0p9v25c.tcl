set design_name $::env(DESIGN_NAME)
set tt_lib_db [file normalize $::env(TT_LIB_DB)]
set sdc_lib_db [file normalize $::env(SDC_LIB_DB)]
set mapped_netlist [file normalize $::env(MAPPED_NETLIST)]
set mapped_sdc [file normalize $::env(MAPPED_SDC)]
set gate_saif [file normalize $::env(GATE_SAIF_FILE)]
set output_dir [file normalize $::env(OUTPUT_DIR)]

proc m448_read_text {path} {
    set fp [open $path r]
    set value [read $fp]
    close $fp
    return $value
}

proc m448_write_text {path value} {
    set fp [open $path w]
    puts -nonewline $fp $value
    close $fp
}

file mkdir "$output_dir/reports"
set_app_var search_path [list [file dirname $tt_lib_db] [file dirname $sdc_lib_db]]
# TT is deliberately first so the mapped references bind to the power library.
# The SS library is also loaded because the immutable mapped SDC names it before
# this run explicitly overrides the operating condition to TT.
set_app_var link_path [list "*" $tt_lib_db $sdc_lib_db]

read_verilog $mapped_netlist
current_design $design_name
link_design $design_name

# Read the frozen M431 SDC exactly, then select the declared typical-power
# point.  The clock/load constraints stay frozen; only the library corner is
# changed from the SDC's SS timing point to TT 0.9 V, 25 C.
read_sdc $mapped_sdc
set_operating_conditions $::env(POWER_OPERATING_CONDITION) \
    -library $::env(POWER_LIBRARY_NAME)
set_wire_load_model -name ZeroWireload -library $::env(POWER_LIBRARY_NAME)
set power_enable_analysis true
set_app_var power_analysis_mode averaged

set scope_text ""
append scope_text "milestone=M448\n"
append scope_text "design=$design_name\n"
append scope_text "analysis=averaged_prelayout_standard_cell_power\n"
append scope_text "power_corner=$::env(POWER_OPERATING_CONDITION)\n"
append scope_text "power_library=$::env(POWER_LIBRARY_NAME)\n"
append scope_text "voltage_v=0.9\n"
append scope_text "temperature_c=25\n"
append scope_text "clock_period_ns=3.0\n"
append scope_text "clock_frequency_mhz=333.333333333\n"
append scope_text "saif_duration_ns=$::env(SAIF_DURATION_NS)\n"
append scope_text "measurement_cycles=$::env(MEASUREMENT_CYCLES)\n"
append scope_text "saif_scope=$::env(SAIF_INSTANCE)\n"
append scope_text "saif_scope_is_gate_only=true\n"
append scope_text "exact_annotation_required_percent=100.0\n"
append scope_text "nonzero_toggle_required_percent=95.0\n"
append scope_text "tx_nonzero_entries=$::env(SAIF_TX_NONZERO_ENTRIES)\n"
append scope_text "clock_network=ideal\n"
append scope_text "wireload=ZeroWireload\n"
append scope_text "spef=false\n"
append scope_text "macros=0\n"
append scope_text "sram=false\n"
append scope_text "interconnect_extracted=false\n"
append scope_text "claim_scope=module_slice_only\n"
m448_write_text "$output_dir/reports/ptpx_scope.rpt" $scope_text

if {$::env(SAIF_TX_NONZERO_ENTRIES) != 0} {
    error "M448_FAIL_TX_NONZERO_BEFORE_SAIF_ANNOTATION"
}

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

# Fail closed inside PrimeTime before update_power.  The direct mapped-gate
# SAIF must retain both exact annotation and the independently reviewed
# nonzero-toggle floor.
set saif_text [m448_read_text "$output_dir/reports/saif_annotation_summary.rpt"]
set coverage_text [m448_read_text "$output_dir/reports/switching_coverage.rpt"]
if {![regexp {Total number of nets = ([0-9]+)} $saif_text -> total_nets]} {
    error "M448_FAIL_CANNOT_PARSE_TOTAL_NETS"
}
if {![regexp {Number of annotated nets = ([0-9]+) \(([0-9.]+)%\)} \
        $saif_text -> annotated_nets annotated_percent]} {
    error "M448_FAIL_CANNOT_PARSE_ANNOTATED_NETS"
}
if {![regexp {Total number of leaf cells = ([0-9]+)} \
        $saif_text -> total_leaf_cells]} {
    error "M448_FAIL_CANNOT_PARSE_TOTAL_LEAF_CELLS"
}
if {![regexp {Number of fully annotated leaf cells = ([0-9]+) \(([0-9.]+)%\)} \
        $saif_text -> annotated_leaf_cells annotated_leaf_percent]} {
    error "M448_FAIL_CANNOT_PARSE_ANNOTATED_LEAF_CELLS"
}
if {![regexp -line \
        {^m405_q32_elastic_selected_slice[[:space:]]+([0-9.]+)[[:space:]]+([0-9]+)[[:space:]]+([0-9]+)[[:space:]]*$} \
        $coverage_text -> nonzero_percent nonzero_nets coverage_total_nets]} {
    error "M448_FAIL_CANNOT_PARSE_NONZERO_COVERAGE"
}
if {$total_nets != 22800 || $annotated_nets != 22800 || \
        $annotated_percent != 100.0 || $coverage_total_nets != 22800} {
    error "M448_FAIL_EXACT_ANNOTATION_GATE"
}
if {$total_leaf_cells != 20803 || $annotated_leaf_cells != 20803 || \
        $annotated_leaf_percent != 100.0} {
    error "M448_FAIL_EXACT_LEAF_ANNOTATION_GATE"
}
if {$nonzero_nets < 21827 || $nonzero_percent < 95.0} {
    error "M448_FAIL_NONZERO_TOGGLE_COVERAGE_GATE"
}

set gate_fp [open "$output_dir/PTPX_POWER_GATE_PASS_PRE_UPDATE.txt" w]
puts $gate_fp "M448_PTPX_POWER_GATE_PASS_PRE_UPDATE=PASS"
puts $gate_fp "exact_annotation=$annotated_nets/$total_nets=$annotated_percent%"
puts $gate_fp "exact_leaf_annotation=$annotated_leaf_cells/$total_leaf_cells=$annotated_leaf_percent%"
puts $gate_fp "nonzero_toggle_coverage=$nonzero_nets/$coverage_total_nets=$nonzero_percent%"
puts $gate_fp "tx_nonzero_entries=$::env(SAIF_TX_NONZERO_ENTRIES)"
close $gate_fp

check_timing -verbose > "$output_dir/reports/ptpx_check_timing.rpt"
check_power > "$output_dir/reports/ptpx_check_power_pre_update.rpt"
update_timing -full
update_power

# Raw PrimeTime PX evidence.  The default report carries total/internal/
# switching/leakage and power-group (including clock-network) breakdowns;
# the hierarchy and verbose reports retain module/cell context.
report_power -unit mW -nosplit -significant_digits 8 \
    > "$output_dir/reports/ptpx_power.rpt"
report_power -hierarchy -area -unit mW -nosplit -significant_digits 8 \
    > "$output_dir/reports/ptpx_power_hierarchy.rpt"
report_power -verbose -unit mW -nosplit -significant_digits 8 \
    > "$output_dir/reports/ptpx_power_verbose.rpt"
report_power -per_clock [get_clocks core_clk] \
    > "$output_dir/reports/ptpx_power_per_clock.rpt"
report_clock > "$output_dir/reports/ptpx_clock.rpt"
report_timing -delay_type max -max_paths 10 -nworst 1 \
    > "$output_dir/reports/ptpx_timing_at_power_corner.rpt"
check_power > "$output_dir/reports/ptpx_check_power_post_update.rpt"

set marker [open "$output_dir/PTPX_INTERNAL_COMPLETE.txt" w]
puts $marker "M448_M431_M438_PRELAYOUT_STDCELL_PTPX_INTERNAL_COMPLETE=PASS"
puts $marker "power_corner=$::env(POWER_OPERATING_CONDITION)"
puts $marker "voltage_v=0.9"
puts $marker "temperature_c=25"
puts $marker "frequency_mhz=333.333333333"
puts $marker "scope=module_slice_prelayout_standard_cells_only"
close $marker
quit
