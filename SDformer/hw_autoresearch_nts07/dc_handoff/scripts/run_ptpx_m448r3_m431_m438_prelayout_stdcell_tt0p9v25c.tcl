set design_name $::env(DESIGN_NAME)
set tt_lib_db [file normalize $::env(TT_LIB_DB)]
set sdc_lib_db [file normalize $::env(SDC_LIB_DB)]
set mapped_netlist [file normalize $::env(MAPPED_NETLIST)]
set mapped_sdc [file normalize $::env(MAPPED_SDC)]
set gate_saif [file normalize $::env(GATE_SAIF_FILE)]
set output_dir [file normalize $::env(OUTPUT_DIR)]

proc m448r3_read_text {path} {
    set fp [open $path r]
    set value [read $fp]
    close $fp
    return $value
}

proc m448r3_write_text {path value} {
    set fp [open $path w]
    puts -nonewline $fp $value
    close $fp
}

proc m448r3_append_ledger {path value} {
    set fp [open $path a]
    puts $fp $value
    close $fp
}

file mkdir "$output_dir/reports"
set power_ledger "$output_dir/power_call_ledger.txt"
m448r3_write_text $power_ledger ""
set_app_var search_path [list [file dirname $tt_lib_db] [file dirname $sdc_lib_db]]
set_app_var link_path [list "*" $tt_lib_db $sdc_lib_db]
read_verilog $mapped_netlist
current_design $design_name
link_design $design_name

read_sdc $mapped_sdc
set_operating_conditions $::env(POWER_OPERATING_CONDITION) \
    -library $::env(POWER_LIBRARY_NAME)
set_wire_load_model -name ZeroWireload -library $::env(POWER_LIBRARY_NAME)
set nonclock_inputs [remove_from_collection [all_inputs] [get_ports clk_core]]
if {[sizeof_collection $nonclock_inputs] != 1666} {
    error "M448R3_FAIL_NONCLOCK_INPUT_POPULATION"
}
# 100 ps follows the repository's frozen Synopsys input-driver convention.
# reset_n is included solely to avoid an unphysical 0 ps power-table ramp; it
# remains false-pathed and static in the measured SAIF window.
set_input_transition 0.100 $nonclock_inputs
set power_enable_analysis true
set_app_var power_analysis_mode averaged
set_app_var timing_save_pin_arrival_and_slack true

set scope_text ""
append scope_text "milestone=M448R3\n"
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
append scope_text "nonclock_input_count=1666\n"
append scope_text "primary_input_transition_ns=0.100\n"
append scope_text "sensitivity_input_transition_ns=0.050,0.100,0.200\n"
append scope_text "exact_annotation_required_percent=100.0\n"
append scope_text "nonzero_toggle_required_percent=95.0\n"
append scope_text "tx_nonzero_entries=$::env(SAIF_TX_NONZERO_ENTRIES)\n"
append scope_text "clock_network=ideal_no_cts\n"
append scope_text "wireload=ZeroWireload\n"
append scope_text "spef=false\n"
append scope_text "macros=0\n"
append scope_text "sram=false\n"
append scope_text "interconnect_extracted=false\n"
append scope_text "claim_scope=M416_selected_slice_only\n"
m448r3_write_text "$output_dir/reports/ptpx_scope.rpt" $scope_text

if {$::env(SAIF_TX_NONZERO_ENTRIES) != 0} {
    error "M448R3_FAIL_TX_NONZERO_BEFORE_SAIF_ANNOTATION"
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

set saif_text [m448r3_read_text "$output_dir/reports/saif_annotation_summary.rpt"]
set coverage_text [m448r3_read_text "$output_dir/reports/switching_coverage.rpt"]
if {![regexp {Total number of nets = ([0-9]+)} $saif_text -> total_nets]} {
    error "M448R3_FAIL_CANNOT_PARSE_TOTAL_NETS"
}
if {![regexp {Number of annotated nets = ([0-9]+) \(([0-9.]+)%\)} \
        $saif_text -> annotated_nets annotated_percent]} {
    error "M448R3_FAIL_CANNOT_PARSE_ANNOTATED_NETS"
}
if {![regexp {Total number of leaf cells = ([0-9]+)} \
        $saif_text -> total_leaf_cells]} {
    error "M448R3_FAIL_CANNOT_PARSE_TOTAL_LEAF_CELLS"
}
if {![regexp {Number of fully annotated leaf cells = ([0-9]+) \(([0-9.]+)%\)} \
        $saif_text -> annotated_leaf_cells annotated_leaf_percent]} {
    error "M448R3_FAIL_CANNOT_PARSE_ANNOTATED_LEAF_CELLS"
}
if {![regexp -line \
        {^m405_q32_elastic_selected_slice[[:space:]]+([0-9.]+)[[:space:]]+([0-9]+)[[:space:]]+([0-9]+)[[:space:]]*$} \
        $coverage_text -> nonzero_percent nonzero_nets coverage_total_nets]} {
    error "M448R3_FAIL_CANNOT_PARSE_NONZERO_COVERAGE"
}
if {$total_nets != 22800 || $annotated_nets != 22800 || \
        $annotated_percent != 100.0 || $coverage_total_nets != 22800} {
    error "M448R3_FAIL_EXACT_ANNOTATION_GATE"
}
if {$total_leaf_cells != 20803 || $annotated_leaf_cells != 20803 || \
        $annotated_leaf_percent != 100.0} {
    error "M448R3_FAIL_EXACT_LEAF_ANNOTATION_GATE"
}
if {$nonzero_nets < 21827 || $nonzero_percent < 95.0} {
    error "M448R3_FAIL_NONZERO_TOGGLE_COVERAGE_GATE"
}

proc m448r3_power_point {label input_slew output_dir nonclock_inputs power_ledger} {
    set_input_transition $input_slew $nonclock_inputs
    update_timing -full
    set check_path "$output_dir/reports/ptpx_check_power_${label}_pre_update.rpt"
    check_power -verbose -significant_digits 8 > $check_path
    set check_text [m448r3_read_text $check_path]
    if {![regexp {check_power succeeded\.} $check_text]} {
        error "M448R3_FAIL_CHECK_POWER_${label}"
    }
    if {[regexp {out_of_range|out of ramp range|Warning:} $check_text]} {
        error "M448R3_FAIL_CHECK_POWER_DETAIL_${label}"
    }
    m448r3_append_ledger $power_ledger "$label check_power_pass"
    update_power
    m448r3_append_ledger $power_ledger "$label update_power_complete"
    set power_path "$output_dir/reports/ptpx_power_${label}.rpt"
    report_power -unit mW -nosplit -significant_digits 8 > $power_path
    if {![file exists $power_path] || [file size $power_path] <= 0} {
        error "M448R3_FAIL_EMPTY_POWER_REPORT_${label}"
    }
    set power_text [m448r3_read_text $power_path]
    foreach field {"Net Switching Power" "Cell Internal Power" \
            "Cell Leakage Power" "Total Power"} {
        set field_pattern [format {%s[[:space:]]*=[[:space:]]*([0-9.eE+-]+)} \
            $field]
        if {[regexp -all $field_pattern $power_text] != 1} {
            error "M448R3_FAIL_NONUNIQUE_POWER_FIELD_${label}_${field}"
        }
    }
    m448r3_append_ledger $power_ledger "$label report_power_complete"
}

check_timing -verbose > "$output_dir/reports/ptpx_check_timing.rpt"
report_port -verbose $nonclock_inputs \
    > "$output_dir/reports/nonclock_input_ports.rpt"
m448r3_power_point primary_100ps 0.100 $output_dir $nonclock_inputs $power_ledger

set gate_fp [open "$output_dir/PTPX_POWER_GATE_PASS_PRE_UPDATE.txt" w]
puts $gate_fp "M448R3_PTPX_POWER_GATE_PASS_PRE_UPDATE=PASS"
puts $gate_fp "exact_annotation=$annotated_nets/$total_nets=$annotated_percent%"
puts $gate_fp "exact_leaf_annotation=$annotated_leaf_cells/$total_leaf_cells=$annotated_leaf_percent%"
puts $gate_fp "nonzero_toggle_coverage=$nonzero_nets/$coverage_total_nets=$nonzero_percent%"
puts $gate_fp "tx_nonzero_entries=$::env(SAIF_TX_NONZERO_ENTRIES)"
puts $gate_fp "primary_input_transition_ns=0.100"
puts $gate_fp "check_power_primary_100ps=succeeded"
close $gate_fp

report_power -hierarchy -area -unit mW -nosplit -significant_digits 8 \
    > "$output_dir/reports/ptpx_power_primary_100ps_hierarchy.rpt"
report_power -verbose -unit mW -nosplit -significant_digits 8 \
    > "$output_dir/reports/ptpx_power_primary_100ps_verbose.rpt"
report_power -per_clock [get_clocks core_clk] \
    > "$output_dir/reports/ptpx_power_primary_100ps_per_clock.rpt"
report_clock > "$output_dir/reports/ptpx_clock.rpt"
report_timing -delay_type max -max_paths 10 -nworst 1 \
    > "$output_dir/reports/ptpx_timing_primary_100ps.rpt"

m448r3_power_point sensitivity_050ps 0.050 $output_dir $nonclock_inputs $power_ledger
m448r3_power_point sensitivity_200ps 0.200 $output_dir $nonclock_inputs $power_ledger

set marker [open "$output_dir/PTPX_INTERNAL_COMPLETE.txt" w]
puts $marker "M448R3_M431_M438_PRELAYOUT_STDCELL_PTPX_INTERNAL_COMPLETE=PASS"
puts $marker "power_corner=$::env(POWER_OPERATING_CONDITION)"
puts $marker "voltage_v=0.9"
puts $marker "temperature_c=25"
puts $marker "frequency_mhz=333.333333333"
puts $marker "primary_input_transition_ns=0.100"
puts $marker "sensitivity_input_transition_ns=0.050,0.200"
puts $marker "scope=M416_selected_slice_prelayout_standard_cells_only"
close $marker
quit
