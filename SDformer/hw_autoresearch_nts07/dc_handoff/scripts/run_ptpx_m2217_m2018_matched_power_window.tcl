# M2217 matched averaged standard-cell power for one axis/window.  The same
# script and TT/0.9-V/25-C/3-ns/ZeroWireload constraints apply to all six
# points.  External SRAM energy is deliberately added by the Python parser.
set axis $::env(M2217_AXIS)
set stratum $::env(M2217_STRATUM)
set design_name $::env(M2217_DESIGN_NAME)
set tt_lib_db [file normalize $::env(M2217_TT_LIB_DB)]
set mapped_netlist [file normalize $::env(M2217_MAPPED_NETLIST)]
set mapped_sdc [file normalize $::env(M2217_MAPPED_SDC)]
set default_map [file normalize $::env(M2217_DEFAULT_MAP)]
set essential_map [file normalize $::env(M2217_ESSENTIAL_MAP)]
set rtl_saif [file normalize $::env(M2217_RTL_SAIF)]
set output_dir [file normalize $::env(M2217_OUTPUT_DIR)]
set measurement_cycles $::env(M2217_MEASUREMENT_CYCLES)
set scalar_weight_reads $::env(M2217_ACCEPTED_BANK_REQUESTS)
if {$axis ne "ordinary_lru4" && $axis ne "tsbg_b4"} { error "M2217_FAIL_AXIS" }
if {$stratum ne "low" && $stratum ne "median" && $stratum ne "high"} {
    error "M2217_FAIL_STRATUM"
}
if {![string is integer -strict $measurement_cycles] || $measurement_cycles <= 0
        || ![string is integer -strict $scalar_weight_reads]
        || $scalar_weight_reads <= 0} { error "M2217_FAIL_LEDGER_INPUT" }
foreach path [list $tt_lib_db $mapped_netlist $mapped_sdc $default_map \
        $essential_map $rtl_saif] {
    if {![file isfile $path]} { error "M2217_FAIL_MISSING_INPUT_$path" }
}
file mkdir "$output_dir/reports"
set_app_var search_path [list [file dirname $tt_lib_db]]
read_db $tt_lib_db
set_app_var link_path [list "*" $tt_lib_db]
read_verilog $mapped_netlist
current_design $design_name
link_design $design_name
if {[sizeof_collection [get_cells -hierarchical -filter "is_black_box==true"]] != 0
        || [sizeof_collection [get_cells -hierarchical -filter "is_memory_cell==true"]] != 0} {
    error "M2217_FAIL_BLACKBOX_OR_MACRO_IN_LOGIC_AXIS"
}
read_sdc $mapped_sdc
set_operating_conditions tt0p9v25c -library tcbn28hpcplusbwp35p140tt0p9v25c
set_wire_load_model -name ZeroWireload -library tcbn28hpcplusbwp35p140tt0p9v25c
set nonclock_inputs [remove_from_collection [all_inputs] [get_ports clk_core]]
set_input_transition 0.100 $nonclock_inputs
set power_enable_analysis true
set_app_var power_analysis_mode averaged
set_app_var timing_save_pin_arrival_and_slack true
source $default_map
source $essential_map
redirect -file "$output_dir/reports/saif_annotation_summary.rpt" {
    read_saif -strip_path tb_m2217_m2018_tsbg_matched_native_saif_power.dut_axis \
        -report_inconsistent_annotation "$output_dir/reports/inconsistent_annotation.rpt" $rtl_saif
}
report_switching_activity -coverage -include_mapping_types \
    > "$output_dir/reports/switching_coverage.rpt"
report_switching_activity -list_annotated -include_mapping_types \
    > "$output_dir/reports/switching_annotated.rpt"
report_switching_activity -list_not_annotated -include_mapping_types \
    > "$output_dir/reports/switching_unannotated.rpt"
proc m2217_read_text {path} { set fp [open $path r]; set value [read $fp]; close $fp; return $value }
set annotation_text [m2217_read_text "$output_dir/reports/saif_annotation_summary.rpt"]
if {![regexp {Total number of nets = ([0-9]+)} $annotation_text -> total_nets]
        || ![regexp {Number of annotated nets = ([0-9]+) \(([0-9.]+)%\)} $annotation_text -> ann_nets ann_pct]
        || ![regexp {Total number of leaf cells = ([0-9]+)} $annotation_text -> total_leaf]
        || ![regexp {Number of fully annotated leaf cells = ([0-9]+) \(([0-9.]+)%\)} $annotation_text -> ann_leaf leaf_pct]
        || $ann_pct < 95.0 || $leaf_pct < 95.0} {
    error "M2217_FAIL_ANNOTATION_GATE"
}
set inconsistent_text [m2217_read_text "$output_dir/reports/inconsistent_annotation.rpt"]
set inconsistent_rows 0
foreach line [split $inconsistent_text "\n"] {
    set trimmed [string trim $line]
    if {$trimmed ne "" && ![string match "obj_name*" $trimmed]
            && ![regexp {^-+$} $trimmed]} { incr inconsistent_rows }
}
if {$inconsistent_rows != 0} { error "M2217_FAIL_INCONSISTENT_ANNOTATION" }
set coverage_text [m2217_read_text "$output_dir/reports/switching_coverage.rpt"]
if {![regexp -line {^m2018[^ \t]*[ \t]+([0-9.]+)[ \t]+([0-9]+)[ \t]+([0-9]+)[ \t]*$} \
        $coverage_text -> toggle_pct toggled_nets toggle_total]
        || $toggle_total <= 0 || $toggled_nets <= 0 || $toggle_pct < 20.0} {
    error "M2217_FAIL_NONZERO_TOGGLE_COVERAGE"
}
foreach cone {mem_req_valid mem_rsp_valid bridge_valid commit_valid \
        mem_req_accept mem_rsp_accept bridge_accept commit_accept} {
    set nets [get_nets -quiet "${cone}*"]
    if {[sizeof_collection $nets] == 0} { error "M2217_FAIL_MISSING_CONE_$cone" }
    report_switching_activity $nets > "$output_dir/reports/critical_${cone}_activity.rpt"
    set live 0
    foreach_in_collection net $nets {
        set rate [get_attribute -quiet $net toggle_rate]
        if {$rate ne "" && $rate > 0.0} { set live 1 }
    }
    if {!$live} { error "M2217_FAIL_ZERO_CONE_$cone" }
}
check_timing -verbose > "$output_dir/reports/check_timing.rpt"
update_timing -full
check_power -verbose -significant_digits 8 > "$output_dir/reports/check_power.rpt"
if {![regexp {check_power succeeded\.} [m2217_read_text "$output_dir/reports/check_power.rpt"]]} {
    error "M2217_FAIL_CHECK_POWER"
}
update_power
report_power -unit mW -nosplit -significant_digits 8 > "$output_dir/reports/power.rpt"
report_power -hierarchy -area -unit mW -nosplit -significant_digits 8 \
    > "$output_dir/reports/power_hierarchy.rpt"
report_timing -delay_type max -max_paths 20 -nworst 2 \
    > "$output_dir/reports/timing_setup_tt.rpt"
set fp [open "$output_dir/reports/scope_and_boundary.rpt" w]
puts $fp "milestone=M2217"
puts $fp "axis=$axis"
puts $fp "stratum=$stratum"
puts $fp "design=$design_name"
puts $fp "saif_scope=tb_m2217_m2018_tsbg_matched_native_saif_power.dut_axis"
puts $fp "measurement_cycles=$measurement_cycles"
puts $fp "measurement_duration_ns=[expr {$measurement_cycles * 3.0}]"
puts $fp "accepted_bank_requests=$scalar_weight_reads"
puts $fp "weight_sram_capacity_bytes=294912"
puts $fp "weight_sram_macro_count=16"
puts $fp "weight_sram_dynamic_energy_in_ptpx=false"
puts $fp "analysis=averaged_prelayout_standard_cell_power"
puts $fp "power_corner=tt0p9v25c"
puts $fp "clock_network=ideal_no_cts"
puts $fp "wireload=ZeroWireload"
puts $fp "macro_count_in_logic_netlist=0"
close $fp
set fp [open "$output_dir/PTPX_INTERNAL_COMPLETE.txt" w]
puts $fp "PASS_M2217_PTPX_INTERNAL_PENDING_M2219_PARSER_AND_M2220_REVIEW"
close $fp
quit
