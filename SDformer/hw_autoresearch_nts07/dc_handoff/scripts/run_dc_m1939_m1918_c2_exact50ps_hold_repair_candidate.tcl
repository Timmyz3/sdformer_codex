# M1939 source-only candidate: C2 fast-min hold repair from the frozen M1811
# mapped DDC.  This file is inert until a separately reviewed one-shot runner
# supplies all environment variables and invokes dc_shell.

set sh_continue_on_error false

set input_ddc [file normalize $::env(M1939_INPUT_DDC)]
set input_sdc [file normalize $::env(M1939_INPUT_SDC)]
set std_slow_db [file normalize $::env(M1939_STD_SLOW_DB)]
set std_fast_db [file normalize $::env(M1939_STD_FAST_DB)]
set output_dir [file normalize $::env(M1939_OUTPUT_DIR)]
set expected_design $::env(M1939_EXPECTED_DESIGN)
set axis $::env(M1939_AXIS)
set area_baseline_um2 [expr {double($::env(M1939_AREA_BASELINE_UM2))}]
set area_ceiling_um2 [expr {double($::env(M1939_AREA_CEILING_UM2))}]

set reported_hold_uncertainty_ns 0.050
set optimization_hold_uncertainty_ns 0.050

file mkdir $output_dir
file mkdir "$output_dir/reports"
file mkdir "$output_dir/netlist"
set_svf "$output_dir/netlist/${expected_design}_m1939_fastmin_hold_repaired.svf"

set_app_var search_path [list [file dirname $std_slow_db] \
    [file dirname $std_fast_db]]
set_app_var target_library [list $std_slow_db]
set_app_var link_library [list "*" $std_slow_db $std_fast_db]
set_app_var verilogout_no_tri true

proc write_delay_summary {path delay_type phase} {
    set negative_paths [get_timing_paths -delay_type $delay_type \
        -max_paths 200000 -slack_lesser_than 0.0]
    set violation_count [sizeof_collection $negative_paths]
    set wns 0.0
    set tns 0.0
    if {$violation_count > 0} {
        set wns 1.0e30
        foreach_in_collection timing_path $negative_paths {
            set path_slack [get_attribute $timing_path slack]
            if {$path_slack < $wns} { set wns $path_slack }
            set tns [expr {$tns + $path_slack}]
        }
        set status VIOLATED_CAPTURED
    } else {
        set worst_path [get_timing_paths -delay_type $delay_type -max_paths 1]
        if {[sizeof_collection $worst_path] > 0} {
            set wns [get_attribute [index_collection $worst_path 0] slack]
        }
        set status MET
    }
    set fp [open $path w]
    puts $fp "phase=$phase"
    puts $fp "delay_type=$delay_type"
    puts $fp "status=$status"
    puts $fp [format "wns_ns=%.9f" $wns]
    puts $fp [format "tns_ns=%.9f" $tns]
    puts $fp "violating_paths=$violation_count"
    puts $fp "negative_path_ceiling=200000"
    close $fp
}

proc mapped_leaf_area {} {
    set cells [get_cells -quiet -hierarchical \
        -filter "is_hierarchical == false"]
    if {[sizeof_collection $cells] == 0} {
        error "M1939_FAIL no_mapped_leaf_cells_for_area"
    }
    set total 0.0
    foreach_in_collection cell $cells {
        set cell_area [get_attribute -quiet $cell area]
        if {$cell_area eq "" || ![string is double -strict $cell_area]} {
            error "M1939_FAIL invalid_cell_area=[get_object_name $cell]"
        }
        set total [expr {$total + double($cell_area)}]
    }
    return $total
}

read_ddc $input_ddc
set design_collection [current_design]
if {[sizeof_collection $design_collection] != 1} {
    error "M1939_FAIL current_design_count=[sizeof_collection $design_collection] expected=1"
}
set active_design [get_object_name $design_collection]
if {$active_design ne $expected_design} {
    error "M1939_FAIL current_design=$active_design expected=$expected_design"
}
set link_status 0
redirect "$output_dir/reports/link.rpt" {set link_status [link]}
if {!$link_status} { error "M1939_FAIL link returned false" }

set_min_library $std_slow_db -min_version $std_fast_db
set_operating_conditions ssg0p9v125c
read_sdc $input_sdc
set_wire_load_model -name ZeroWireload [current_design]

set core_clock [get_clocks core_clk]
if {[sizeof_collection $core_clock] != 1} {
    error "M1939_FAIL core_clk_count=[sizeof_collection $core_clock] expected=1"
}
set clock_period [get_attribute $core_clock period]
if {[expr {abs($clock_period - 3.000)}] > 0.000001} {
    error "M1939_FAIL core_clk_period=$clock_period expected=3.000"
}

set pre_area [mapped_leaf_area]
if {[expr {abs($pre_area - $area_baseline_um2)}] > 0.010} {
    error "M1939_FAIL pre_area=$pre_area baseline=$area_baseline_um2"
}

set flow_fp [open "$output_dir/reports/flow_contract.rpt" w]
puts $flow_fp "flow=m1939_m1918_c2_exact50ps_fastmin_hold_repair"
puts $flow_fp "m1918_70ps_guard_result=failed_area_gate_do_not_cite"
puts $flow_fp "optimization_matches_reported_contract=true"
puts $flow_fp "axis=$axis"
puts $flow_fp "input_generation=frozen_m1811_ddc_only"
puts $flow_fp "m1877_output_used_as_design_input=false"
puts $flow_fp "m1877_k8_formality_internal_passing_points=33656"
puts $flow_fp "m1877_k8_pt_setup_wns_ns=0.001767"
puts $flow_fp "m1877_k8_pt_fastmin_hold_wns_ns=-0.023259"
puts $flow_fp "m1877_k8_pt_hold_violating_paths=30442"
puts $flow_fp "m1811_k8_dc_fastmin_hold_wns_ns=-0.019042"
puts $flow_fp "observed_pt_minus_dc_hold_delta_ns=-0.004217"
puts $flow_fp "optimization_hold_uncertainty_ns=$optimization_hold_uncertainty_ns"
puts $flow_fp "reported_hold_uncertainty_ns=$reported_hold_uncertainty_ns"
puts $flow_fp "clock_period_ns=3.000"
puts $flow_fp "setup_uncertainty_ns=0.200"
puts $flow_fp "ideal_clock=true"
puts $flow_fp "wireload=ZeroWireload"
puts $flow_fp "set_fix_hold_count=1"
puts $flow_fp "hold_only_incremental_mapping_count=1"
puts $flow_fp "all_compile_command_count=1"
puts $flow_fp "false_path_count=0"
puts $flow_fp "multicycle_path_count=0"
puts $flow_fp "min_delay_exception_count=0"
puts $flow_fp "max_delay_exception_count=0"
puts $flow_fp "disabled_timing_arc_count=0"
puts $flow_fp "case_analysis_count=0"
puts $flow_fp "area_baseline_um2=$area_baseline_um2"
puts $flow_fp "area_ceiling_um2=$area_ceiling_um2"
puts $flow_fp "retry=false"
close $flow_fp

check_design > "$output_dir/reports/check_design_prehold.rpt"
redirect "$output_dir/reports/check_timing_prehold.rpt" {check_timing}
report_qor > "$output_dir/reports/qor_prehold.rpt"
report_area -hierarchy > "$output_dir/reports/area_prehold.rpt"
report_reference -hierarchy > "$output_dir/reports/references_prehold.rpt"
report_timing -delay_type max -max_paths 100 -nworst 10 \
    -significant_digits 6 > "$output_dir/reports/timing_setup_prehold_top100.rpt"
report_timing -delay_type min -max_paths 100 -nworst 10 \
    -significant_digits 6 > "$output_dir/reports/timing_hold_prehold_top100.rpt"
write_delay_summary "$output_dir/reports/setup_prehold_summary_machine.txt" max PRE_FIX
write_delay_summary "$output_dir/reports/hold_prehold_summary_machine.txt" min PRE_FIX

# The sole mapping operation uses the exact 50-ps reporting contract.  M1918
# showed that a 70-ps guard exceeded the frozen +5% area ceiling while
# substantial min-delay cost remained, so this successor does not overconstrain
# every short path by an additional 20 ps.
set_clock_uncertainty -hold $optimization_hold_uncertainty_ns $core_clock
set_fix_hold $core_clock
compile -incremental_mapping -only_hold_time
set_clock_uncertainty -hold $reported_hold_uncertainty_ns $core_clock
update_timing

set final_clock_period [get_attribute [get_clocks core_clk] period]
if {[expr {abs($final_clock_period - 3.000)}] > 0.000001} {
    error "M1939_FAIL final_clock_period=$final_clock_period expected=3.000"
}
set post_area [mapped_leaf_area]
if {$post_area > $area_ceiling_um2} {
    error "M1939_FAIL post_area=$post_area ceiling=$area_ceiling_um2"
}

report_qor > "$output_dir/reports/qor_posthold.rpt"
report_area -hierarchy > "$output_dir/reports/area_posthold.rpt"
report_hierarchy > "$output_dir/reports/hierarchy_posthold.rpt"
report_resources -hierarchy > "$output_dir/reports/resources_posthold.rpt"
report_reference -hierarchy > "$output_dir/reports/references_posthold.rpt"
report_clocks > "$output_dir/reports/clocks_posthold.rpt"
report_timing -delay_type max -max_paths 100 -nworst 10 \
    -significant_digits 6 > "$output_dir/reports/timing_setup_posthold_top100.rpt"
report_timing -delay_type min -max_paths 100 -nworst 10 \
    -significant_digits 6 > "$output_dir/reports/timing_hold_posthold_top100.rpt"
redirect "$output_dir/reports/constraint_setup_posthold_all.rpt" {
    report_constraint -max_delay -all_violators -significant_digits 6
}
redirect "$output_dir/reports/constraint_hold_posthold_all.rpt" {
    report_constraint -min_delay -all_violators -significant_digits 6
}
redirect "$output_dir/reports/constraint_design_rules_posthold.rpt" {
    report_constraint -max_capacitance -all_violators -significant_digits 6
    report_constraint -max_transition -all_violators -significant_digits 6
    report_constraint -max_fanout -all_violators -significant_digits 6
    report_constraint -min_pulse_width -all_violators -significant_digits 6
    report_constraint -min_period -all_violators -significant_digits 6
}
check_design > "$output_dir/reports/check_design_posthold.rpt"
redirect "$output_dir/reports/check_timing_posthold.rpt" {check_timing}
write_delay_summary "$output_dir/reports/setup_posthold_summary_machine.txt" max POST_RESTORE_REPORTED
write_delay_summary "$output_dir/reports/hold_posthold_summary_machine.txt" min POST_RESTORE_REPORTED

change_names -rules verilog -hierarchy
write_file -format verilog -hierarchy -output \
    "$output_dir/netlist/${expected_design}_m1939_fastmin_hold_repaired_mapped.v"
write_sdc "$output_dir/netlist/${expected_design}_m1939_fastmin_hold_repaired_mapped.sdc"
write -format ddc -hierarchy -output \
    "$output_dir/netlist/${expected_design}_m1939_fastmin_hold_repaired.ddc"
set_svf -off

set terminal_fp [open "$output_dir/TCL_INTERNAL_COMPLETE.txt" w]
puts $terminal_fp "status=M1939_DC_INTERNAL_COMPLETE__RUNNER_GATE_REQUIRED"
puts $terminal_fp "axis=$axis"
puts $terminal_fp "input_generation=frozen_m1811_ddc_only"
puts $terminal_fp "optimization_hold_uncertainty_ns=$optimization_hold_uncertainty_ns"
puts $terminal_fp "reported_hold_uncertainty_ns=$reported_hold_uncertainty_ns"
puts $terminal_fp "set_fix_hold_count=1"
puts $terminal_fp "hold_only_incremental_mapping_count=1"
puts $terminal_fp "functional_rtl_modified=false"
puts $terminal_fp "mapped_identity_modified=true"
puts $terminal_fp "formality_required=true"
puts $terminal_fp "independent_pt_required=true"
puts $terminal_fp "paper_citable=false"
close $terminal_fp
quit
