# M1630 additive source-only residual-hold closure candidate.
#
# The only mapped input is the original admitted M993/M1006 DDC.  The failed
# M1614 mapped output is evidence only and is never read.  M1614 left just
# 0.000353523 ns hold WNS across three paths while setup, area and DRC passed.
# M1630 therefore uses one deliberate 0.001 ns optimization guardband: it
# optimizes once at 0.051 ns hold uncertainty, restores 0.050 ns, and reports
# the unchanged 3.000/0.200/0.050 ns paper point.  No EDA is authorized by
# this source file alone.

set design_name m935_m912_three_stage_exact_parent_match_product_capture_island
set macro_cell TS1N28HPCPHVTB128X128M4S
set expected_macro_count 9
set area_baseline_um2 147246.392090
set area_ceiling_um2 154608.7116945
set reported_hold_uncertainty_ns 0.050
set optimization_hold_guardband_ns 0.051

set input_ddc [file normalize $::env(M1630_INPUT_DDC)]
set input_sdc [file normalize $::env(M1630_INPUT_SDC)]
set input_mapped_v [file normalize $::env(M1630_INPUT_MAPPED_V)]
set input_svf [file normalize $::env(M1630_INPUT_SVF)]
set std_slow_db [file normalize $::env(M1630_STD_SLOW_DB)]
set std_fast_db [file normalize $::env(M1630_STD_FAST_DB)]
set macro_slow_db [file normalize $::env(M1630_MACRO_SLOW_DB)]
set macro_fast_db [file normalize $::env(M1630_MACRO_FAST_DB)]
set output_dir [file normalize $::env(M1630_OUTPUT_DIR)]

file mkdir $output_dir
file mkdir "$output_dir/reports"
file mkdir "$output_dir/netlist"
set_svf "$output_dir/netlist/${design_name}_m1630_residual_hold_closed.svf"
set_app_var search_path [list [file dirname $std_slow_db] \
    [file dirname $std_fast_db] [file dirname $macro_slow_db] \
    [file dirname $macro_fast_db]]
set_app_var target_library [list $std_slow_db]
set_app_var link_library [list "*" $std_slow_db $std_fast_db \
    $macro_slow_db $macro_fast_db]
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

read_ddc $input_ddc
set design_collection [current_design]
if {[sizeof_collection $design_collection] != 1} {
    error "M1630_FAIL current_design_count=[sizeof_collection $design_collection] expected=1"
}
set active_design [get_object_name $design_collection]
if {$active_design ne $design_name} {
    error "M1630_FAIL current_design=$active_design expected=$design_name"
}
set link_status 0
redirect "$output_dir/reports/link.rpt" {set link_status [link]}
if {!$link_status} {
    error "M1630_FAIL link returned false"
}
set_min_library $std_slow_db -min_version $std_fast_db
set_min_library $macro_slow_db -min_version $macro_fast_db
set_operating_conditions ssg0p9v125c
read_sdc $input_sdc
set_wire_load_model -name ZeroWireload [current_design]

set macro_lib_cells [get_lib_cells -quiet */$macro_cell]
set macro_lib_cell_count [sizeof_collection $macro_lib_cells]
if {$macro_lib_cell_count < 1} {
    error "M1630_FAIL unresolved macro library cell $macro_cell"
}
set macro_cells_pre [get_cells -hierarchical -filter "ref_name == $macro_cell"]
set macro_count_pre [sizeof_collection $macro_cells_pre]
if {$macro_count_pre != $expected_macro_count} {
    error "M1630_FAIL macro_count_pre=$macro_count_pre expected=$expected_macro_count"
}
set_dont_touch $macro_cells_pre true

set core_clock [get_clocks core_clk]
set clock_count [sizeof_collection $core_clock]
if {$clock_count != 1} {
    error "M1630_FAIL core_clk_count=$clock_count expected=1"
}
set clock_period [get_attribute $core_clock period]
if {[expr {abs($clock_period - 3.000)}] > 0.000001} {
    error "M1630_FAIL core_clk_period=$clock_period expected=3.000"
}

set flow_fp [open "$output_dir/reports/flow_contract.rpt" w]
puts $flow_fp "flow=m1630_m993_c1_residual_hold_guardband"
puts $flow_fp "input_generation=original_m993_m1006_admitted_ddc"
puts $flow_fp "failed_m1614_output_used=false"
puts $flow_fp "input_ddc=$input_ddc"
puts $flow_fp "input_sdc=$input_sdc"
puts $flow_fp "input_mapped_v=$input_mapped_v"
puts $flow_fp "input_svf=$input_svf"
puts $flow_fp "clock_period_ns=3.000"
puts $flow_fp "setup_uncertainty_ns=0.200"
puts $flow_fp "optimization_hold_guardband_ns=0.051"
puts $flow_fp "reported_hold_uncertainty_ns=0.050"
puts $flow_fp "hold_guardband_delta_ns=0.001"
puts $flow_fp "hold_uncertainty_restore_count=1"
puts $flow_fp "ideal_clock=true"
puts $flow_fp "wireload=ZeroWireload"
puts $flow_fp "set_fix_hold_count=1"
puts $flow_fp "hold_only_incremental_mapping_count=1"
puts $flow_fp "all_compile_command_count=1"
puts $flow_fp "compile_ultra_incremental_count=0"
puts $flow_fp "generic_incremental_mapping_count=0"
puts $flow_fp "retry=false"
puts $flow_fp "false_path_count=0"
puts $flow_fp "multicycle_path_count=0"
puts $flow_fp "min_delay_exception_count=0"
puts $flow_fp "max_delay_exception_count=0"
puts $flow_fp "disabled_timing_arc_count=0"
puts $flow_fp "case_analysis_count=0"
puts $flow_fp "area_baseline_um2=$area_baseline_um2"
puts $flow_fp "area_ceiling_um2=$area_ceiling_um2"
close $flow_fp

check_design > "$output_dir/reports/check_design_prehold.rpt"
redirect "$output_dir/reports/check_timing_prehold.rpt" {check_timing}
report_qor > "$output_dir/reports/qor_prehold.rpt"
report_area -hierarchy > "$output_dir/reports/area_prehold.rpt"
report_clocks > "$output_dir/reports/clocks_prehold.rpt"
report_reference -hierarchy > "$output_dir/reports/references_prehold.rpt"
report_timing -delay_type max -max_paths 100 -nworst 10 \
    -significant_digits 6 > "$output_dir/reports/timing_setup_prehold_top100.rpt"
report_timing -delay_type min -max_paths 100 -nworst 10 \
    -significant_digits 6 > "$output_dir/reports/timing_hold_prehold_top100.rpt"
redirect "$output_dir/reports/constraint_setup_prehold_all.rpt" {
    report_constraint -max_delay -all_violators -significant_digits 6
}
redirect "$output_dir/reports/constraint_hold_prehold_all.rpt" {
    report_constraint -min_delay -all_violators -significant_digits 6
}
write_delay_summary "$output_dir/reports/setup_prehold_summary_machine.txt" \
    max PRE_GUARDBAND
write_delay_summary "$output_dir/reports/hold_prehold_summary_machine.txt" \
    min PRE_GUARDBAND

# The only optimization freedom in M1630: a 1 ps hold guardband around one
# hold-only incremental mapping.  The final reporting constraint is restored
# immediately after this command; no second pass or generic optimizer exists.
set_clock_uncertainty -hold $optimization_hold_guardband_ns $core_clock
set_fix_hold $core_clock
compile -incremental_mapping -only_hold_time
set_clock_uncertainty -hold $reported_hold_uncertainty_ns $core_clock
update_timing

set final_clock_period [get_attribute [get_clocks core_clk] period]
if {[expr {abs($final_clock_period - 3.000)}] > 0.000001} {
    error "M1630_FAIL final_clock_period=$final_clock_period expected=3.000"
}
set macro_cells_post [get_cells -hierarchical -filter "ref_name == $macro_cell"]
set macro_count_post [sizeof_collection $macro_cells_post]
if {$macro_count_post != $expected_macro_count} {
    error "M1630_FAIL macro_count_post=$macro_count_post expected=$expected_macro_count"
}

set audit_fp [open "$output_dir/reports/macro_binding_audit.txt" w]
puts $audit_fp "status=PASS_M1630_RESOLVED_LIBRARY_MACRO_STRUCTURE"
puts $audit_fp "macro_cell=$macro_cell"
puts $audit_fp "macro_lib_cell_count=$macro_lib_cell_count"
puts $audit_fp "macro_count_pre=$macro_count_pre"
puts $audit_fp "macro_count_post=$macro_count_post"
puts $audit_fp "expected_macro_count=$expected_macro_count"
puts $audit_fp "behavioral_macro_verilog_read_by_dc=false"
puts $audit_fp "inferred_parent_array_allowed=false"
close $audit_fp

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
write_delay_summary "$output_dir/reports/setup_posthold_summary_machine.txt" \
    max POST_RESTORE_REPORTED
write_delay_summary "$output_dir/reports/hold_posthold_summary_machine.txt" \
    min POST_RESTORE_REPORTED

change_names -rules verilog -hierarchy
write_file -format verilog -hierarchy -output \
    "$output_dir/netlist/${design_name}_m1630_residual_hold_closed_mapped.v"
write_sdc \
    "$output_dir/netlist/${design_name}_m1630_residual_hold_closed_mapped.sdc"
write -format ddc -hierarchy -output \
    "$output_dir/netlist/${design_name}_m1630_residual_hold_closed.ddc"
set_svf -off

set terminal_fp [open "$output_dir/TCL_INTERNAL_COMPLETE.txt" w]
puts $terminal_fp "status=M1630_DC_INTERNAL_COMPLETE__RUNNER_GATE_REQUIRED"
puts $terminal_fp "design=$design_name"
puts $terminal_fp "input_generation=original_m993_m1006_admitted_ddc"
puts $terminal_fp "failed_m1614_output_used=false"
puts $terminal_fp "optimization_hold_guardband_ns=0.051"
puts $terminal_fp "reported_hold_uncertainty_ns=0.050"
puts $terminal_fp "set_fix_hold_count=1"
puts $terminal_fp "hold_only_incremental_mapping_count=1"
puts $terminal_fp "functional_rtl_modified=false"
puts $terminal_fp "mapped_identity_modified=true"
puts $terminal_fp "formality_required=true"
puts $terminal_fp "independent_pt_required=true"
puts $terminal_fp "paper_citable=false"
close $terminal_fp
quit
