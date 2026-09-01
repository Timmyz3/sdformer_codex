# M1695 additive source-only C1 fast-min hold closure.
#
# The exact sealed M1665 DDC is the only design input.  M1678 is evidence:
# both transitive Formality segments passed, while independent PrimeTime found
# setup +0.002221 ns and fast-min hold -0.028168 ns.  The same SRAM write path
# uses a 0.097685 ns macro hold check in DC and 0.126859 ns in PrimeTime, a
# 29.174 ps delta.  M1695 therefore optimizes once at 0.081 ns hold uncertainty
# (0.050 + 0.030 correction + 0.001 guard), restores 0.050 ns, and emits the
# unchanged 3.000/0.200/0.050 ns reporting point.  This source does not itself
# authorize or execute EDA.

set design_name m935_m912_three_stage_exact_parent_match_product_capture_island
set macro_cell TS1N28HPCPHVTB128X128M4S
set expected_macro_count 9
set area_baseline_um2 152898.625984
set area_ceiling_um2 168188.4885824
set reported_hold_uncertainty_ns 0.050
set optimization_hold_uncertainty_ns 0.081

set input_ddc [file normalize $::env(M1695_INPUT_DDC)]
set input_sdc [file normalize $::env(M1695_INPUT_SDC)]
set std_slow_db [file normalize $::env(M1695_STD_SLOW_DB)]
set std_fast_db [file normalize $::env(M1695_STD_FAST_DB)]
set macro_slow_db [file normalize $::env(M1695_MACRO_SLOW_DB)]
set macro_fast_db [file normalize $::env(M1695_MACRO_FAST_DB)]
set output_dir [file normalize $::env(M1695_OUTPUT_DIR)]

file mkdir $output_dir
file mkdir "$output_dir/reports"
file mkdir "$output_dir/netlist"
set_svf "$output_dir/netlist/${design_name}_m1695_fastmin_hold_closed.svf"
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
    error "M1695_FAIL current_design_count=[sizeof_collection $design_collection] expected=1"
}
set active_design [get_object_name $design_collection]
if {$active_design ne $design_name} {
    error "M1695_FAIL current_design=$active_design expected=$design_name"
}
set link_status 0
redirect "$output_dir/reports/link.rpt" {set link_status [link]}
if {!$link_status} { error "M1695_FAIL link returned false" }
set_min_library $std_slow_db -min_version $std_fast_db
set_min_library $macro_slow_db -min_version $macro_fast_db
set_operating_conditions ssg0p9v125c
read_sdc $input_sdc
set_wire_load_model -name ZeroWireload [current_design]

set macro_lib_cells [get_lib_cells -quiet */$macro_cell]
set macro_lib_cell_count [sizeof_collection $macro_lib_cells]
if {$macro_lib_cell_count < 1} {
    error "M1695_FAIL unresolved macro library cell $macro_cell"
}
set macro_cells_pre [get_cells -hierarchical -filter "ref_name == $macro_cell"]
set macro_count_pre [sizeof_collection $macro_cells_pre]
if {$macro_count_pre != $expected_macro_count} {
    error "M1695_FAIL macro_count_pre=$macro_count_pre expected=$expected_macro_count"
}
set_dont_touch $macro_cells_pre true

set core_clock [get_clocks core_clk]
if {[sizeof_collection $core_clock] != 1} {
    error "M1695_FAIL core_clk_count=[sizeof_collection $core_clock] expected=1"
}
set clock_period [get_attribute $core_clock period]
if {[expr {abs($clock_period - 3.000)}] > 0.000001} {
    error "M1695_FAIL core_clk_period=$clock_period expected=3.000"
}

set flow_fp [open "$output_dir/reports/flow_contract.rpt" w]
puts $flow_fp "flow=m1695_m1665_c1_fastmin_hold_closure"
puts $flow_fp "input_generation=frozen_m1665_ddc_only"
puts $flow_fp "m1678_output_used_as_design_input=false"
puts $flow_fp "m1678_transitive_formality_segments_passed=2"
puts $flow_fp "m1678_pt_setup_wns_ns=0.002221"
puts $flow_fp "m1678_pt_fastmin_hold_wns_ns=-0.028168"
puts $flow_fp "m1678_pt_hold_violating_paths=10610"
puts $flow_fp "m1678_pt_hold_tns_ns=-40.24"
puts $flow_fp "macro_hold_check_dc_ns=0.097685"
puts $flow_fp "macro_hold_check_pt_ns=0.126859"
puts $flow_fp "macro_hold_check_delta_ns=0.029174"
puts $flow_fp "optimization_hold_uncertainty_ns=0.081"
puts $flow_fp "reported_hold_uncertainty_ns=0.050"
puts $flow_fp "hold_uncertainty_restore_count=1"
puts $flow_fp "clock_period_ns=3.000"
puts $flow_fp "setup_uncertainty_ns=0.200"
puts $flow_fp "ideal_clock=true"
puts $flow_fp "wireload=ZeroWireload"
puts $flow_fp "set_fix_hold_count=1"
puts $flow_fp "hold_only_incremental_mapping_count=1"
puts $flow_fp "all_compile_command_count=1"
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
report_reference -hierarchy > "$output_dir/reports/references_prehold.rpt"
report_timing -delay_type max -max_paths 100 -nworst 10 \
    -significant_digits 6 > "$output_dir/reports/timing_setup_prehold_top100.rpt"
report_timing -delay_type min -max_paths 100 -nworst 10 \
    -significant_digits 6 > "$output_dir/reports/timing_hold_prehold_top100.rpt"
write_delay_summary "$output_dir/reports/setup_prehold_summary_machine.txt" max PRE_FIX
write_delay_summary "$output_dir/reports/hold_prehold_summary_machine.txt" min PRE_FIX

# The sole optimization operation: one hold-only incremental mapping with a
# PT-calibrated fast-min optimization margin.  Restore 0.050 ns immediately;
# all final reports and the emitted SDC use the unchanged paper constraint.
set_clock_uncertainty -hold $optimization_hold_uncertainty_ns $core_clock
set_fix_hold $core_clock
compile -incremental_mapping -only_hold_time
set_clock_uncertainty -hold $reported_hold_uncertainty_ns $core_clock
update_timing

set final_clock_period [get_attribute [get_clocks core_clk] period]
if {[expr {abs($final_clock_period - 3.000)}] > 0.000001} {
    error "M1695_FAIL final_clock_period=$final_clock_period expected=3.000"
}
set macro_cells_post [get_cells -hierarchical -filter "ref_name == $macro_cell"]
set macro_count_post [sizeof_collection $macro_cells_post]
if {$macro_count_post != $expected_macro_count} {
    error "M1695_FAIL macro_count_post=$macro_count_post expected=$expected_macro_count"
}

set audit_fp [open "$output_dir/reports/macro_binding_audit.txt" w]
puts $audit_fp "status=PASS_M1695_RESOLVED_LIBRARY_MACRO_STRUCTURE"
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
write_delay_summary "$output_dir/reports/setup_posthold_summary_machine.txt" max POST_RESTORE_REPORTED
write_delay_summary "$output_dir/reports/hold_posthold_summary_machine.txt" min POST_RESTORE_REPORTED

change_names -rules verilog -hierarchy
write_file -format verilog -hierarchy -output \
    "$output_dir/netlist/${design_name}_m1695_fastmin_hold_closed_mapped.v"
write_sdc "$output_dir/netlist/${design_name}_m1695_fastmin_hold_closed_mapped.sdc"
write -format ddc -hierarchy -output \
    "$output_dir/netlist/${design_name}_m1695_fastmin_hold_closed.ddc"
set_svf -off

set terminal_fp [open "$output_dir/TCL_INTERNAL_COMPLETE.txt" w]
puts $terminal_fp "status=M1695_DC_INTERNAL_COMPLETE__RUNNER_GATE_REQUIRED"
puts $terminal_fp "input_generation=frozen_m1665_ddc_only"
puts $terminal_fp "optimization_hold_uncertainty_ns=0.081"
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
