set design_name m935_m912_three_stage_exact_parent_match_product_capture_island
set macro_cell TS1N28HPCPHVTB128X128M4S
set expected_macro_count 9
set hw_root [file normalize $::env(HW_ROOT)]
set rtl_filelist [file normalize $::env(RTL_FILELIST)]
set sdc_file [file normalize $::env(SDC_FILE)]
set std_slow_db [file normalize $::env(STD_SLOW_DB)]
set std_fast_db [file normalize $::env(STD_FAST_DB)]
set macro_slow_db [file normalize $::env(MACRO_SLOW_DB)]
set macro_fast_db [file normalize $::env(MACRO_FAST_DB)]
set output_dir [file normalize $::env(OUTPUT_DIR)]

file mkdir $output_dir
file mkdir "$output_dir/reports"
file mkdir "$output_dir/netlist"
set_svf "$output_dir/netlist/${design_name}.svf"
set_app_var search_path [list $hw_root [file dirname $std_slow_db] \
    [file dirname $std_fast_db] [file dirname $macro_slow_db] \
    [file dirname $macro_fast_db]]
set_app_var target_library [list $std_slow_db]
set_app_var link_library [list "*" $std_slow_db $std_fast_db \
    $macro_slow_db $macro_fast_db]
set_app_var verilogout_no_tri true
set_app_var hdlin_auto_save_templates true

set fp [open $rtl_filelist r]
set rtl_files {}
while {[gets $fp line] >= 0} {
    set line [string trim $line]
    if {$line ne "" && ![string match "#*" $line]} {
        lappend rtl_files [file normalize "$hw_root/$line"]
    }
}
close $fp

analyze -format sverilog -define SYNTHESIS $rtl_files
elaborate $design_name
current_design $design_name
redirect "$output_dir/reports/link.rpt" {link}
uniquify
set_min_library $std_slow_db -min_version $std_fast_db
set_min_library $macro_slow_db -min_version $macro_fast_db
set_operating_conditions ssg0p9v125c

set macro_lib_cells [get_lib_cells -quiet */$macro_cell]
set macro_lib_cell_count [sizeof_collection $macro_lib_cells]
if {$macro_lib_cell_count < 1} {
    error "M962_FAIL unresolved macro library cell $macro_cell"
}
set macro_cells_pre [get_cells -hierarchical -filter "ref_name == $macro_cell"]
set macro_count_pre [sizeof_collection $macro_cells_pre]
if {$macro_count_pre != $expected_macro_count} {
    error "M962_FAIL macro_count_pre=$macro_count_pre expected=$expected_macro_count"
}
set_dont_touch $macro_cells_pre true

source $sdc_file
set_wire_load_model -name ZeroWireload [current_design]
set flow_fp [open "$output_dir/reports/flow_contract.rpt" w]
puts $flow_fp "flow=m962_m935_three_stage_match_macro_aware_candidate"
puts $flow_fp "clock_period_ns=3.000"
puts $flow_fp "ideal_clock=true"
puts $flow_fp "wireload=ZeroWireload"
puts $flow_fp "compile_ultra_count=1"
puts $flow_fp "incremental_compile_count=0"
puts $flow_fp "hold_fix_command_count=0"
puts $flow_fp "false_path_count=0"
puts $flow_fp "multicycle_path_count=0"
puts $flow_fp "disabled_timing_arc_count=0"
puts $flow_fp "setup_failure_is_sealed_negative_evidence=true"
puts $flow_fp "hold_diagnostic_only=true"
close $flow_fp

check_design > "$output_dir/reports/check_design_precompile.rpt"
redirect "$output_dir/reports/check_timing_precompile.rpt" {check_timing}
set pre_fp [open "$output_dir/reports/check_timing_precompile.rpt" r]
set pre_text [read $pre_fp]
close $pre_fp
set pre_tim209 [regexp -all -- {TIM-209} $pre_text]
set pre_opt150 [regexp -all -- {OPT-150} $pre_text]
set loop_fp [open "$output_dir/reports/precompile_loop_gate.rpt" w]
puts $loop_fp "TIM-209=$pre_tim209"
puts $loop_fp "OPT-150=$pre_opt150"
if {$pre_tim209 != 0 || $pre_opt150 != 0} {
    puts $loop_fp "status=FAIL_PRECOMPILE_LOOP"
    close $loop_fp
    error "M962_FAIL precompile TIM-209/OPT-150"
}
puts $loop_fp "status=PASS_PRECOMPILE_LOOP_GATE"
close $loop_fp

report_resources -hierarchy > "$output_dir/reports/resources_precompile.rpt"
report_reference -hierarchy > "$output_dir/reports/references_precompile.rpt"
compile_ultra -no_autoungroup
update_timing

set macro_cells_post [get_cells -hierarchical -filter "ref_name == $macro_cell"]
set macro_count_post [sizeof_collection $macro_cells_post]
if {$macro_count_post != $expected_macro_count} {
    error "M962_FAIL macro_count_post=$macro_count_post expected=$expected_macro_count"
}
set audit [open "$output_dir/reports/macro_binding_audit.txt" w]
puts $audit "status=PASS_M962_RESOLVED_LIBRARY_MACRO_STRUCTURE"
puts $audit "macro_cell=$macro_cell"
puts $audit "macro_lib_cell_count=$macro_lib_cell_count"
puts $audit "macro_count_pre=$macro_count_pre"
puts $audit "macro_count_post=$macro_count_post"
puts $audit "expected_macro_count=$expected_macro_count"
puts $audit "behavioral_macro_verilog_read_by_dc=false"
puts $audit "inferred_parent_array_allowed=false"
close $audit

report_hierarchy > "$output_dir/reports/hierarchy_postcompile.rpt"
report_resources -hierarchy > "$output_dir/reports/resources_postcompile.rpt"
report_reference -hierarchy > "$output_dir/reports/references_postcompile.rpt"
report_qor > "$output_dir/reports/qor.rpt"
report_area -hierarchy > "$output_dir/reports/area_hierarchy.rpt"
report_clocks > "$output_dir/reports/clocks.rpt"
report_timing -delay_type max -max_paths 100 -nworst 10 \
    -significant_digits 4 > "$output_dir/reports/timing_setup_top100.rpt"
redirect "$output_dir/reports/constraint_setup_all.rpt" {
    report_constraint -max_delay -all_violators -significant_digits 4
}
redirect "$output_dir/reports/constraint_max_capacitance.rpt" {
    report_constraint -max_capacitance -all_violators -significant_digits 4
}
redirect "$output_dir/reports/constraint_max_transition.rpt" {
    report_constraint -max_transition -all_violators -significant_digits 4
}
redirect "$output_dir/reports/constraint_max_fanout.rpt" {
    report_constraint -max_fanout -all_violators -significant_digits 4
}
check_design > "$output_dir/reports/check_design_postcompile.rpt"
redirect "$output_dir/reports/check_timing_postcompile.rpt" {check_timing}

# Capture every negative max-delay path (up to a deliberately high ceiling)
# so a failed 3 ns experiment remains useful and cannot disappear into a
# FAILED_DO_NOT_CITE quarantine merely because setup is negative.
set negative_paths [get_timing_paths -delay_type max -max_paths 200000 \
    -slack_lesser_than 0.0]
set violation_count [sizeof_collection $negative_paths]
set setup_wns 0.0
set setup_tns 0.0
if {$violation_count > 0} {
    set setup_wns 1.0e30
    foreach_in_collection path $negative_paths {
        set path_slack [get_attribute $path slack]
        if {$path_slack < $setup_wns} { set setup_wns $path_slack }
        set setup_tns [expr {$setup_tns + $path_slack}]
    }
    set setup_status VIOLATED_CAPTURED
} else {
    set worst_path [get_timing_paths -delay_type max -max_paths 1]
    if {[sizeof_collection $worst_path] > 0} {
        set setup_wns [get_attribute [index_collection $worst_path 0] slack]
    }
    set setup_status MET
}
set sum_fp [open "$output_dir/reports/setup_summary_machine.txt" w]
puts $sum_fp "status=$setup_status"
puts $sum_fp [format "setup_wns_ns=%.6f" $setup_wns]
puts $sum_fp [format "setup_tns_ns=%.6f" $setup_tns]
puts $sum_fp "setup_violating_paths=$violation_count"
puts $sum_fp "clock_period_ns=3.000"
close $sum_fp

change_names -rules verilog -hierarchy
write_file -format verilog -hierarchy \
    -output "$output_dir/netlist/${design_name}_mapped.v"
write_sdc "$output_dir/netlist/${design_name}_mapped.sdc"
write -format ddc -hierarchy -output "$output_dir/netlist/${design_name}.ddc"
set_svf -off

set terminal_fp [open "$output_dir/TCL_PASS_TERMINAL.txt" w]
puts $terminal_fp "status=PASS_M962_DC_EXECUTION_AND_REPORT_CLOSURE"
puts $terminal_fp "setup_status=$setup_status"
puts $terminal_fp "TIM-209=$pre_tim209"
puts $terminal_fp "OPT-150=$pre_opt150"
puts $terminal_fp "macro_count_pre=$macro_count_pre"
puts $terminal_fp "macro_count_post=$macro_count_post"
puts $terminal_fp "hold_signoff=false"
puts $terminal_fp "power_measured=false"
close $terminal_fp
quit
