set design_name m935_m912_three_stage_exact_parent_match_product_capture_island
set std_slow_db [file normalize $::env(M1722_STD_SLOW_DB)]
set std_fast_db [file normalize $::env(M1722_STD_FAST_DB)]
set macro_slow_db [file normalize $::env(M1722_MACRO_SLOW_DB)]
set macro_fast_db [file normalize $::env(M1722_MACRO_FAST_DB)]
set mapped_netlist [file normalize $::env(M1722_M1701_IMPLEMENTATION_NETLIST)]
set mapped_sdc [file normalize $::env(M1722_M1701_IMPLEMENTATION_SDC)]
set output_dir [file normalize $::env(M1722_PT_OUTPUT_DIR)]

set std_slow_lib tcbn28hpcplusbwp35p140ssg0p9v125c
set std_fast_lib tcbn28hpcplusbwp35p140ffg1p05vm40c
set macro_cell TS1N28HPCPHVTB128X128M4S

file mkdir "$output_dir/reports"
set_app_var search_path [list \
    [file dirname $std_slow_db] [file dirname $std_fast_db] \
    [file dirname $macro_slow_db] [file dirname $macro_fast_db]]
set_app_var link_path [list "*" $std_slow_db $macro_slow_db]

read_verilog $mapped_netlist
current_design $design_name
link_design $design_name
set_min_library $std_slow_db -min_version $std_fast_db
set_min_library $macro_slow_db -min_version $macro_fast_db
read_sdc $mapped_sdc
set_operating_conditions -analysis_type on_chip_variation \
    -max ssg0p9v125c -max_library $std_slow_lib \
    -min ffg1p05vm40c -min_library $std_fast_lib

set macro_count [sizeof_collection \
    [get_cells -hierarchical -filter "ref_name==$macro_cell"]]
if {$macro_count != 9} {
    error "M1722 expected exactly nine physical SRAM macro instances"
}
set core_clock [get_clocks core_clk]
if {[sizeof_collection $core_clock] != 1} {
    error "M1722 expected exactly one core_clk"
}
set clock_period [get_attribute $core_clock period]
if {[expr {abs($clock_period - 3.000)}] > 0.000001} {
    error "M1722 3 ns clock contract failed"
}

update_timing -full
check_timing -verbose > "$output_dir/reports/check_timing.rpt"
report_analysis_coverage -status_details untested > "$output_dir/reports/analysis_coverage.rpt"
report_global_timing > "$output_dir/reports/global_timing.rpt"
report_timing -delay_type max -slack_lesser_than 1000000 \
    -max_paths 100 -nworst 10 -path_type full_clock_expanded \
    -significant_digits 9 > "$output_dir/reports/timing_setup_slow.rpt"
report_timing -delay_type min -slack_lesser_than 1000000 \
    -max_paths 100 -nworst 10 -path_type full_clock_expanded \
    -significant_digits 9 > "$output_dir/reports/timing_hold_fast.rpt"
report_constraint -all_violators -verbose -significant_digits 9 \
    > "$output_dir/reports/constraint_violators.rpt"
report_clock > "$output_dir/reports/clock.rpt"
report_exceptions -ignored > "$output_dir/reports/exceptions.rpt"
report_design > "$output_dir/reports/design.rpt"
report_wire_load > "$output_dir/reports/wire_load.rpt"
list_libs > "$output_dir/reports/libraries.rpt"

set setup_paths [get_timing_paths -delay_type max -nworst 1 -max_paths 1]
set hold_paths [get_timing_paths -delay_type min -nworst 1 -max_paths 1]
if {[sizeof_collection $setup_paths] != 1 || \
    [sizeof_collection $hold_paths] != 1} {
    error "M1722 missing max/min timing path"
}
set setup_slack [get_attribute $setup_paths slack]
set hold_slack [get_attribute $hold_paths slack]
if {$setup_slack < 0.0 || $hold_slack < 0.0} {
    error "M1722 independent PrimeTime setup/hold gate failed"
}

set scope_fp [open "$output_dir/reports/runtime_scope.rpt" w]
puts $scope_fp "milestone=M1722"
puts $scope_fp "scope=M1701_C1_salvage_macro_aware_prelayout_independent_PrimeTime"
puts $scope_fp "clock_period_ns=3.000"
puts $scope_fp "setup_uncertainty_ns=0.200"
puts $scope_fp "hold_uncertainty_ns=0.050"
puts $scope_fp "setup_view=std_and_macro_slow_ssg0p9v125c"
puts $scope_fp "hold_view=std_and_macro_fast_ffg1p05vm40c"
puts $scope_fp "macro_cell=$macro_cell"
puts $scope_fp "macro_count=$macro_count"
puts $scope_fp "wireload=ZeroWireload_from_exact_M1701_SDC"
puts $scope_fp "parasitics=none_no_read_parasitics_command"
puts $scope_fp "ideal_clock=true"
puts $scope_fp "false_path_or_multicycle_added_by_M1722=false"
puts $scope_fp "pt_eco=false"
close $scope_fp

set machine_fp [open "$output_dir/reports/timing_summary_machine.txt" w]
puts $machine_fp "setup_wns_ns=$setup_slack"
puts $machine_fp "setup_tns_ns=0.0"
puts $machine_fp "setup_violating_paths=0"
puts $machine_fp "hold_wns_ns=$hold_slack"
puts $machine_fp "hold_tns_ns=0.0"
puts $machine_fp "hold_violating_paths=0"
puts $machine_fp "macro_count=$macro_count"
puts $machine_fp "clock_period_ns=3.000"
puts $machine_fp "setup_uncertainty_ns=0.200"
puts $machine_fp "hold_uncertainty_ns=0.050"
close $machine_fp

set marker [open "$output_dir/PTSTA_INTERNAL_COMPLETE.txt" w]
puts $marker "M1722_C1_M1701_PRELAYOUT_PTSTA_INTERNAL_COMPLETE=PASS"
puts $marker "meaning=REPORTS_COMPLETE_AND_NONNEGATIVE_MAX_MIN_NOT_RESULT_ADMISSION"
puts $marker "paper_claim=false"
close $marker
quit
