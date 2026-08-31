set design_name m518_matched_fixed_t10_atlif
set slow_db [file normalize $::env(M1456_SLOW_DB)]
set fast_db [file normalize $::env(M1456_FAST_DB)]
set mapped_netlist [file normalize $::env(M1456_MAPPED_NETLIST)]
set mapped_sdc [file normalize $::env(M1456_MAPPED_SDC)]
set output_dir [file normalize $::env(M1456_PT_OUTPUT_DIR)]

set slow_lib_name tcbn28hpcplusbwp35p140ssg0p9v125c
set fast_lib_name tcbn28hpcplusbwp35p140ffg1p05vm40c
set slow_opcond ssg0p9v125c
set fast_opcond ffg1p05vm40c

file mkdir "$output_dir/reports"
set_app_var search_path [list [file dirname $slow_db] [file dirname $fast_db]]
set_app_var link_path [list "*" $slow_db]

read_verilog $mapped_netlist
current_design $design_name
link_design $design_name
set_min_library $slow_db -min_version $fast_db
read_sdc $mapped_sdc
set_operating_conditions -analysis_type on_chip_variation \
    -max $slow_opcond -max_library $slow_lib_name \
    -min $fast_opcond -min_library $fast_lib_name

update_timing -full
check_timing -verbose > "$output_dir/reports/check_timing.rpt"
report_analysis_coverage -status_details untested \
    > "$output_dir/reports/analysis_coverage.rpt"
report_global_timing > "$output_dir/reports/global_timing.rpt"
report_timing -delay_type max -slack_lesser_than 1000000 \
    -max_paths 100 -nworst 10 -path_type full_clock_expanded \
    -significant_digits 6 > "$output_dir/reports/timing_setup_slow.rpt"
report_timing -delay_type min -slack_lesser_than 1000000 \
    -max_paths 100 -nworst 10 -path_type full_clock_expanded \
    -significant_digits 6 > "$output_dir/reports/timing_hold_fast.rpt"
report_constraint -all_violators -verbose -significant_digits 6 \
    > "$output_dir/reports/constraint_violators.rpt"
report_clock > "$output_dir/reports/clock.rpt"
report_exceptions -ignored > "$output_dir/reports/exceptions.rpt"
report_design > "$output_dir/reports/design.rpt"
report_wire_load > "$output_dir/reports/wire_load.rpt"
list_libs > "$output_dir/reports/libraries.rpt"

set scope_fp [open "$output_dir/reports/runtime_scope.rpt" w]
puts $scope_fp "milestone=M1456"
puts $scope_fp "scope=M1454_C3_Fixed_T10_hold-repaired_exact_mapped_netlist_prelayout_logic_only"
puts $scope_fp "design=$design_name"
puts $scope_fp "mapped_netlist=$mapped_netlist"
puts $scope_fp "mapped_sdc=$mapped_sdc"
puts $scope_fp "parasitics=none_no_read_parasitics_command"
puts $scope_fp "clock=ideal_from_M1454_mapped_sdc"
puts $scope_fp "wireload=ZeroWireload_from_M1454_mapped_sdc"
puts $scope_fp "clock_period_ns=3.0"
puts $scope_fp "setup_library=$slow_lib_name"
puts $scope_fp "setup_operating_condition=$slow_opcond"
puts $scope_fp "hold_library=$fast_lib_name"
puts $scope_fp "hold_operating_condition=$fast_opcond"
puts $scope_fp "macro_count=0"
puts $scope_fp "physical_sram=false"
puts $scope_fp "physical_interconnect=false"
puts $scope_fp "hold_fix_command_count=0"
puts $scope_fp "mapped_identity_mutated=false"
close $scope_fp

set marker [open "$output_dir/PTSTA_INTERNAL_COMPLETE.txt" w]
puts $marker "M1456_M1454_C3_FIXED_T10_PRELAYOUT_PTSTA_INTERNAL_COMPLETE=PASS"
puts $marker "meaning=REPORTS_COMPLETE_NOT_RESULT_ADMISSION"
puts $marker "scope=prelayout_no_spef_ideal_clock_zero_macro"
close $marker
quit
