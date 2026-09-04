set design_name $::env(M441_DESIGN_NAME)
set lib_slow [file normalize $::env(M441_LIB_SLOW)]
set lib_fast [file normalize $::env(M441_LIB_FAST)]
set mapped_netlist [file normalize $::env(M441_MAPPED_NETLIST)]
set mapped_sdc [file normalize $::env(M441_MAPPED_SDC)]
set output_dir [file normalize $::env(M441_PT_OUTPUT_DIR)]

set slow_lib_name tcbn28hpcplusbwp35p140ssg0p9v125c
set fast_lib_name tcbn28hpcplusbwp35p140ffg1p05vm40c
set slow_opcond ssg0p9v125c
set fast_opcond ffg1p05vm40c

file mkdir "$output_dir/reports"
set_app_var search_path [list [file dirname $lib_slow] [file dirname $lib_fast]]
set_app_var link_path [list "*" $lib_slow]

read_verilog $mapped_netlist
current_design $design_name
link_design $design_name
set_min_library $lib_slow -min_version $lib_fast
read_sdc $mapped_sdc
# The frozen mapped SDC selects the slow max condition.  Bind the already
# declared fast min library explicitly so the hold report is a true fast-corner
# analysis while preserving every clock, IO delay, load and exception in SDC.
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
puts $scope_fp "scope=standalone_dual_coread_adapter_prelayout_logic_only"
puts $scope_fp "design=$design_name"
puts $scope_fp "parasitics=none_no_read_parasitics_command"
puts $scope_fp "clock=ideal_from_frozen_m439_mapped_sdc"
puts $scope_fp "wireload=ZeroWireload_from_frozen_m439_mapped_sdc"
puts $scope_fp "clock_period_ns=3.0"
puts $scope_fp "setup_library=[file tail $lib_slow]"
puts $scope_fp "setup_operating_condition=$slow_opcond"
puts $scope_fp "hold_library=[file tail $lib_fast]"
puts $scope_fp "hold_operating_condition=$fast_opcond"
puts $scope_fp "macro_count=0"
puts $scope_fp "physical_sram=false"
puts $scope_fp "physical_interconnect=false"
puts $scope_fp "reset_n=false_path_from_frozen_mapped_sdc_not_recovery_removal_signoff"
puts $scope_fp "leaf_cells=[sizeof_collection [get_cells -hierarchical -filter {is_hierarchical == false}]]"
puts $scope_fp "hierarchical_cells=[sizeof_collection [get_cells -quiet -hierarchical -filter {is_hierarchical == true}]]"
close $scope_fp

set marker [open "$output_dir/PTSTA_INTERNAL_COMPLETE.txt" w]
puts $marker "M441_M439_DUALCOREAD_PRELAYOUT_PTSTA_INTERNAL_COMPLETE=PASS"
puts $marker "setup_corner=ssg0p9v125c"
puts $marker "hold_corner=ffg1p05vm40c"
puts $marker "scope=prelayout_no_spef_ideal_clock_zero_macro"
close $marker
quit
