set design_name $::env(DESIGN_NAME)
set lib_slow [file normalize $::env(LIB_SLOW)]
set lib_fast [file normalize $::env(LIB_FAST)]
set mapped_netlist [file normalize $::env(MAPPED_NETLIST)]
set mapped_sdc [file normalize $::env(MAPPED_SDC)]
set output_dir [file normalize $::env(OUTPUT_DIR)]

file mkdir "$output_dir/reports"
set_app_var search_path [list [file dirname $lib_slow] [file dirname $lib_fast]]
set_app_var link_path [list "*" $lib_slow]

read_verilog $mapped_netlist
current_design $design_name
link_design $design_name
set_min_library $lib_slow -min_version $lib_fast
read_sdc $mapped_sdc
set_false_path -from [get_ports reset_n]

set scope_fp [open "$output_dir/reports/ptsta_scope.rpt" w]
puts $scope_fp "prelayout_no_spef"
puts $scope_fp "setup_corner=ssg0p9v125c"
puts $scope_fp "hold_corner=ffg1p05vm40c_via_set_min_library"
puts $scope_fp "clock_network=ideal_from_mapped_sdc"
puts $scope_fp "wireload=ZeroWireload_from_mapped_sdc"
puts $scope_fp "macros=0"
puts $scope_fp "async_reset_false_path=reset_n"
puts $scope_fp "physical_descriptor_sram=false"
puts $scope_fp "physical_pwp_sram=false"
close $scope_fp

check_timing -verbose > "$output_dir/reports/ptsta_check_timing.rpt"
update_timing -full
report_analysis_coverage -status_details untested \
    > "$output_dir/reports/ptsta_analysis_coverage.rpt"
report_global_timing > "$output_dir/reports/ptsta_global_timing.rpt"
report_timing -delay_type max -slack_lesser_than 1000000 \
    -max_paths 100 -nworst 10 -path_type full_clock_expanded \
    -significant_digits 4 > "$output_dir/reports/ptsta_timing_setup.rpt"
report_timing -delay_type min -slack_lesser_than 1000000 \
    -max_paths 100 -nworst 10 -path_type full_clock_expanded \
    -significant_digits 4 > "$output_dir/reports/ptsta_timing_hold.rpt"
report_constraint -all_violators -verbose -significant_digits 4 \
    > "$output_dir/reports/ptsta_constraint_violators.rpt"
report_clock > "$output_dir/reports/ptsta_clock.rpt"
report_exceptions -ignored > "$output_dir/reports/ptsta_exceptions.rpt"

set marker [open "$output_dir/PTSTA_INTERNAL_COMPLETE.txt" w]
puts $marker "M391_M384_PTSTA_INTERNAL_COMPLETE=PASS"
puts $marker "scope=prelayout_no_spef"
puts $marker "setup_corner=ssg0p9v125c"
puts $marker "hold_corner=ffg1p05vm40c_via_set_min_library"
close $marker
quit
