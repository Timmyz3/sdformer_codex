set design_name $::env(M424_DESIGN_NAME)
set lib_slow [file normalize $::env(M424_LIB_SLOW)]
set lib_fast [file normalize $::env(M424_LIB_FAST)]
set mapped_netlist [file normalize $::env(M424_MAPPED_NETLIST)]
set mapped_sdc [file normalize $::env(M424_MAPPED_SDC)]
set output_dir [file normalize $::env(M424_OUTPUT_DIR)]

file mkdir "$output_dir/reports"
set_app_var search_path [list [file dirname $lib_slow] [file dirname $lib_fast]]
set_app_var link_path [list "*" $lib_slow]

read_verilog $mapped_netlist
current_design $design_name
link_design $design_name
set_min_library $lib_slow -min_version $lib_fast
read_sdc $mapped_sdc
set_false_path -from [get_ports reset_n]

update_timing -full
check_timing -verbose > "$output_dir/reports/check_timing_independent.rpt"
report_analysis_coverage -status_details untested \
    > "$output_dir/reports/analysis_coverage_independent.rpt"
report_global_timing > "$output_dir/reports/global_timing_independent.rpt"
report_timing -delay_type max -slack_lesser_than 1000000 \
    -max_paths 1 -nworst 1 -path_type full_clock_expanded \
    -significant_digits 6 > "$output_dir/reports/worst_setup_independent.rpt"
report_timing -delay_type min -slack_lesser_than 1000000 \
    -max_paths 1 -nworst 1 -path_type full_clock_expanded \
    -significant_digits 6 > "$output_dir/reports/worst_hold_independent.rpt"
report_constraint -all_violators -verbose -significant_digits 6 \
    > "$output_dir/reports/constraint_violators_independent.rpt"
report_clock > "$output_dir/reports/clock_independent.rpt"
report_exceptions -ignored > "$output_dir/reports/exceptions_independent.rpt"
report_design > "$output_dir/reports/design_independent.rpt"
list_libs > "$output_dir/reports/libraries_independent.rpt"

set audit_fp [open "$output_dir/reports/runtime_scope_independent.rpt" w]
puts $audit_fp "m424_scope=prelayout_logic_only_independent_reproduction"
puts $audit_fp "m424_parasitics=none_no_read_parasitics_command"
puts $audit_fp "m424_clock=ideal_from_frozen_mapped_sdc"
puts $audit_fp "m424_wireload=ZeroWireload_from_frozen_mapped_sdc"
puts $audit_fp "m424_setup_library=[file tail $lib_slow]"
puts $audit_fp "m424_hold_library=[file tail $lib_fast]"
puts $audit_fp "m424_reset_n=false_path_not_recovery_removal_signoff"
puts $audit_fp "m424_leaf_cells=[sizeof_collection [get_cells -hierarchical -filter {is_hierarchical == false}]]"
puts $audit_fp "m424_hierarchical_cells=[sizeof_collection [get_cells -quiet -hierarchical -filter {is_hierarchical == true}]]"
close $audit_fp

set marker [open "$output_dir/PTSTA_INDEPENDENT_INTERNAL_COMPLETE.txt" w]
puts $marker "M424_M422_PTSTA_INDEPENDENT_INTERNAL_COMPLETE=PASS"
close $marker
quit
