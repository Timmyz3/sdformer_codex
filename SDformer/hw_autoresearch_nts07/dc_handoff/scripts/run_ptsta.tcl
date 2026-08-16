set design_name $::env(DESIGN_NAME)
set lib_db [file normalize $::env(LIB_DB)]
set mapped_netlist [file normalize $::env(MAPPED_NETLIST)]
set mapped_sdc [file normalize $::env(MAPPED_SDC)]
set output_dir [file normalize $::env(OUTPUT_DIR)]

file mkdir "$output_dir/reports"
set search_paths [list [file dirname $lib_db]]
set macro_dbs {}
if {[info exists ::env(MACRO_DBS)] && $::env(MACRO_DBS) ne ""} {
    foreach macro_db [split $::env(MACRO_DBS) ":"] {
        set normalized_macro_db [file normalize $macro_db]
        lappend macro_dbs $normalized_macro_db
        lappend search_paths [file dirname $normalized_macro_db]
    }
}
set_app_var search_path $search_paths
set_app_var link_path [concat [list "*" $lib_db] $macro_dbs]

read_verilog $mapped_netlist
current_design $design_name
link_design $design_name
if {![info exists ::env(OPERATING_CONDITION)] || $::env(OPERATING_CONDITION) eq ""} {
    error "OPERATING_CONDITION is required"
}
set_operating_conditions $::env(OPERATING_CONDITION)
read_sdc $mapped_sdc

set analysis_scope "prelayout_no_spef"
if {[info exists ::env(SPEF_FILE)] && $::env(SPEF_FILE) ne ""} {
    set spef_file [file normalize $::env(SPEF_FILE)]
    if {![file exists $spef_file]} {
        error "SPEF_FILE does not exist: $spef_file"
    }
    read_parasitics $spef_file
    set analysis_scope "extracted_spef"
}

set scope_fp [open "$output_dir/reports/ptsta_scope.rpt" w]
puts $scope_fp $analysis_scope
puts $scope_fp "operating_condition=$::env(OPERATING_CONDITION)"
puts $scope_fp "corner_role=$::env(CORNER_ROLE)"
puts $scope_fp "netlist=$mapped_netlist"
close $scope_fp
if {[llength [info commands report_annotated_parasitics]] > 0} {
    report_annotated_parasitics -check \
        > "$output_dir/reports/ptsta_annotated_parasitics.rpt"
} else {
    set parasitic_fp [open "$output_dir/reports/ptsta_annotated_parasitics.rpt" w]
    puts $parasitic_fp "report_annotated_parasitics unavailable"
    close $parasitic_fp
}
check_timing -verbose > "$output_dir/reports/ptsta_check_timing.rpt"
update_timing -full
report_analysis_coverage -status_details untested \
    > "$output_dir/reports/ptsta_analysis_coverage.rpt"
report_global_timing > "$output_dir/reports/ptsta_global_timing.rpt"
report_timing -delay_type max -max_paths 100 -nworst 10 -path_type full_clock_expanded \
    > "$output_dir/reports/ptsta_timing_setup.rpt"
report_timing -delay_type min -max_paths 100 -nworst 10 -path_type full_clock_expanded \
    > "$output_dir/reports/ptsta_timing_hold.rpt"
report_constraint -all_violators -verbose \
    > "$output_dir/reports/ptsta_constraint_violators.rpt"
write_sdf "$output_dir/netlist/${design_name}.sdf"
quit
