set design_name $::env(DESIGN_NAME)
set lib_db [file normalize $::env(LIB_DB)]
set mapped_netlist [file normalize $::env(MAPPED_NETLIST)]
set mapped_sdc [file normalize $::env(MAPPED_SDC)]
set saif_file [file normalize $::env(SAIF_FILE)]
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
set power_enable_analysis true
set_power_analysis_mode averaged

if {[info exists ::env(SPEF_FILE)] && $::env(SPEF_FILE) ne ""} {
    set spef_file [file normalize $::env(SPEF_FILE)]
    if {![file exists $spef_file]} {
        error "SPEF_FILE does not exist: $spef_file"
    }
    read_parasitics $spef_file
}

set scope_fp [open "$output_dir/reports/ptpx_scope.rpt" w]
puts $scope_fp "operating_condition=$::env(OPERATING_CONDITION)"
puts $scope_fp "corner_role=$::env(CORNER_ROLE)"
puts $scope_fp "netlist=$mapped_netlist"
puts $scope_fp "saif=$saif_file"
close $scope_fp
if {[llength [info commands report_annotated_parasitics]] > 0} {
    report_annotated_parasitics -check \
        > "$output_dir/reports/ptpx_annotated_parasitics.rpt"
} else {
    set parasitic_fp [open "$output_dir/reports/ptpx_annotated_parasitics.rpt" w]
    puts $parasitic_fp "report_annotated_parasitics unavailable"
    close $parasitic_fp
}

if {![info exists ::env(SAIF_INSTANCE)] || $::env(SAIF_INSTANCE) eq ""} {
    error "SAIF_INSTANCE is required"
}
read_saif -strip_path $::env(SAIF_INSTANCE) $saif_file

check_timing > "$output_dir/reports/ptpx_check_timing.rpt"
check_power > "$output_dir/reports/ptpx_check_power.rpt"
update_timing
update_power
report_switching_activity -list_not_annotated \
    > "$output_dir/reports/ptpx_unannotated.rpt"
report_switching_activity > "$output_dir/reports/ptpx_switching_summary.rpt"
report_power -hierarchy > "$output_dir/reports/ptpx_power_hierarchy.rpt"
report_power > "$output_dir/reports/ptpx_power.rpt"
report_timing -delay_type max -max_paths 50 -nworst 5 \
    > "$output_dir/reports/ptpx_timing_setup.rpt"
quit
