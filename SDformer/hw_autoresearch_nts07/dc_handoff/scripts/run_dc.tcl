set design_name $::env(DESIGN_NAME)
set hw_root [file normalize $::env(HW_ROOT)]
set rtl_filelist [file normalize $::env(RTL_FILELIST)]
set lib_db [file normalize $::env(LIB_DB)]
set sdc_file [file normalize $::env(SDC_FILE)]
set output_dir [file normalize $::env(OUTPUT_DIR)]

file mkdir $output_dir
file mkdir "$output_dir/reports"
file mkdir "$output_dir/netlist"
set_svf "$output_dir/netlist/${design_name}.svf"

set search_paths [list $hw_root [file dirname $lib_db]]
set_app_var target_library [list $lib_db]
set macro_dbs {}
if {[info exists ::env(MACRO_DBS)] && $::env(MACRO_DBS) ne ""} {
    foreach macro_db [split $::env(MACRO_DBS) ":"] {
        set normalized_macro_db [file normalize $macro_db]
        lappend macro_dbs $normalized_macro_db
        lappend search_paths [file dirname $normalized_macro_db]
    }
}
set_app_var search_path $search_paths
set_app_var link_library [concat [list "*" $lib_db] $macro_dbs]
set_app_var verilogout_no_tri true

set fp [open $rtl_filelist r]
set rtl_files {}
while {[gets $fp line] >= 0} {
    set line [string trim $line]
    if {$line ne "" && ![string match "#*" $line]} {
        lappend rtl_files [file normalize "$hw_root/$line"]
    }
}
close $fp

analyze -format sverilog $rtl_files
elaborate $design_name
current_design $design_name
link
uniquify

if {[info exists ::env(OPERATING_CONDITION)] && $::env(OPERATING_CONDITION) ne ""} {
    set_operating_conditions $::env(OPERATING_CONDITION)
}

source $sdc_file
if {[info exists ::env(SAIF_FILE)] && $::env(SAIF_FILE) ne ""} {
    set saif_file [file normalize $::env(SAIF_FILE)]
    if {![file exists $saif_file]} {
        error "SAIF_FILE不存在: $saif_file"
    }
    if {![info exists ::env(SAIF_INSTANCE)] || $::env(SAIF_INSTANCE) eq ""} {
        error "提供SAIF_FILE时必须同时提供SAIF_INSTANCE"
    }
    read_saif -input $saif_file -instance_name $::env(SAIF_INSTANCE)
}
set power_scope_fp [open "$output_dir/reports/power_scope.rpt" w]
if {[info exists ::env(SAIF_FILE)] && $::env(SAIF_FILE) ne ""} {
    puts $power_scope_fp "scope=SAIF_ANNOTATED_EXPLORATORY"
    puts $power_scope_fp "saif=$saif_file"
    puts $power_scope_fp "instance=$::env(SAIF_INSTANCE)"
} else {
    puts $power_scope_fp "scope=NO_SAIF_POWER_NOT_RUN"
}
close $power_scope_fp
check_design > "$output_dir/reports/check_design.rpt"
check_timing > "$output_dir/reports/check_timing_precompile.rpt"
report_clocks > "$output_dir/reports/clocks.rpt"
report_port -verbose > "$output_dir/reports/ports.rpt"

compile_ultra -no_autoungroup
compile_ultra -incremental -no_autoungroup

change_names -rules verilog -hierarchy
write_file -format verilog -hierarchy -output "$output_dir/netlist/${design_name}_mapped.v"
write_sdc "$output_dir/netlist/${design_name}_mapped.sdc"
write -format ddc -hierarchy -output "$output_dir/netlist/${design_name}.ddc"

report_qor > "$output_dir/reports/qor.rpt"
report_area -hierarchy > "$output_dir/reports/area.rpt"
if {[info exists ::env(SAIF_FILE)] && $::env(SAIF_FILE) ne ""} {
    report_power -analysis_effort high > "$output_dir/reports/power.rpt"
    report_power -hierarchy -analysis_effort high \
        > "$output_dir/reports/power_hierarchy.rpt"
}
report_resources -hierarchy > "$output_dir/reports/resources.rpt"
report_reference -hierarchy > "$output_dir/reports/references.rpt"
report_timing -delay_type max -max_paths 50 -nworst 5 > "$output_dir/reports/timing_setup.rpt"
report_timing -delay_type min -max_paths 50 -nworst 5 > "$output_dir/reports/timing_hold.rpt"
report_timing -unconstrained -max_paths 100 > "$output_dir/reports/timing_unconstrained.rpt"
report_constraint -all_violators > "$output_dir/reports/constraint_violators.rpt"
if {[llength [info commands report_clock_gating]] > 0} {
    report_clock_gating -multi_stage -verbose \
        > "$output_dir/reports/clock_gating.rpt"
} else {
    set clock_gating_fp [open "$output_dir/reports/clock_gating.rpt" w]
    puts $clock_gating_fp "report_clock_gating is unavailable in this DC version"
    close $clock_gating_fp
}
check_design > "$output_dir/reports/check_design_postcompile.rpt"
check_timing > "$output_dir/reports/check_timing_postcompile.rpt"
set_svf -off
quit
