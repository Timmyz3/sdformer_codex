set design_name $::env(M441_DESIGN_NAME)
set snapshot_root [file normalize $::env(M441_SNAPSHOT_ROOT)]
set rtl_filelist [file normalize $::env(M441_RTL_FILELIST)]
set lib_db [file normalize $::env(M441_LIB_DB)]
set mapped_netlist [file normalize $::env(M441_MAPPED_NETLIST)]
set svf_file [file normalize $::env(M441_SVF_FILE)]
set output_dir [file normalize $::env(M441_FM_OUTPUT_DIR)]

proc m441_read_filelist {filelist snapshot_root} {
    set fp [open $filelist r]
    set rtl_files {}
    while {[gets $fp line] >= 0} {
        set line [string trim $line]
        if {$line ne "" && ![string match "#*" $line]} {
            lappend rtl_files [file normalize "$snapshot_root/$line"]
        }
    }
    close $fp
    return $rtl_files
}

file mkdir "$output_dir/reports"
set_svf $svf_file
read_db -technology_library $lib_db
set rtl_files [m441_read_filelist $rtl_filelist $snapshot_root]
read_sverilog -r $rtl_files
set_top r:/WORK/$design_name
read_verilog -i $mapped_netlist
set_top i:/WORK/$design_name

match
report_unmatched_points > "$output_dir/reports/formality_unmatched.rpt"
report_unmatched_points -status unread -point_type DFF \
    > "$output_dir/reports/formality_unmatched_unread_dff.rpt"
report_not_compared_points -status unread -point_type DFF \
    > "$output_dir/reports/formality_not_compared_unread_dff.rpt"
report_unread_endpoints -all -point_type DFF \
    > "$output_dir/reports/formality_unread_endpoints_dff.rpt"

set verification_succeeded [verify]
redirect "$output_dir/reports/formality_status.rpt" {
    echo "verify_return=$verification_succeeded"
    report_status
}
report_failing_points > "$output_dir/reports/formality_failing.rpt"
report_aborted_points > "$output_dir/reports/formality_aborted.rpt"
report_unverified_points > "$output_dir/reports/formality_unverified.rpt"
if {!$verification_succeeded} {
    error "M441 M433 RTL to M439 mapped-netlist Formality failed"
}

set marker [open "$output_dir/FORMALITY_INTERNAL_COMPLETE.txt" w]
puts $marker "M441_M433_RTL_TO_M439_NETLIST_FORMALITY_INTERNAL_COMPLETE=PASS"
puts $marker "design=$design_name"
puts $marker "verify_return=$verification_succeeded"
close $marker
quit
