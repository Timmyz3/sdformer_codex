set design_name $::env(DESIGN_NAME)
set snapshot_root [file normalize $::env(SNAPSHOT_ROOT)]
set reference_filelist [file normalize $::env(REFERENCE_RTL_FILELIST)]
set implementation_kind $::env(IMPLEMENTATION_KIND)
set output_dir [file normalize $::env(OUTPUT_DIR)]

proc m421_read_filelist {filelist snapshot_root} {
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
set reference_files [m421_read_filelist $reference_filelist $snapshot_root]

if {$implementation_kind eq "rtl"} {
    set implementation_filelist [file normalize $::env(IMPLEMENTATION_RTL_FILELIST)]
    set implementation_files [m421_read_filelist $implementation_filelist $snapshot_root]
    read_sverilog -r $reference_files
    set_top r:/WORK/$design_name
    read_sverilog -i $implementation_files
    set_top i:/WORK/$design_name
} elseif {$implementation_kind eq "netlist"} {
    set lib_db [file normalize $::env(LIB_DB)]
    set mapped_netlist [file normalize $::env(MAPPED_NETLIST)]
    set svf_file [file normalize $::env(SVF_FILE)]
    set_svf $svf_file
    read_db -technology_library $lib_db
    read_sverilog -r $reference_files
    set_top r:/WORK/$design_name
    read_verilog -i $mapped_netlist
    set_top i:/WORK/$design_name
} else {
    error "unsupported IMPLEMENTATION_KIND=$implementation_kind"
}

match
set verification_succeeded [verify]
redirect "$output_dir/reports/formality_status.rpt" {
    echo "verify_return=$verification_succeeded"
    report_status
}
report_unmatched_points -status unread -point_type DFF \
    > "$output_dir/reports/unmatched_unread_dff.rpt"
report_not_compared_points -status unread -point_type DFF \
    > "$output_dir/reports/not_compared_unread_dff.rpt"
report_unread_endpoints -all -point_type DFF \
    > "$output_dir/reports/unread_endpoints_dff.rpt"
report_failing_points > "$output_dir/reports/formality_failing.rpt"
report_aborted_points > "$output_dir/reports/formality_aborted.rpt"
report_unverified_points > "$output_dir/reports/formality_unverified.rpt"
if {!$verification_succeeded} {
    error "M421 independent diagnostic verification failed"
}
set marker [open "$output_dir/FORMALITY_INTERNAL_COMPLETE.txt" w]
puts $marker "M421_INDEPENDENT_UNREAD_DIAGNOSTIC=PASS"
puts $marker "implementation_kind=$implementation_kind"
puts $marker "verify_return=$verification_succeeded"
close $marker
quit
