set design_name $::env(DESIGN_NAME)
set snapshot_root [file normalize $::env(SNAPSHOT_ROOT)]
set reference_filelist [file normalize $::env(REFERENCE_RTL_FILELIST)]
set implementation_filelist [file normalize $::env(IMPLEMENTATION_RTL_FILELIST)]
set output_dir [file normalize $::env(OUTPUT_DIR)]

proc m420_read_filelist {filelist snapshot_root} {
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
set reference_files [m420_read_filelist $reference_filelist $snapshot_root]
set implementation_files [m420_read_filelist $implementation_filelist $snapshot_root]
read_sverilog -r $reference_files
set_top r:/WORK/$design_name
read_sverilog -i $implementation_files
set_top i:/WORK/$design_name
match
report_unmatched_points > "$output_dir/reports/formality_unmatched.rpt"
set verification_succeeded [verify]
redirect "$output_dir/reports/formality_status.rpt" {
    echo "verify_return=$verification_succeeded"
    report_status
}
report_failing_points > "$output_dir/reports/formality_failing.rpt"
report_aborted_points > "$output_dir/reports/formality_aborted.rpt"
report_unverified_points > "$output_dir/reports/formality_unverified.rpt"
if {!$verification_succeeded} {
    error "M420 serial RTL to balanced RTL Formality failed"
}
set marker [open "$output_dir/FORMALITY_INTERNAL_COMPLETE.txt" w]
puts $marker "M420_SERIAL_TO_BALANCED_RTL_FORMALITY_INTERNAL_COMPLETE=PASS"
puts $marker "design=$design_name"
puts $marker "verify_return=$verification_succeeded"
close $marker
quit
