set design_name $::env(DESIGN_NAME)
set snapshot_root [file normalize $::env(SNAPSHOT_ROOT)]
set rtl_filelist [file normalize $::env(RTL_FILELIST)]
set lib_db [file normalize $::env(LIB_DB)]
set mapped_netlist [file normalize $::env(MAPPED_NETLIST)]
set svf_file [file normalize $::env(SVF_FILE)]
set output_dir [file normalize $::env(OUTPUT_DIR)]

file mkdir "$output_dir/reports"
set_svf $svf_file
read_db -technology_library $lib_db
set fp [open $rtl_filelist r]
set rtl_files {}
while {[gets $fp line] >= 0} {
    set line [string trim $line]
    if {$line ne "" && ![string match "#*" $line]} {
        lappend rtl_files [file normalize "$snapshot_root/$line"]
    }
}
close $fp

read_sverilog -r $rtl_files
set_top r:/WORK/$design_name
read_verilog -i $mapped_netlist
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
    error "M389 M384 Formality failed against M387 r1b netlist"
}
set marker [open "$output_dir/FORMALITY_INTERNAL_COMPLETE.txt" w]
puts $marker "M389_M384_FORMALITY_INTERNAL_COMPLETE=PASS"
puts $marker "design=$design_name"
puts $marker "verify_return=$verification_succeeded"
close $marker
quit
