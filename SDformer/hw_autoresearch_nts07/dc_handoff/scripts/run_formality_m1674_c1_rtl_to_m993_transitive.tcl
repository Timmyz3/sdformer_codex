set design_name m935_m912_three_stage_exact_parent_match_product_capture_island
set snapshot_root [file normalize $::env(M1674_SNAPSHOT_ROOT)]
set rtl_filelist [file normalize $::env(M1674_RTL_FILELIST)]
set std_slow_db [file normalize $::env(M1674_STD_SLOW_DB)]
set macro_slow_db [file normalize $::env(M1674_MACRO_SLOW_DB)]
set mapped_netlist [file normalize $::env(M1674_M993_MAPPED_NETLIST)]
set svf_file [file normalize $::env(M1674_M993_SVF)]
set output_dir [file normalize $::env(M1674_FM_OUTPUT_DIR)]

file mkdir "$output_dir/reports"
set_svf $svf_file
read_db -technology_library $std_slow_db
read_db -technology_library $macro_slow_db

set fp [open $rtl_filelist r]
set rtl_files {}
while {[gets $fp line] >= 0} {
    set line [string trim $line]
    if {$line ne "" && ![string match "#*" $line]} {
        lappend rtl_files [file normalize "$snapshot_root/$line"]
    }
}
close $fp
if {[llength $rtl_files] != 2} {
    error "M1674 expected exactly two frozen RTL sources"
}

read_sverilog -r $rtl_files
set_top r:/WORK/$design_name
read_verilog -i $mapped_netlist
set_top i:/WORK/$design_name
match
report_unmatched_points > "$output_dir/reports/formality_unmatched.rpt"
report_black_boxes > "$output_dir/reports/formality_black_boxes.rpt"
set verification_succeeded [verify]
redirect "$output_dir/reports/formality_status.rpt" {
    echo "verify_return=$verification_succeeded"
    report_status
}
report_failing_points > "$output_dir/reports/formality_failing.rpt"
report_aborted_points > "$output_dir/reports/formality_aborted.rpt"
report_unverified_points > "$output_dir/reports/formality_unverified.rpt"
if {!$verification_succeeded} {
    error "M1674 RTL-to-M993 transitive Formality failed"
}
set marker [open "$output_dir/FORMALITY_INTERNAL_COMPLETE.txt" w]
puts $marker "M1674_C1_RTL_TO_M993_FORMALITY_INTERNAL_COMPLETE=PASS"
puts $marker "meaning=RTL_TO_ORIGINAL_ADMITTED_M993_LINK_OF_TRANSITIVE_PROOF"
puts $marker "paper_claim=false"
close $marker
quit
