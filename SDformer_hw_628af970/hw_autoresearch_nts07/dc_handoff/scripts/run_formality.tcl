set design_name $::env(DESIGN_NAME)
set hw_root [file normalize $::env(HW_ROOT)]
set rtl_filelist [file normalize $::env(RTL_FILELIST)]
set lib_db [file normalize $::env(LIB_DB)]
set mapped_netlist [file normalize $::env(MAPPED_NETLIST)]
set svf_file [file normalize $::env(SVF_FILE)]
set output_dir [file normalize $::env(OUTPUT_DIR)]
set implementation_top $::env(IMPLEMENTATION_TOP)

file mkdir "$output_dir/reports"
set_svf $svf_file
read_db -technology_library $lib_db
if {[info exists ::env(MACRO_DBS)] && $::env(MACRO_DBS) ne ""} {
    foreach macro_db [split $::env(MACRO_DBS) ":"] {
        read_db -technology_library [file normalize $macro_db]
    }
}

set fp [open $rtl_filelist r]
set rtl_files {}
while {[gets $fp line] >= 0} {
    set line [string trim $line]
    if {$line ne "" && ![string match "#*" $line]} {
        lappend rtl_files [file normalize "$hw_root/$line"]
    }
}
close $fp

read_sverilog -r $rtl_files
# The descriptor engines use a protocol-bounded 4-bit index for a 12-entry
# array. Formality otherwise treats the unreachable values 12..15 as a fatal
# interpretation diagnostic before equivalence can run. Keep that descriptor-
# only diagnostic visible as a warning and let compare-point verification prove
# the netlist. M31 uses statically bounded phase slices and must never rely on
# this filter.
if {$design_name eq "qfit_dual_line_descriptor_resident_engine"
        || $design_name eq "qfit_dual_line_descriptor_stateful_engine"} {
    set_mismatch_message_filter -warn FMR_ELAB-147
}
if {[info exists ::env(ELAB_PARAMETERS)] && $::env(ELAB_PARAMETERS) ne ""} {
    set_top r:/WORK/$design_name -parameter $::env(ELAB_PARAMETERS)
} else {
    set_top r:/WORK/$design_name
}
read_verilog -i $mapped_netlist
set_top i:/WORK/$implementation_top
match
report_unmatched_points > "$output_dir/reports/formality_unmatched.rpt"
set verification_succeeded [verify]
if {[llength [info commands report_failing_points]] > 0} {
    report_failing_points > "$output_dir/reports/formality_verify.rpt"
} else {
    redirect "$output_dir/reports/formality_verify.rpt" {
        echo "verify returned $verification_succeeded"
        report_status
    }
}
set status_fp [open "$output_dir/reports/formality_status.txt" w]
if {$verification_succeeded} {
    puts $status_fp "PASS"
} else {
    puts $status_fp "FAIL"
}
close $status_fp
if {!$verification_succeeded} {
    error "Formality verification failed"
}
quit
