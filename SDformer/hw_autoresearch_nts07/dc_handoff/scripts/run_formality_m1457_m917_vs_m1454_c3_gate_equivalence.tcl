set design_name m518_matched_fixed_t10_atlif
set lib_db [file normalize $::env(M1457_LIB_DB)]
set reference_netlist [file normalize $::env(M1457_REFERENCE_NETLIST)]
set implementation_netlist [file normalize $::env(M1457_IMPLEMENTATION_NETLIST)]
set output_dir [file normalize $::env(M1457_OUTPUT_DIR)]

file mkdir "$output_dir/reports"
read_db -technology_library $lib_db
read_verilog -r $reference_netlist
set_top r:/WORK/$design_name
read_verilog -i $implementation_netlist
set_top i:/WORK/$design_name

match
report_unmatched_points > "$output_dir/reports/formality_unmatched.rpt"
set verification_succeeded [verify]
redirect "$output_dir/reports/formality_status.rpt" {
    echo "verify_return=$verification_succeeded"
    report_status
}
if {[llength [info commands report_failing_points]] > 0} {
    report_failing_points > "$output_dir/reports/formality_failing.rpt"
}
if {[llength [info commands report_aborted_points]] > 0} {
    report_aborted_points > "$output_dir/reports/formality_aborted.rpt"
}
if {[llength [info commands report_unverified_points]] > 0} {
    report_unverified_points > "$output_dir/reports/formality_unverified.rpt"
}
if {!$verification_succeeded} {
    error "M1457 M917-to-M1454 C3 gate equivalence failed"
}

set marker [open "$output_dir/FORMALITY_INTERNAL_COMPLETE.txt" w]
puts $marker "M1457_M917_VS_M1454_C3_GATE_EQUIVALENCE_INTERNAL_COMPLETE=PASS"
puts $marker "design=$design_name"
puts $marker "verify_return=$verification_succeeded"
puts $marker "scope=gate_to_gate_hold_repair_preservation_not_direct_RTL_proof"
close $marker
quit
