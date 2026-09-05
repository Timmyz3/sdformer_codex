# Prove that incremental hold repair preserves the mapped reference.
# This is gate-to-gate preservation, not a new RTL-to-gate proof.
set output $::env(M2250_FM_OUTPUT)
file mkdir "$output/reports"
read_db -technology_library $::env(M2250_FM_LIBRARY)
read_verilog -r $::env(M2250_FM_REFERENCE)
set_top r:/WORK/$::env(M2250_FM_DESIGN)
read_verilog -i $::env(M2250_FM_IMPLEMENTATION)
set_top i:/WORK/$::env(M2250_FM_DESIGN)
match
redirect "$output/reports/unmatched.rpt" {report_unmatched_points}
set ok [verify]
redirect "$output/reports/status.rpt" {report_status}
redirect "$output/reports/failing.rpt" {report_failing_points}
redirect "$output/reports/aborted.rpt" {report_aborted_points}
if {!$ok} {error "Mapped hold-repair equivalence did not pass"}
set fp [open "$output/PASS.txt" w]
puts $fp "Mapped-to-mapped hold repair preservation PASS"
close $fp
quit
