# Prove that incremental hold repair preserves the mapped reference.
# This is gate-to-gate preservation, not a new RTL-to-gate proof.
set output $::env(M2250_FM_OUTPUT)
file mkdir "$output/reports"
if {$::env(M2250_GATE_CLOCK) eq "1"} {
    # Standard latch-based clock-gating recognition, not a false path or
    # an arithmetic assumption. All data compare points remain in scope.
    set verification_clock_gate_hold_mode low
    if {[info exists ::env(M2255_SVF_LIST)]} {
        foreach path [split $::env(M2255_SVF_LIST) ":"] {set_svf -append $path}
    } else {
        set_svf "$::env(M2250_OUTPUT)/netlist/hold_repair.svf"
    }
}
read_db -technology_library $::env(M2250_FM_LIBRARY)
read_verilog -r $::env(M2250_FM_REFERENCE)
set_top r:/WORK/$::env(M2250_FM_DESIGN)
read_verilog -i $::env(M2250_FM_IMPLEMENTATION)
set_top i:/WORK/$::env(M2250_FM_DESIGN)
if {$::env(M2250_GATE_CLOCK) eq "1"} {
    # A generated ICG test pin is inactive in this functional comparison.
    foreach pattern {*test* *scan_enable*} {
        foreach_in_collection port [get_ports -quiet "i:/WORK/$::env(M2250_FM_DESIGN)/$pattern"] {
            set_constant $port 0
        }
    }
}
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
