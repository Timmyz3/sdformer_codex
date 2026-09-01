set design_name m935_m912_three_stage_exact_parent_match_product_capture_island
set std_slow_db [file normalize $::env(M1674_STD_SLOW_DB)]
set macro_slow_db [file normalize $::env(M1674_MACRO_SLOW_DB)]
set reference_netlist [file normalize $::env(M1674_M993_MAPPED_NETLIST)]
set implementation_netlist [file normalize $::env(M1674_M1665_MAPPED_NETLIST)]
set output_dir [file normalize $::env(M1674_FM_OUTPUT_DIR)]

file mkdir "$output_dir/reports"
read_db -technology_library $std_slow_db
read_db -technology_library $macro_slow_db
read_verilog -r $reference_netlist
set_top r:/WORK/$design_name
read_verilog -i $implementation_netlist
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
    error "M1674 M993-to-M1665 gate-to-gate Formality failed"
}
set marker [open "$output_dir/FORMALITY_INTERNAL_COMPLETE.txt" w]
puts $marker "M1674_C1_M993_TO_M1665_GATE_FORMALITY_INTERNAL_COMPLETE=PASS"
puts $marker "meaning=ORIGINAL_M993_GATE_TO_RESIDUAL_HOLD_CLOSED_M1665_GATE"
puts $marker "paper_claim=false"
close $marker
quit
