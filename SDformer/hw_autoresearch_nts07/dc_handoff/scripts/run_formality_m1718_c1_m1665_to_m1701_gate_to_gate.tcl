set design_name m935_m912_three_stage_exact_parent_match_product_capture_island
set std_slow_db [file normalize $::env(M1718_STD_SLOW_DB)]
set macro_slow_db [file normalize $::env(M1718_MACRO_SLOW_DB)]
set reference_netlist [file normalize $::env(M1718_M1665_REFERENCE_NETLIST)]
set implementation_netlist [file normalize $::env(M1718_M1701_IMPLEMENTATION_NETLIST)]
set output_dir [file normalize $::env(M1718_FM_OUTPUT_DIR)]

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
    error "M1718 M1665-reference to M1701-hold-fixed gate equivalence failed"
}
set marker [open "$output_dir/FORMALITY_INTERNAL_COMPLETE.txt" w]
puts $marker "M1718_C1_M1665_TO_M1701_GATE_FORMALITY_INTERNAL_COMPLETE=PASS"
puts $marker "meaning=FROZEN_M1665_REFERENCE_GATE_TO_M1701_HOLD_FIXED_GATE"
puts $marker "paper_claim=false"
close $marker
quit
