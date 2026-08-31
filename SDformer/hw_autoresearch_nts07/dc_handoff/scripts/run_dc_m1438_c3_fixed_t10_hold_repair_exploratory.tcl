# M1438 exploratory netlist-only hold repair for the frozen M917 Fixed-T10.
# This creates a new mapped identity; it never overwrites M917 and is not
# paper-citable until Formality, independent PT and a result hammer pass.
set input_ddc [file normalize $::env(M1438_INPUT_DDC)]
set input_sdc [file normalize $::env(M1438_INPUT_SDC)]
set slow_db [file normalize $::env(M1438_SLOW_DB)]
set fast_db [file normalize $::env(M1438_FAST_DB)]
set output_dir [file normalize $::env(M1438_OUTPUT_DIR)]

file mkdir $output_dir
file mkdir "$output_dir/reports"
file mkdir "$output_dir/netlist"
set_app_var search_path [list [file dirname $slow_db] [file dirname $fast_db]]
set_app_var target_library [list $slow_db]
set_app_var link_library [list "*" $slow_db $fast_db]
set_app_var verilogout_no_tri true

read_ddc $input_ddc
set design_name [current_design]
if {$design_name eq ""} {
    set fp [open "$output_dir/TCL_EXPLICIT_FAILURE.txt" w]
    puts $fp "status=FAIL_NO_CURRENT_DESIGN_AFTER_DDC"
    close $fp
    exit 35
}
link
set_min_library $slow_db -min_version $fast_db
read_sdc $input_sdc
set_wire_load_model -name ZeroWireload [current_design]

# Give DC a deliberate 100-ps hold target while preserving the functional RTL.
# The reporting SDC is restored to zero additional hold uncertainty afterward;
# an independent PT slow/max + fast/min run decides whether the new netlist met.
set_clock_uncertainty -hold 0.100 [get_clocks core_clk]
set_fix_hold [get_clocks core_clk]
compile_ultra -incremental -no_autoungroup
compile -incremental_mapping -only_hold_time
compile -incremental_mapping
set_clock_uncertainty -hold 0.000 [get_clocks core_clk]
update_timing

report_qor > "$output_dir/reports/qor.rpt"
report_area -hierarchy > "$output_dir/reports/area.rpt"
report_timing -delay_type max -max_paths 100 -nworst 10 \
    -significant_digits 6 > "$output_dir/reports/timing_setup_slow_dc.rpt"
report_timing -delay_type min -max_paths 100 -nworst 10 \
    -significant_digits 6 > "$output_dir/reports/timing_hold_fast_dc.rpt"
redirect "$output_dir/reports/constraint_violators.rpt" {
    report_constraint -all_violators -significant_digits 6
}
check_design > "$output_dir/reports/check_design.rpt"
check_timing > "$output_dir/reports/check_timing.rpt"

change_names -rules verilog -hierarchy
write_file -format verilog -hierarchy \
    -output "$output_dir/netlist/${design_name}_hold_repaired_mapped.v"
write_sdc "$output_dir/netlist/${design_name}_hold_repaired_mapped.sdc"
write -format ddc -hierarchy \
    -output "$output_dir/netlist/${design_name}_hold_repaired.ddc"

set fp [open "$output_dir/M1438_INTERNAL_COMPLETE.txt" w]
puts $fp "status=M1438_EXPLORATORY_INTERNAL_COMPLETE__FORMALITY_PT_HAMMER_REQUIRED"
puts $fp "design=$design_name"
puts $fp "hold_target_ns=0.100"
puts $fp "functional_rtl_modified=false"
puts $fp "mapped_identity_modified=true"
puts $fp "paper_citable=false"
close $fp
quit
