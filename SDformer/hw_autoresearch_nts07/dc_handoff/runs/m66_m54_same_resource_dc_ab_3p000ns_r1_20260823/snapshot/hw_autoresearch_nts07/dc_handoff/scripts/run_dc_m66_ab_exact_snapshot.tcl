set design_name $::env(DESIGN_NAME)
set variant_name $::env(VARIANT_NAME)
set snapshot_root [file normalize $::env(SNAPSHOT_ROOT)]
set rtl_filelist [file normalize $::env(RTL_FILELIST)]
set lib_db [file normalize $::env(LIB_DB)]
set min_lib_db [file normalize $::env(MIN_LIB_DB)]
set sdc_file [file normalize $::env(SDC_FILE)]
set output_dir [file normalize $::env(OUTPUT_DIR)]

file mkdir "$output_dir/reports"
file mkdir "$output_dir/netlist"
set_svf "$output_dir/netlist/${design_name}.svf"
set_app_var search_path [list $snapshot_root [file dirname $lib_db] \
    [file dirname $min_lib_db]]
set_app_var target_library [list $lib_db]
set_app_var link_library [list "*" $lib_db $min_lib_db]
set_app_var verilogout_no_tri true
set_app_var auto_wire_load_selection false

set fp [open $rtl_filelist r]
set rtl_files {}
while {[gets $fp line] >= 0} {
    set line [string trim $line]
    if {$line ne "" && ![string match "#*" $line]} {
        lappend rtl_files [file normalize "$snapshot_root/$line"]
    }
}
close $fp
if {![analyze -format sverilog $rtl_files]} { exit 11 }
if {![elaborate $design_name]} { exit 12 }
current_design $design_name
link
uniquify
set_min_library $lib_db -min_version $min_lib_db
set_operating_conditions $::env(OPERATING_CONDITION)
set_wire_load_mode top
set_wire_load_model -name ZeroWireload [current_design]
source $sdc_file
set_clock_uncertainty -hold 0.100 [get_clocks core_clk]
set_fix_hold [get_clocks core_clk]

redirect "$output_dir/reports/constraint_contract_precompile.rpt" {
    echo "scope=standalone_m54_m66_same_resource_ab"
    echo "variant=$variant_name"
    echo "physical_contract=ZERO_WIRELOAD_IDEAL_CLOCK_NO_SRAM_MACRO"
    echo "setup_corner=ssg0p9v125c"
    echo "hold_corner=ffg1p05vm40c"
    report_design
    report_clocks
}
check_design > "$output_dir/reports/check_design_precompile.rpt"
check_timing > "$output_dir/reports/check_timing_precompile.rpt"
report_resources -hierarchy > "$output_dir/reports/resources_precompile.rpt"
report_reference -hierarchy > "$output_dir/reports/references_precompile.rpt"

compile_ultra
compile_ultra -incremental
compile -incremental_mapping -only_hold_time
set_clock_uncertainty -hold 0.090 [get_clocks core_clk]
update_timing

redirect "$output_dir/reports/constraint_contract_postcompile.rpt" {
    echo "scope=standalone_m54_m66_same_resource_ab"
    echo "variant=$variant_name"
    echo "physical_contract=ZERO_WIRELOAD_IDEAL_CLOCK_NO_SRAM_MACRO"
    echo "setup_corner=ssg0p9v125c"
    echo "hold_corner=ffg1p05vm40c"
    report_design
    report_clocks
}
report_hierarchy > "$output_dir/reports/hierarchy_postcompile.rpt"
report_resources -hierarchy > "$output_dir/reports/resources_postcompile.rpt"
report_reference -hierarchy > "$output_dir/reports/references_postcompile.rpt"
report_qor > "$output_dir/reports/qor.rpt"
report_area -hierarchy > "$output_dir/reports/area.rpt"
report_clocks > "$output_dir/reports/clocks.rpt"
report_port -verbose > "$output_dir/reports/ports.rpt"
report_timing -delay_type max -max_paths 100 -nworst 10 \
    -significant_digits 4 > "$output_dir/reports/timing_setup.rpt"
report_timing -delay_type min -max_paths 100 -nworst 10 \
    -significant_digits 4 > "$output_dir/reports/timing_hold.rpt"
set ab_data_pins [all_registers -data_pins]
set ab_clock_pins [all_registers -clock_pins]
set ab_data_inputs [remove_from_collection [all_inputs] [get_ports clk_core]]
report_timing -delay_type max -from $ab_clock_pins -to $ab_data_pins \
    -max_paths 20 -nworst 5 -significant_digits 4 \
    > "$output_dir/reports/timing_reg2reg_setup.rpt"
report_timing -delay_type max -from $ab_data_inputs -to $ab_data_pins \
    -max_paths 20 -nworst 5 -significant_digits 4 \
    > "$output_dir/reports/timing_input2reg_setup.rpt"
report_timing -delay_type max -from $ab_clock_pins -to [all_outputs] \
    -max_paths 20 -nworst 5 -significant_digits 4 \
    > "$output_dir/reports/timing_reg2out_setup.rpt"
report_timing -delay_type max -from $ab_data_inputs -to [all_outputs] \
    -max_paths 20 -nworst 5 -significant_digits 4 \
    > "$output_dir/reports/timing_input2out_setup.rpt"
redirect "$output_dir/reports/constraint_violators.rpt" {
    report_constraint -max_delay -all_violators -significant_digits 4
    report_constraint -min_delay -all_violators -significant_digits 4
    report_constraint -max_capacitance -all_violators -significant_digits 4
    report_constraint -max_transition -all_violators -significant_digits 4
    report_constraint -max_fanout -all_violators -significant_digits 4
}
check_design > "$output_dir/reports/check_design_postcompile.rpt"
check_timing > "$output_dir/reports/check_timing_postcompile.rpt"

change_names -rules verilog -hierarchy
write_file -format verilog -hierarchy \
    -output "$output_dir/netlist/${design_name}_mapped.v"
write_sdc "$output_dir/netlist/${design_name}_mapped.sdc"
write -format ddc -hierarchy -output "$output_dir/netlist/${design_name}.ddc"
set_svf -off
set marker [open "$output_dir/DC_INTERNAL_COMPLETE.txt" w]
puts $marker "M66_AB_DC_INTERNAL_COMPLETE=PASS"
puts $marker "variant=$variant_name"
puts $marker "design=$design_name"
puts $marker "clock_period_ns=$::env(CLOCK_PERIOD_NS)"
close $marker
quit
