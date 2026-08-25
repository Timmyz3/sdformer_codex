set design_name $::env(DESIGN_NAME)
set lib_db [file normalize $::env(LIB_DB)]
set min_lib_db [file normalize $::env(MIN_LIB_DB)]
set ddc_file [file normalize $::env(DDC_FILE)]
set output_dir [file normalize $::env(OUTPUT_DIR)]

set_app_var target_library [list $lib_db]
set_app_var link_library [list "*" $lib_db $min_lib_db]
read_ddc $ddc_file
current_design $design_name
link
set_min_library $lib_db -min_version $min_lib_db
set_operating_conditions $::env(OPERATING_CONDITION)
set_clock_uncertainty -hold $::env(DC_HOLD_REPORT_UNCERTAINTY_NS) \
    [get_clocks core_clk]
update_timing

write_sdc "$output_dir/netlist/${design_name}_mapped.sdc"
report_timing -delay_type max -max_paths 50 -nworst 5 -significant_digits 4 \
    > "$output_dir/reports/timing_setup.rpt"
report_timing -delay_type min -max_paths 50 -nworst 5 -significant_digits 4 \
    > "$output_dir/reports/timing_hold.rpt"
redirect "$output_dir/reports/constraint_violators.rpt" {
    report_constraint -max_delay -all_violators -significant_digits 4
    report_constraint -min_delay -all_violators -significant_digits 4
    report_constraint -max_capacitance -all_violators -significant_digits 4
    report_constraint -max_transition -all_violators -significant_digits 4
    report_constraint -max_fanout -all_violators -significant_digits 4
    report_constraint -min_pulse_width -all_violators -significant_digits 4
    report_constraint -min_period -all_violators -significant_digits 4
}
check_design > "$output_dir/reports/check_design_postcompile.rpt"
check_timing > "$output_dir/reports/check_timing_postcompile.rpt"
quit
