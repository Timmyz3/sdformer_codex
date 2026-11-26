# Run only in this probe's results/dc_dynamic or results/dc_forced directory.
set_host_options -max_cores 2
set probe_root [file normalize ../..]
set probe_library /opt/tech/tsmc28_2026/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp7t35p140_180a/tcbn28hpcplusbwp7t35p140tt0p9v25c.db
if {![file readable $probe_library]} {error "PROBE_LIBRARY_NOT_READABLE"}
set force 0
if {[info exists env(FLOW_ISSUE_FORCE_FULL)]} {set force $env(FLOW_ISSUE_FORCE_FULL)}
if {$force != 0 && $force != 1} {error "INVALID_FORCE_PARAMETER"}
set_app_var target_library [list $probe_library]
set_app_var link_library [concat * $target_library]
define_design_lib WORK -path ./work
analyze -format sverilog [file join $probe_root rtl risk_group_issue.sv]
elaborate risk_group_issue -parameters "FORCE_FULL=$force"
link
uniquify
create_clock -name clk -period 2.000 [get_ports clk]
set_clock_uncertainty 0.050 [get_clocks clk]
set_clock_transition 0.050 [get_clocks clk]
set probe_inputs [remove_from_collection [all_inputs] [get_ports clk]]
set_input_delay 0.200 -clock clk $probe_inputs
set_output_delay 0.200 -clock clk [all_outputs]
set_input_transition 0.050 $probe_inputs
set_load 0.010 [all_outputs]
redirect units.rpt {report_units}
redirect check_before.rpt {check_design}
compile -map_effort high
redirect check_after.rpt {check_design}
redirect area.rpt {report_area -hierarchy}
redirect timing.rpt {report_timing -delay_type max -max_paths 10 -nets -capacitance -transition_time}
redirect constraints.rpt {report_constraint -all_violators}
redirect qor.rpt {report_qor}
redirect references.rpt {report_reference -hierarchy}
write_file -format verilog -hierarchy -output mapped.v
write_file -format ddc -hierarchy -output mapped.ddc
write_sdc constraints.sdc
set worst_path [get_timing_paths -delay_type max -max_paths 1]
if {[sizeof_collection $worst_path] != 1} {error "NO_TIMED_PATH"}
set worst_slack [get_attribute $worst_path slack]
set area_file [open area.rpt r]
set area_text [read $area_file]
close $area_file
if {![regexp {Total cell area:[ \t]+([0-9.eE+-]+)} $area_text matched design_area]} {
    error "NO_REPORTED_TOTAL_CELL_AREA"
}
set summary [open summary.json w]
puts $summary "{"
puts $summary "  \"forced_full\": $force,"
puts $summary "  \"clock_period_ns\": 2.0,"
puts $summary "  \"clock_uncertainty_ns\": 0.05,"
puts $summary "  \"io_delay_ns\": 0.2,"
puts $summary "  \"input_transition_ns\": 0.05,"
puts $summary "  \"output_load_library_units\": 0.01,"
puts $summary "  \"total_cell_area_library_units\": $design_area,"
puts $summary "  \"worst_setup_slack_ns\": $worst_slack,"
puts $summary "  \"max_cores\": 2"
puts $summary "}"
close $summary
puts "FLOW_ISSUE_SYNTH_DONE force=$force area=$design_area slack=$worst_slack"
quit
