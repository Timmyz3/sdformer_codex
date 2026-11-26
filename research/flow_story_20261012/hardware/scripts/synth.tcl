set_host_options -max_cores 2
set probe_root [file normalize ../..]
set probe_library /opt/tech/tsmc28_2026/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp7t35p140_180a/tcbn28hpcplusbwp7t35p140tt0p9v25c.db
set lanes $env(FLOW_DOT_LANES)
set_app_var target_library [list $probe_library]
set_app_var link_library [concat * $target_library]
define_design_lib WORK -path ./work
analyze -format sverilog [file join $probe_root rtl dot_residual.sv]
elaborate dot_residual -parameters "LANES=$lanes"
link
uniquify
create_clock -name clk -period 4.000 [get_ports clk]
set_clock_uncertainty 0.050 [get_clocks clk]
set_clock_transition 0.050 [get_clocks clk]
set data_inputs [remove_from_collection [all_inputs] [get_ports clk]]
set_input_delay 0.200 -clock clk $data_inputs
set_output_delay 0.200 -clock clk [all_outputs]
set_input_transition 0.050 $data_inputs
set_load 0.010 [all_outputs]
redirect units.rpt {report_units}
redirect before.rpt {check_design}
compile -map_effort high
redirect after.rpt {check_design}
redirect area.rpt {report_area -hierarchy}
redirect timing.rpt {report_timing -delay_type max -max_paths 10 -nets -capacitance -transition_time}
redirect constraints.rpt {report_constraint -all_violators}
redirect references.rpt {report_reference -hierarchy}
write_file -format verilog -hierarchy -output mapped.v
write_file -format ddc -hierarchy -output mapped.ddc
write_sdc constraints.sdc
set paths [get_timing_paths -delay_type max -max_paths 1]
set slack [get_attribute $paths slack]
set f [open area.rpt r];set content [read $f];close $f
if {![regexp {Total cell area:[ \t]+([0-9.eE+-]+)} $content match area]} {error "missing area"}
set f [open summary.json w]
puts $f "{\"lanes\":$lanes,\"period_ns\":4.0,\"area_library_units\":$area,\"slack_ns\":$slack,\"max_cores\":2}"
close $f
puts "FLOW_DOT_SYNTH_DONE lanes=$lanes area=$area slack=$slack"
quit
