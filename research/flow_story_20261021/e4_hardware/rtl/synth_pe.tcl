# Both engines use the identical library, clock, I/O and load constraints.
# Run from its own empty result directory, FLOW_TOP binary_plane_pe or parallel_gateadd_pe.
set_host_options -max_cores 2
set probe_top $env(FLOW_TOP)
if {$probe_top ni {binary_plane_pe parallel_gateadd_pe}} {error "BAD_FLOW_TOP"}
set probe_library /opt/tech/tsmc28_2026/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp7t35p140_180a/tcbn28hpcplusbwp7t35p140tt0p9v25c.db
set probe_source [file normalize $env(FLOW_SOURCE)]
if {![file readable $probe_library]} {error "FLOW_LIBRARY_NOT_READABLE"}
set_app_var target_library [list $probe_library]
set_app_var link_library [concat * $target_library]
define_design_lib WORK -path ./work
analyze -format sverilog $probe_source
elaborate $probe_top
current_design $probe_top
link
uniquify
create_clock -name clk -period 4.000 [get_ports clk]
set_clock_uncertainty 0.050 [get_clocks clk]
set_clock_transition 0.050 [get_clocks clk]
set probe_inputs [remove_from_collection [all_inputs] [get_ports clk]]
set_input_delay 0.200 -clock clk $probe_inputs
set_output_delay 0.200 -clock clk [all_outputs]
set_input_transition 0.050 $probe_inputs
set_load 0.010 [all_outputs]
redirect units.rpt {report_units}
redirect before.rpt {check_design}
compile -map_effort high
redirect after.rpt {check_design}
redirect area.rpt {report_area -hierarchy}
redirect timing.rpt {report_timing -delay_type max -max_paths 10 -nets -capacitance -transition_time}
redirect constraints.rpt {report_constraint -all_violators}
redirect references.rpt {report_reference -hierarchy}
redirect qor.rpt {report_qor}
write_file -format verilog -hierarchy -output mapped.v
write_file -format ddc -hierarchy -output mapped.ddc
write_sdc constraints.sdc
set paths [get_timing_paths -delay_type max -max_paths 1]
if {[sizeof_collection $paths] != 1} {error "NO_TIMED_PATH"}
set slack [get_attribute $paths slack]
set f [open area.rpt r];set content [read $f];close $f
if {![regexp {Total cell area:[ \t]+([0-9.eE+-]+)} $content match area]} {error "MISSING_AREA"}
set f [open summary.json w]
puts $f "{\"module\":\"$probe_top\",\"clock_period_ns\":4.0,\"area_library_units\":$area,\"worst_setup_slack_ns\":$slack,\"max_cores\":2,\"SRAM_included\":false,\"FP_scale_or_BN_included\":false,\"whole_accelerator\":false}"
close $f
puts "FLOW_INTEGER_PE_SYNTH_DONE top=$probe_top area=$area slack=$slack"
quit
