# M931 macro-aware C1/M912 timing point.  Pre-route, ideal-clock and
# ZeroWireload; hold is diagnostic only even though slow/fast views are paired.
create_clock -name core_clk -period 3.000 \
    -waveform {0.000 1.500} [get_ports clk_core]
set_clock_uncertainty -setup 0.200 [get_clocks core_clk]
set_clock_uncertainty -hold 0.050 [get_clocks core_clk]

set data_inputs [remove_from_collection [all_inputs] \
    [get_ports {clk_core reset_n}]]
set_input_delay 0.250 -clock core_clk $data_inputs
set_input_transition 0.100 $data_inputs
set_output_delay 0.250 -clock core_clk [all_outputs]
set_load 0.010 [all_outputs]
set_false_path -from [get_ports reset_n]
set_max_fanout 32 [current_design]
set_fix_multiple_port_nets -all -buffer_constants
