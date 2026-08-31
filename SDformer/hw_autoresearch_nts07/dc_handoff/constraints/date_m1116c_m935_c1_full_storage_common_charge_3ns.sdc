# M1116C 3.000-ns logic-plus-live-parent-macro boundary constraint.
# Psum, weight, and residual metadata/reserve capacity are identical external
# common charges and are not instantiated in this DC top.
# No false path, multicycle path, disabled arc, case analysis, or max/min-delay
# exception is permitted.  reset_n is timed as an ordinary async input.
create_clock -name core_clk -period 3.000 \
    -waveform {0.000 1.500} [get_ports clk_core]
set_clock_uncertainty -setup 0.200 [get_clocks core_clk]
set_clock_uncertainty -hold 0.050 [get_clocks core_clk]

set data_inputs [remove_from_collection [all_inputs] [get_ports clk_core]]
set_input_delay 0.250 -clock core_clk $data_inputs
set_input_transition 0.100 $data_inputs
set_output_delay 0.250 -clock core_clk [all_outputs]
set_load 0.010 [all_outputs]
set_max_fanout 32 [current_design]
set_fix_multiple_port_nets -all -buffer_constants
