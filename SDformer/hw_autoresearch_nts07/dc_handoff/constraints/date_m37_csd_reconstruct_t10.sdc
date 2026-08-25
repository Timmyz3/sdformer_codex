# M37 standalone T10 CSD reconstruction logic-only constraint.
#
# This is deliberately a zero-wire-load, ideal-clock, no-macro contract.  It
# is useful for a reproducible standard-cell logic comparison, but it is not a
# placed-and-routed or paper-PPA timing claim.
set clock_period_ns 3.000
if {[info exists ::env(CLOCK_PERIOD_NS)] && $::env(CLOCK_PERIOD_NS) ne ""} {
    set clock_period_ns $::env(CLOCK_PERIOD_NS)
}

create_clock -name core_clk -period $clock_period_ns \
    -waveform [list 0.000 [expr {$clock_period_ns / 2.0}]] \
    [get_ports clk_core]
set_ideal_network [get_ports clk_core]
set_clock_uncertainty -setup 0.200 [get_clocks core_clk]
set_clock_uncertainty -hold 0.050 [get_clocks core_clk]

# The selected TSMC28 library provides the explicit ZeroWireload model.  Do
# not silently substitute a foundry/default area-based wire-load selection.
set_wire_load_mode top
set_wire_load_model -name ZeroWireload [current_design]

# rst_core is synchronous RTL state and is intentionally timed.  No false or
# multicycle path is admitted for any functional input or output.
set data_inputs [remove_from_collection [all_inputs] [get_ports clk_core]]
set_input_delay 0.250 -clock core_clk $data_inputs
set_input_transition 0.100 $data_inputs
set_output_delay 0.250 -clock core_clk [all_outputs]
set_load 0.010 [all_outputs]
set_max_fanout 32 [current_design]
set_fix_multiple_port_nets -all -buffer_constants
