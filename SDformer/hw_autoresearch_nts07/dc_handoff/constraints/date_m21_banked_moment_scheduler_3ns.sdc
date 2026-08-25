# Standalone 3 ns logic-only contract for M21.  Raw-moment register banks and
# the four-entry 96x32 payload FIFO are inferred flops, not paper-PPA macros.
set clock_period_ns 3.000
if {[info exists ::env(CLOCK_PERIOD_NS)] && $::env(CLOCK_PERIOD_NS) ne ""} {
    set clock_period_ns $::env(CLOCK_PERIOD_NS)
}

create_clock -name core_clk -period $clock_period_ns \
    -waveform [list 0.000 [expr {$clock_period_ns / 2.0}]] \
    [get_ports clk_core]
set_clock_uncertainty -setup 0.200 [get_clocks core_clk]
set_clock_uncertainty -hold 0.050 [get_clocks core_clk]

set data_inputs [remove_from_collection [all_inputs] [get_ports clk_core]]
set_input_delay 0.250 -clock core_clk $data_inputs
set_input_transition 0.100 $data_inputs
set_output_delay 0.250 -clock core_clk [all_outputs]
set_load 0.010 [all_outputs]

# rst_core is synchronous and timed.  No false or multicycle functional paths
# are granted by this standalone experiment.
