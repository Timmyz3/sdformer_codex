# M476 pre-macro logic-only constraint.  The external parent scratch is
# 64x1152b (9 KiB) synchronous 1R1W, not 144 B total.  A 64-row resident psum
# store would be 64x1824b (14.25 KiB).  Both memories remain I/O cuts; only the
# two 1152b response slots are synthesized here.
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
set_false_path -from [get_ports reset_n]
set_max_fanout 32 [current_design]
set_fix_multiple_port_nets -all -buffer_constants
