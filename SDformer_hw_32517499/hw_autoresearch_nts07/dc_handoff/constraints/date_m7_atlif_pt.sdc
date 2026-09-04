# Corner-neutral PrimeTime contract for the M7 L16 ATLIF/DPTME logic slice.
# DC may use a larger hold guard while optimizing; this file fixes the
# independently reported paper-facing pre-layout contract at 50 ps.
set clock_period_ns 3.000
if {[info exists ::env(CLOCK_PERIOD_NS)] && $::env(CLOCK_PERIOD_NS) ne ""} {
    set clock_period_ns $::env(CLOCK_PERIOD_NS)
}

create_clock -name core_clk -period $clock_period_ns \
    -waveform [list 0.000 [expr {$clock_period_ns / 2.0}]] \
    [get_ports clk_core]
set_clock_uncertainty -setup 0.200 [get_clocks core_clk]
set_clock_uncertainty -hold 0.050 [get_clocks core_clk]

set data_inputs [get_ports {
    rst_core step_valid mode_t2 step_first step_last group_valid[*]
    x_groups[*] weight_slots[*] bias_slots[*] threshold_slots[*]
    step_tag[*] out_ready
}]
set_input_delay 0.250 -clock core_clk $data_inputs
set_input_transition 0.100 $data_inputs
set_output_delay 0.250 -clock core_clk [all_outputs]
set_load 0.010 [all_outputs]
