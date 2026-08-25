# Strict-SDC PrimeTime constraints for the M3 P16C4 dual-line top.
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
    rst_core command_valid command_tag[*] command_object_tag[*]
    command_batch_last command_use_motion command_source_bits[*]
    command_negative_bits[*] command_seed_acc[*]
    weight_request_ready weight_response_valid
    weight_response_bank_valid[*] weight_response_data[*] output_ready
}]
set_input_delay 0.250 -clock core_clk $data_inputs
set_input_transition 0.100 $data_inputs
set_output_delay 0.250 -clock core_clk [all_outputs]
set_load 0.010 [all_outputs]
