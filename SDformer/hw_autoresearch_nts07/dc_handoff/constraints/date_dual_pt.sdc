# Corner-neutral PrimeTime constraints for the DATE Local/Motion slice.
# Operating condition and max/min library are supplied by the run script.
set clock_period_ns 3.000
if {[info exists ::env(CLOCK_PERIOD_NS)] && $::env(CLOCK_PERIOD_NS) ne ""} {
    set clock_period_ns $::env(CLOCK_PERIOD_NS)
}

create_clock -name core_clk -period $clock_period_ns \
    -waveform [list 0.000 [expr {$clock_period_ns / 2.0}]] \
    [get_ports clk_core]
set_clock_uncertainty -setup 0.200 [get_clocks core_clk]
set_clock_uncertainty -hold 0.050 [get_clocks core_clk]

# Strict SDC readers do not admit the collection-manipulation commands used by
# DC.  This constraint is intentionally scoped to the banked Local tops used
# by M2C; enumerate their non-clock input port families explicitly.
set data_inputs [get_ports {
    rst_core command_valid command_tag[*] command_current_bits[*]
    command_seed_acc[*] weight_request_ready weight_response_valid
    weight_response_bank_valid[*] weight_response_data[*] output_ready
}]
set_input_delay 0.250 -clock core_clk $data_inputs
set_input_transition 0.100 $data_inputs
set_output_delay 0.250 -clock core_clk [all_outputs]
set_load 0.010 [all_outputs]
