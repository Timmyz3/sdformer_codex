# Strict-SDC PrimeTime constraints for the M4 P16C4L96 descriptor-resident top.
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
    rst_core descriptor_valid descriptor_row_first descriptor_row_last
    descriptor_batch_last descriptor_tag[*] descriptor_object_tag[*]
    descriptor_chunk_index[*] descriptor_chunk_count[*]
    descriptor_lane_tile_count[*] descriptor_use_motion
    descriptor_source_bits[*] descriptor_negative_bits[*]
    weight_request_ready weight_response_valid
    weight_response_bank_valid[*] weight_response_data[*] output_ready
}]
set_input_delay 0.250 -clock core_clk $data_inputs
set_input_transition 0.100 $data_inputs
set_output_delay 0.250 -clock core_clk [all_outputs]
set_load 0.010 [all_outputs]
