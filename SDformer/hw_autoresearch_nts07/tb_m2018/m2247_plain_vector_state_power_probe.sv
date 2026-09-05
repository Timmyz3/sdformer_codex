`timescale 1ns/1ps
// VCS native SAIF emits duplicate unindexed names for the DUT enum. Mirror
// only its representation as a plain vector. No RTL or stimulus is changed.
module m2247_plain_vector_state_power_probe(input wire [3:0] state_q);
endmodule
bind tb_m2217_m2018_tsbg_matched_native_saif_power
    m2247_plain_vector_state_power_probe state_power_probe (.state_q(dut_axis.state_q));
