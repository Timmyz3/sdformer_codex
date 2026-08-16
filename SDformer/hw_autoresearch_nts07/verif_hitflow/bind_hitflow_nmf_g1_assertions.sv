`timescale 1ns/1ps
`default_nettype none

bind hitflow_nmf_g1_builder hitflow_nmf_g1_assertions #(
    .TOKENS(TOKENS),
    .LANES(LANES),
    .GATE_W(GATE_W),
    .TAG_W(TAG_W),
    .COUNTER_W(COUNTER_W)
) u_hitflow_nmf_g1_assertions (
    .clk_core(clk_core),
    .rst_core(rst_core),
    .token_valid(token_valid),
    .token_ready(token_ready),
    .protocol_error(protocol_error),
    .term_valid(term_valid),
    .term_ready(term_ready),
    .term_tag(term_tag),
    .term_gate_code(term_gate_code),
    .term_lane(term_lane),
    .term_destination_bitmap(term_destination_bitmap),
    .fallback_valid(fallback_valid),
    .fallback_ready(fallback_ready),
    .fallback_tag(fallback_tag),
    .fallback_token_id(fallback_token_id),
    .fallback_gate_code(fallback_gate_code),
    .fallback_k_bits(fallback_k_bits),
    .group_done_valid(group_done_valid),
    .group_done_ready(group_done_ready),
    .group_done_tag(group_done_tag),
    .count_tokens(count_tokens)
);

`default_nettype wire
