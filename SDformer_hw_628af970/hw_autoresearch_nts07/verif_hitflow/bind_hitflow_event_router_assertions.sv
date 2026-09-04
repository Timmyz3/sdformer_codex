`timescale 1ns/1ps
`default_nettype none

bind hitflow_event_lifetime_router hitflow_event_router_assertions #(
    .DATA_W(DATA_W),
    .TAG_W(TAG_W)
) u_hitflow_event_router_assertions (
    .clk_core(clk_core),
    .rst_core(rst_core),
    .single_valid(single_valid),
    .single_ready(single_ready),
    .single_data(single_data),
    .single_tag(single_tag),
    .fanout_q_valid(fanout_q_valid),
    .fanout_q_ready(fanout_q_ready),
    .fanout_q_data(fanout_q_data),
    .fanout_q_tag(fanout_q_tag),
    .fanout_k_valid(fanout_k_valid),
    .fanout_k_ready(fanout_k_ready),
    .fanout_k_data(fanout_k_data),
    .fanout_k_tag(fanout_k_tag),
    .pair_valid(pair_valid),
        .pair_ready(pair_ready),
        .pair_data(pair_data),
        .pair_tag(pair_tag),
        .pair_tag_mismatch(pair_tag_mismatch),
        .pair_duplicate_slot(pair_duplicate_slot),
        .route_unsupported(route_unsupported),
        .in_ready(in_ready)
    );

`default_nettype wire
