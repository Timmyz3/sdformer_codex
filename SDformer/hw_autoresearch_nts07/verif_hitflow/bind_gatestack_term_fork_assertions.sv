`timescale 1ns/1ps
`default_nettype none

bind gatestack_term_fork gatestack_term_fork_assertions #(
    .TAG_W(TAG_W), .LANE_ID_W(LANE_ID_W), .INPUT_CH_W(INPUT_CH_W),
    .OUTPUT_TILE_W(OUTPUT_TILE_W), .ISSUE_SEQ_W(ISSUE_SEQ_W)
) i_gatestack_term_fork_assertions (.*);

`default_nettype wire
