`timescale 1ns/1ps
`default_nettype none

bind gatestack_output_tile_scheduler
gatestack_output_tile_scheduler_assertions #(
    .TAG_W(TAG_W),
    .INPUT_CH_W(INPUT_CH_W),
    .OUTPUT_TILE_W(OUTPUT_TILE_W),
    .HEAD_COUNT_W(HEAD_COUNT_W),
    .CONTEXT_ID_W(CONTEXT_ID_W),
    .HEAD_ID_W(HEAD_ID_W)
) u_gatestack_output_tile_scheduler_assertions (.*);

`default_nettype wire
