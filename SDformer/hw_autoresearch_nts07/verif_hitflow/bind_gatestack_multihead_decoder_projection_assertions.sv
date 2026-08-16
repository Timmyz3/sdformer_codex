`timescale 1ns/1ps
`default_nettype none

bind gatestack_multihead_decoder_projection_top
    gatestack_multihead_tile_projection_assertions #(
        .TAG_W(TAG_W), .OUTPUT_TILE_W(OUTPUT_TILE_W),
        .HEAD_COUNT_W(HEAD_COUNT_W), .INPUT_CH_W(INPUT_CH_W),
        .TOKEN_ID_W(TOKEN_ID_W), .OUT_TILE(OUT_TILE), .ACC_W(ACC_W),
        .COUNTER_W(COUNTER_W)
    ) u_gatestack_multihead_decoder_projection_assertions (.*);

`default_nettype wire
