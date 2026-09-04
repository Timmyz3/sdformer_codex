`timescale 1ns/1ps
`default_nettype none

bind gatestack_single_head_projection_top
    gatestack_single_head_projection_assertions #(
        .TOKENS(TOKENS), .BANKS(BANKS), .TOKEN_ID_W(TOKEN_ID_W),
        .OUT_TILE(OUT_TILE), .ACC_W(ACC_W),
        .OUTPUT_TILE_W(OUTPUT_TILE_W),
        .TAG_W(TAG_W), .COUNTER_W(COUNTER_W),
        .BIAS_STATIONARY_ENABLE(BIAS_STATIONARY_ENABLE),
        .IMPLICIT_BIAS_FINALIZE_ENABLE(IMPLICIT_BIAS_FINALIZE_ENABLE)
    ) u_gatestack_single_head_projection_assertions (.*);

`default_nettype wire
