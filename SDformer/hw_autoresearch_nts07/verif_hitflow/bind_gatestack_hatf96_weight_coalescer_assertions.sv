`timescale 1ns/1ps
`default_nettype none

bind gatestack_hatf96_weight_coalescer
    gatestack_hatf96_weight_coalescer_assertions #(
        .BANK_COUNT(BANK_COUNT),
        .LANES_PER_BANK(LANES_PER_BANK),
        .WEIGHT_W(WEIGHT_W),
        .TAG_W(TAG_W),
        .INPUT_CH_W(INPUT_CH_W),
        .OUTPUT_TILE_W(OUTPUT_TILE_W)
    ) u_gatestack_hatf96_weight_coalescer_assertions (
        .*
    );

`default_nettype wire
