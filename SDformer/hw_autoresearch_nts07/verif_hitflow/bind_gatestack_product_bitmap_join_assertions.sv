`timescale 1ns/1ps
`default_nettype none

bind gatestack_product_bitmap_join
    gatestack_product_bitmap_join_assertions #(
        .TOKENS(TOKENS),
        .OUT_TILE(OUT_TILE),
        .PRODUCT_W(PRODUCT_W),
        .INPUT_CH_W(INPUT_CH_W),
        .OUTPUT_TILE_W(OUTPUT_TILE_W),
        .ISSUE_SEQ_W(ISSUE_SEQ_W),
        .TAG_W(TAG_W)
    ) i_gatestack_product_bitmap_join_assertions (.*);

`default_nettype wire
