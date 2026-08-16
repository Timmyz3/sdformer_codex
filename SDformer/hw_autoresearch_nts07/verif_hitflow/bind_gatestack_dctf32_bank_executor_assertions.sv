`timescale 1ns/1ps
`default_nettype none

bind gatestack_dctf32_bank_executor
    gatestack_dctf32_bank_executor_assertions #(
        .OUT_TILE(OUT_TILE),
        .PRODUCT_W(PRODUCT_W),
        .GROUP_TAG_W(GROUP_TAG_W),
        .ISSUE_SEQ_W(ISSUE_SEQ_W),
        .INPUT_CH_W(INPUT_CH_W),
        .TOKEN_ID_W(TOKEN_ID_W),
        .OUTPUT_TILE_W(OUTPUT_TILE_W),
        .EPOCH_W(EPOCH_W),
        .COUNTER_W(COUNTER_W)
    ) i_gatestack_dctf32_bank_executor_assertions (.*);

`default_nettype wire
