`timescale 1ns/1ps
`default_nettype none

bind gatestack_ppdi_dctf32_bank_executor
    gatestack_ppdi_dctf32_bank_executor_assertions #(
        .OUT_TILE(OUT_TILE), .PRODUCT_W(PRODUCT_W),
        .GROUP_TAG_W(GROUP_TAG_W), .CMD_SEQUENCE_W(CMD_SEQUENCE_W),
        .ISSUE_SEQ_W(ISSUE_SEQ_W), .INPUT_CH_W(INPUT_CH_W),
        .GATE_W(GATE_W), .LANE_ID_W(LANE_ID_W),
        .TOKEN_ID_W(TOKEN_ID_W),
        .LOGICAL_SUPERTILE_W(LOGICAL_SUPERTILE_W),
        .COUNTER_W(COUNTER_W)
    ) i_gatestack_ppdi_dctf32_bank_executor_assertions (.*);

`default_nettype wire
