`timescale 1ns/1ps
`default_nettype none

bind gatestack_destination_bitmap_assembler
    gatestack_destination_bitmap_assembler_assertions #(
        .TOKENS(TOKENS),
        .LANE_ID_W(LANE_ID_W),
        .ISSUE_SEQ_W(ISSUE_SEQ_W),
        .TAG_W(TAG_W)
    ) i_gatestack_destination_bitmap_assembler_assertions (.*);

`default_nettype wire
