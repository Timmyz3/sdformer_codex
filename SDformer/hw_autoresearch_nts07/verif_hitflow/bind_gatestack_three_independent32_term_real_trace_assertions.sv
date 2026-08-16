`timescale 1ns/1ps
`default_nettype none

bind gatestack_three_independent32_term_projection_top
    gatestack_three_independent32_term_real_trace_assertions #(
        .EVENT_WAYS(EVENT_WAYS),
        .BANKS(BANKS),
        .GATE_W(GATE_W),
        .ACC_W(ACC_W),
        .TAG_W(TAG_W),
        .INPUT_CH_W(INPUT_CH_W),
        .OUTPUT_TILE_W(OUTPUT_TILE_W),
        .ISSUE_SEQ_W(ISSUE_SEQ_W),
        .TOKEN_ID_W(TOKEN_ID_W),
        .LANE_ID_W(LANE_ID_W),
        .WAY_COUNT_W(WAY_COUNT_W)
    ) i_gatestack_three_independent32_term_real_trace_assertions (.*);

`default_nettype wire
