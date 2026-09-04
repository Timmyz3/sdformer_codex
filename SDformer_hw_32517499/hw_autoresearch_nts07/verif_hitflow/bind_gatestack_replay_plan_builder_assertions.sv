`timescale 1ns/1ps
`default_nettype none

bind gatestack_replay_plan_builder
gatestack_replay_plan_builder_assertions #(
    .TAG_W(TAG_W), .CONTEXT_ID_W(CONTEXT_ID_W),
    .HEAD_ID_W(HEAD_ID_W), .ROUTE_W(ROUTE_W), .FORMAT_W(FORMAT_W),
    .HEAD_COUNT_W(HEAD_COUNT_W), .INPUT_CH_W(INPUT_CH_W),
    .OUTPUT_TILE_W(OUTPUT_TILE_W), .WORD_INDEX_W(WORD_INDEX_W),
    .EVENT_COUNT_W(EVENT_COUNT_W)
) u_gatestack_replay_plan_builder_assertions (.*);

`default_nettype wire
