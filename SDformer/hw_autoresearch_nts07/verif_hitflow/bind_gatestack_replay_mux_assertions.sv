`timescale 1ns/1ps
`default_nettype none

bind gatestack_replay_mux gatestack_replay_mux_assertions #(
    .SOURCES(SOURCES),
    .EVENT_WAYS(EVENT_WAYS),
    .TOKEN_ID_W(TOKEN_ID_W),
    .LANE_ID_W(LANE_ID_W),
    .ISSUE_SEQ_W(ISSUE_SEQ_W),
    .TAG_W(TAG_W),
    .WAY_COUNT_W(WAY_COUNT_W),
    .ROUTE_W(ROUTE_W)
) i_gatestack_replay_mux_assertions (.*);

`default_nettype wire
