`timescale 1ns/1ps
`default_nettype none

bind gatestack_resident_replay_joiner
    gatestack_resident_replay_joiner_assertions #(
        .EVENT_WAYS(EVENT_WAYS),
        .TAG_W(TAG_W),
        .TOKEN_ID_W(TOKEN_ID_W),
        .LANE_ID_W(LANE_ID_W),
        .TERM_INDEX_W(TERM_INDEX_W),
        .WAY_COUNT_W(WAY_COUNT_W)
    ) i_gatestack_resident_replay_joiner_assertions (.*);

`default_nettype wire
