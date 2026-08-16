`timescale 1ns/1ps
`default_nettype none

bind gatestack_fadc24_streaming_replay_decoder
    gatestack_fadc24_streaming_replay_decoder_assertions #(
        .EVENT_WAYS(EVENT_WAYS),
        .TAG_W(TAG_W),
        .TOKEN_ID_W(TOKEN_ID_W),
        .LANE_ID_W(LANE_ID_W),
        .WAY_COUNT_W(WAY_COUNT_W)
    ) i_gatestack_fadc24_streaming_replay_decoder_assertions (.*);

`default_nettype wire
