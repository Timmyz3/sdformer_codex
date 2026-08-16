`timescale 1ns/1ps
`default_nettype none

bind gatestack_adaptive_csr_replay_decoder
    gatestack_adaptive_csr_replay_decoder_assertions #(
        .EVENT_WAYS(EVENT_WAYS),
        .TAG_W(TAG_W),
        .LANE_ID_W(LANE_ID_W),
        .TOKEN_ID_W(TOKEN_ID_W),
        .WAY_COUNT_W(WAY_COUNT_W)
    ) u_adaptive_csr_external_assertions (.*);

`default_nettype wire
