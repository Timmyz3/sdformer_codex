`timescale 1ns/1ps
`default_nettype none

bind gatestack_raw41_replay_decoder
    gatestack_raw41_replay_decoder_assertions #(
        .TAG_W(TAG_W),
        .TOKEN_ID_W(TOKEN_ID_W),
        .LANE_ID_W(LANE_ID_W)
    ) i_gatestack_raw41_replay_decoder_assertions (.*);

`default_nettype wire
