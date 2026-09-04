`timescale 1ns/1ps
`default_nettype none

bind gatestack_replay_control_plane_top
gatestack_replay_control_plane_assertions #(
    .TAG_W(TAG_W),
    .CONTEXT_ID_W(CONTEXT_ID_W),
    .HEAD_ID_W(HEAD_ID_W),
    .HEAD_COUNT_W(HEAD_COUNT_W)
) u_gatestack_replay_control_plane_assertions (.*);

`default_nettype wire
