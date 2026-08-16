`timescale 1ns/1ps
`default_nettype none

bind gatestack_dualtag_replay_lifecycle_manager
gatestack_dualtag_replay_lifecycle_assertions #(
    .TAG_W(TAG_W), .CONTEXT_ID_W(CONTEXT_ID_W),
    .HEAD_ID_W(HEAD_ID_W)
) u_gatestack_dualtag_replay_lifecycle_assertions (.*);

`default_nettype wire
