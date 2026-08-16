`timescale 1ns/1ps
`default_nettype none

bind gatestack_event_compactor gatestack_event_compactor_assertions #(
    .WAYS(WAYS),
    .TAG_W(TAG_W),
    .TOKEN_ID_W(TOKEN_ID_W),
    .SLOT_ID_W(SLOT_ID_W),
    .LANE_ID_W(LANE_ID_W),
    .COUNT_W(COUNT_W)
) u_gatestack_event_compactor_assertions (.*);

`default_nettype wire
