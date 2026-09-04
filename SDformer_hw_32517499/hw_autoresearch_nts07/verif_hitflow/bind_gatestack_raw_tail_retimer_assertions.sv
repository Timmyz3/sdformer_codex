`timescale 1ns/1ps
`default_nettype none

bind gatestack_raw_tail_retimer gatestack_raw_tail_retimer_assertions #(
    .TAG_W(TAG_W), .LANE_ID_W(LANE_ID_W), .TOKEN_ID_W(TOKEN_ID_W)
) u_gatestack_raw_tail_retimer_assertions (.*);

`default_nettype wire
