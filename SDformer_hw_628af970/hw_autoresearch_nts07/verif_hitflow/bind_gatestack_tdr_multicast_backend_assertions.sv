`timescale 1ns/1ps
`default_nettype none

bind gatestack_tdr_multicast_backend
    gatestack_tdr_multicast_backend_assertions #(
        .TAG_W(TAG_W), .BANKS(BANKS), .TOKEN_ID_W(TOKEN_ID_W),
        .OUT_TILE(OUT_TILE), .PRODUCT_W(PRODUCT_W),
        .OUTSTANDING_W(OUTSTANDING_W)
    ) u_gatestack_tdr_multicast_backend_assertions (.*);

`default_nettype wire
