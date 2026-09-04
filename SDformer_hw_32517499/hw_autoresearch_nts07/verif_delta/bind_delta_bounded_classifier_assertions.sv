`timescale 1ns/1ps
`default_nettype none

bind delta_bounded_classifier delta_bounded_classifier_assertions #(
    .TAG_W(TAG_W),
    .PAYLOAD_W(PAYLOAD_W)
) u_delta_bounded_classifier_assertions (.*);

`default_nettype wire
