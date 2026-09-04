`timescale 1ns/1ps
`default_nettype none

bind gatestack_capacity_mode_selector gatestack_capacity_mode_selector_assertions #(
    .TAG_W(TAG_W),
    .SIZE_W(SIZE_W)
) u_gatestack_capacity_mode_selector_assertions (.*);

`default_nettype wire
