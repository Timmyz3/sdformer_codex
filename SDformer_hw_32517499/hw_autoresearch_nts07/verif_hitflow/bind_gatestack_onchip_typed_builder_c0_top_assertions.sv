`timescale 1ns/1ps
`default_nettype none

bind gatestack_onchip_typed_builder_c0_top
    gatestack_onchip_typed_builder_c0_top_assertions #(
        .TAG_W(TAG_W), .FORMAT_W(FORMAT_W), .SIZE_W(SIZE_W)
    ) i_gatestack_onchip_typed_builder_c0_top_assertions (.*);

`default_nettype wire

