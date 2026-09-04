`timescale 1ns/1ps
`default_nettype none

bind gatestack_descriptor_residency_cache
    gatestack_descriptor_residency_cache_assertions #(
        .TAG_W(TAG_W),
        .TERM_INDEX_W(TERM_INDEX_W)
    ) i_gatestack_descriptor_residency_cache_assertions (.*);

`default_nettype wire
