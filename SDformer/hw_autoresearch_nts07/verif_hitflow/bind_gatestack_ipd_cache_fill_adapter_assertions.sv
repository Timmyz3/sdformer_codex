`timescale 1ns/1ps
`default_nettype none

bind gatestack_ipd_cache_fill_adapter
gatestack_ipd_cache_fill_adapter_assertions #(
    .TAG_W(TAG_W), .HEAD_ID_W(HEAD_ID_W)
) u_gatestack_ipd_cache_fill_adapter_assertions (.*);

`default_nettype wire
