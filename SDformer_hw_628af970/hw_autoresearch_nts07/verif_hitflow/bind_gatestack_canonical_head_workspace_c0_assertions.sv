`timescale 1ns/1ps
`default_nettype none

bind gatestack_canonical_head_workspace_c0
    gatestack_canonical_head_workspace_c0_assertions #(
        .TOKENS(TOKENS),
        .LANES(LANES),
        .GATE_W(GATE_W),
        .CONTEXTS(CONTEXTS),
        .HEADS(HEADS),
        .TAG_W(TAG_W),
        .CONTEXT_ID_W(CONTEXT_ID_W),
        .HEAD_ID_W(HEAD_ID_W)
    ) i_gatestack_canonical_head_workspace_c0_assertions (.*);

`default_nettype wire

