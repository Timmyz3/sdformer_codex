`timescale 1ns/1ps
`default_nettype none

bind gatestack_head_slot_sram_adapter
    gatestack_head_slot_sram_adapter_assertions #(
        .CONTEXTS(CONTEXTS),
        .HEADS(HEADS),
        .HEAD_BITS(HEAD_BITS),
        .WORD_W(WORD_W),
        .SLOT_CAPACITY_BITS(SLOT_CAPACITY_BITS),
        .TAG_W(TAG_W),
        .SIZE_W(SIZE_W),
        .FORMAT_W(FORMAT_W),
        .COUNTER_W(COUNTER_W),
        .CONTEXT_ID_W(CONTEXT_ID_W),
        .HEAD_ID_W(HEAD_ID_W),
        .WORD_INDEX_W(WORD_INDEX_W)
    ) i_gatestack_head_slot_sram_adapter_assertions (.*);

`default_nettype wire
