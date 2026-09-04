`timescale 1ns/1ps
`default_nettype none

bind gatestack_typed_payload_serializer
    gatestack_typed_payload_serializer_assertions #(
        .TOKENS(TOKENS),
        .TAG_W(TAG_W),
        .FORMAT_W(FORMAT_W),
        .SIZE_W(SIZE_W),
        .CONTEXT_ID_W(CONTEXT_ID_W),
        .HEAD_ID_W(HEAD_ID_W),
        .SLOT_CAPACITY_BITS(SLOT_WORDS * WORD_W)
    ) u_gatestack_typed_payload_serializer_assertions (
        .clk_core,
        .rst_core,
        .commit_begin_valid,
        .commit_begin_ready,
        .commit_context_id,
        .commit_head_id,
        .commit_tag,
        .commit_mode_is_csr,
        .commit_payload_bits,
        .commit_word_valid,
        .commit_word_ready,
        .commit_word_data,
        .commit_word_last,
        .done_valid,
        .done_ready,
        .done_tag,
        .done_format,
        .done_error,
        .done_word_count,
        .protocol_error,
        .destination_bitmap_valid,
        .destination_bitmap_ready,
        .destination_bitmap
    );

`default_nettype wire
