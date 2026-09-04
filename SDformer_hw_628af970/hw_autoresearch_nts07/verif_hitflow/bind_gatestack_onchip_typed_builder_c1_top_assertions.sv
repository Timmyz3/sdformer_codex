`timescale 1ns/1ps
`default_nettype none

bind gatestack_onchip_typed_builder_c1_top
    gatestack_onchip_typed_builder_c1_top_assertions #(
        .TAG_W(TAG_W),
        .FORMAT_W(FORMAT_W),
        .SIZE_W(SIZE_W),
        .COUNTER_W(COUNTER_W)
    ) u_gatestack_onchip_typed_builder_c1_top_assertions (
        .clk_core,
        .rst_core,
        .head_begin_valid,
        .head_begin_ready,
        .token_valid,
        .token_ready,
        .token_last,
        .done_valid,
        .done_ready,
        .done_tag,
        .done_format,
        .done_error,
        .done_word_count,
        .selected_reason,
        .selected_payload_bits,
        .done_sequence,
        .capture_active_q,
        .session_active_q,
        .session_abort_q,
        .emit_started_q,
        .emit_owner_q,
        .ws_head_begin_valid,
        .ws_metadata_ready,
        .ws_emit_start_valid,
        .oldest_valid,
        .builder_begin_valid,
        .next_capture_sequence_q,
        .next_emit_sequence_q
    );

`default_nettype wire
