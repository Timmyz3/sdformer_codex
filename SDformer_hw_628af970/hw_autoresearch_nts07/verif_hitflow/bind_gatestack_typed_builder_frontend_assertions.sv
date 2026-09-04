`timescale 1ns/1ps
`default_nettype none

bind gatestack_typed_builder_frontend
    gatestack_typed_builder_frontend_assertions #(
        .TAG_W(TAG_W),
        .FORMAT_W(FORMAT_W),
        .REASON_W(REASON_W),
        .SIZE_W(SIZE_W),
        .WORD_COUNT_W(WORD_COUNT_W),
        .TERM_COUNT_W(TERM_COUNT_W),
        .EVENT_COUNT_W(EVENT_COUNT_W),
        .SLOT_BYTES(SLOT_CAPACITY_BITS / 8)
    ) u_gatestack_typed_builder_frontend_assertions (
        .clk_core,
        .rst_core,
        .head_start_valid,
        .head_start_ready,
        .term_valid,
        .term_ready,
        .term_destination_count,
        .head_end_valid,
        .head_end_ready,
        .head_end_builder_error,
        .decision_valid,
        .decision_ready,
        .decision_tag,
        .decision_format,
        .decision_reason,
        .decision_payload_bits,
        .decision_word_count,
        .decision_ipd_payload_bytes,
        .decision_fadc_payload_bytes,
        .decision_term_count,
        .decision_event_count,
        .decision_bitmap_term_count,
        .decision_metadata_overflow
    );

`default_nettype wire
