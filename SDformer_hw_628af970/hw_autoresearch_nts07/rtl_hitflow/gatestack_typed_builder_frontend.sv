`timescale 1ns/1ps
`default_nettype none

// Wiring-only frontend joining streamed fanout metadata to the typed policy.
// Payload serialization is intentionally a separate backend contract.
module gatestack_typed_builder_frontend #(
    parameter int TOKENS              = 162,
    parameter int HEAD_DIM            = 32,
    parameter int GATE_W              = 9,
    parameter int WORD_W              = 64,
    parameter int SLOT_CAPACITY_BITS  = 6656,
    parameter int IPD_CLASS_SLOTS     = 4,
    parameter int TAG_W               = 32,
    parameter int CLASS_COUNT_W       = 4,
    parameter int TERM_COUNT_W        = 8,
    parameter int EVENT_COUNT_W       = 13,
    parameter int DEST_COUNT_W        = 8,
    parameter int DEST_BYTES_W        = 13,
    parameter int FORMAT_W            = 2,
    parameter int REASON_W            = 3,
    parameter int SIZE_W              = 16,
    parameter int WORD_COUNT_W        = 8,
    parameter int COUNTER_W           = 32
) (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         head_start_valid,
    output logic                         head_start_ready,
    input  logic [TAG_W-1:0]             head_start_tag,
    input  logic [CLASS_COUNT_W-1:0]     head_start_active_classes,
    input  logic                         term_valid,
    output logic                         term_ready,
    input  logic [DEST_COUNT_W-1:0]      term_destination_count,
    input  logic                         head_end_valid,
    output logic                         head_end_ready,
    input  logic                         head_end_builder_error,

    output logic                         decision_valid,
    input  logic                         decision_ready,
    output logic [TAG_W-1:0]             decision_tag,
    output logic [FORMAT_W-1:0]          decision_format,
    output logic [REASON_W-1:0]          decision_reason,
    output logic [SIZE_W-1:0]            decision_payload_bits,
    output logic [WORD_COUNT_W-1:0]      decision_word_count,
    output logic [SIZE_W-1:0]            decision_ipd_payload_bytes,
    output logic [SIZE_W-1:0]            decision_fadc_payload_bytes,
    output logic [TERM_COUNT_W-1:0]      decision_term_count,
    output logic [EVENT_COUNT_W-1:0]     decision_event_count,
    output logic [TERM_COUNT_W-1:0]      decision_bitmap_term_count,
    output logic                         decision_metadata_overflow,

    output logic [COUNTER_W-1:0]         count_heads,
    output logic [COUNTER_W-1:0]         count_terms,
    output logic [COUNTER_W-1:0]         count_invalid_terms,
    output logic [COUNTER_W-1:0]         count_metadata_overflows
);

    logic [CLASS_COUNT_W-1:0] metadata_active_classes;
    logic [DEST_BYTES_W-1:0] metadata_fadc_destination_bytes;

    gatestack_format_metadata_accumulator #(
        .TAG_W(TAG_W),
        .CLASS_COUNT_W(CLASS_COUNT_W),
        .TERM_COUNT_W(TERM_COUNT_W),
        .EVENT_COUNT_W(EVENT_COUNT_W),
        .DEST_COUNT_W(DEST_COUNT_W),
        .DEST_BYTES_W(DEST_BYTES_W),
        .COUNTER_W(COUNTER_W)
    ) u_metadata_accumulator (
        .clk_core,
        .rst_core,
        .head_start_valid,
        .head_start_ready,
        .head_start_tag,
        .head_start_active_classes,
        .term_valid,
        .term_ready,
        .term_destination_count,
        .head_end_valid,
        .head_end_ready,
        .head_end_builder_error,
        .metadata_valid(decision_valid),
        .metadata_ready(decision_ready),
        .metadata_tag(decision_tag),
        .metadata_active_classes,
        .metadata_term_count(decision_term_count),
        .metadata_event_count(decision_event_count),
        .metadata_bitmap_term_count(decision_bitmap_term_count),
        .metadata_fadc_destination_bytes,
        .metadata_overflow(decision_metadata_overflow),
        .count_heads,
        .count_terms,
        .count_invalid_terms,
        .count_metadata_overflows
    );

    gatestack_typed_format_policy #(
        .TOKENS(TOKENS),
        .HEAD_DIM(HEAD_DIM),
        .GATE_W(GATE_W),
        .WORD_W(WORD_W),
        .SLOT_CAPACITY_BITS(SLOT_CAPACITY_BITS),
        .IPD_CLASS_SLOTS(IPD_CLASS_SLOTS),
        .CLASS_COUNT_W(CLASS_COUNT_W),
        .TERM_COUNT_W(TERM_COUNT_W),
        .EVENT_COUNT_W(EVENT_COUNT_W),
        .DEST_BYTES_W(DEST_BYTES_W),
        .FORMAT_W(FORMAT_W),
        .REASON_W(REASON_W),
        .SIZE_W(SIZE_W),
        .WORD_COUNT_W(WORD_COUNT_W)
    ) u_policy (
        .metadata_active_classes,
        .metadata_term_count(decision_term_count),
        .metadata_event_count(decision_event_count),
        .metadata_fadc_destination_bytes,
        .metadata_overflow(decision_metadata_overflow),
        .decision_format,
        .decision_reason,
        .decision_payload_bits,
        .decision_word_count,
        .decision_ipd_payload_bytes,
        .decision_fadc_payload_bytes
    );

endmodule

`default_nettype wire
