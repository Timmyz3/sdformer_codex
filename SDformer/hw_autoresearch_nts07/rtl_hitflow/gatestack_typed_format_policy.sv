`timescale 1ns/1ps
`default_nettype none

// Capacity-first exact policy: prefer IPD32W, use FADC24 only when IPD32W
// cannot fit or exceeds the supported class count, and otherwise fall back RAW.
module gatestack_typed_format_policy #(
    parameter int TOKENS              = 162,
    parameter int HEAD_DIM            = 32,
    parameter int GATE_W              = 9,
    parameter int WORD_W              = 64,
    parameter int SLOT_CAPACITY_BITS  =
        (((TOKENS * (HEAD_DIM + GATE_W)) + WORD_W - 1) / WORD_W) * WORD_W,
    parameter int IPD_CLASS_SLOTS     = 4,
    parameter int HEADER_BYTES        = 16,
    parameter int CLASS_COUNT_W       = 4,
    parameter int TERM_COUNT_W        = 8,
    parameter int EVENT_COUNT_W       = 13,
    parameter int DEST_BYTES_W        = 13,
    parameter int FORMAT_W            = 2,
    parameter int REASON_W            = 3,
    parameter int SIZE_W              = 16,
    parameter int WORD_COUNT_W        = 8
) (
    input  logic [CLASS_COUNT_W-1:0]     metadata_active_classes,
    input  logic [TERM_COUNT_W-1:0]      metadata_term_count,
    input  logic [EVENT_COUNT_W-1:0]     metadata_event_count,
    input  logic [DEST_BYTES_W-1:0]      metadata_fadc_destination_bytes,
    input  logic                         metadata_overflow,

    output logic [FORMAT_W-1:0]          decision_format,
    output logic [REASON_W-1:0]          decision_reason,
    output logic [SIZE_W-1:0]            decision_payload_bits,
    output logic [WORD_COUNT_W-1:0]      decision_word_count,
    output logic [SIZE_W-1:0]            decision_ipd_payload_bytes,
    output logic [SIZE_W-1:0]            decision_fadc_payload_bytes
);

    localparam int RAW_PAYLOAD_BITS = TOKENS * (HEAD_DIM + GATE_W);
    localparam int SLOT_BYTES = SLOT_CAPACITY_BITS / 8;
    localparam logic [FORMAT_W-1:0] FORMAT_RAW = FORMAT_W'(0);
    localparam logic [FORMAT_W-1:0] FORMAT_IPD32W = FORMAT_W'(1);
    localparam logic [FORMAT_W-1:0] FORMAT_FADC24 = FORMAT_W'(2);
    localparam logic [REASON_W-1:0] REASON_IPD_FIT = REASON_W'(0);
    localparam logic [REASON_W-1:0] REASON_FADC_IPD_CLASS = REASON_W'(1);
    localparam logic [REASON_W-1:0] REASON_FADC_IPD_CAPACITY = REASON_W'(2);
    localparam logic [REASON_W-1:0] REASON_RAW_FADC_CAPACITY = REASON_W'(3);
    localparam logic [REASON_W-1:0] REASON_RAW_METADATA_OVERFLOW = REASON_W'(4);

    logic [31:0] ipd_payload_bytes_comb;
    logic [31:0] fadc_payload_bytes_comb;
    logic [31:0] selected_payload_bits_comb;
    logic ipd_class_fits;
    logic ipd_capacity_fits;
    logic fadc_capacity_fits;

    always_comb begin
        ipd_payload_bytes_comb = HEADER_BYTES +
            ((((32'(metadata_term_count) + 1) >> 1)) << 3) +
            32'(metadata_event_count);
        // FADC24 descriptor is exactly three bytes: 3*T = T + 2*T.
        fadc_payload_bytes_comb = HEADER_BYTES +
            32'(metadata_term_count) + (32'(metadata_term_count) << 1) +
            32'(metadata_fadc_destination_bytes);
        ipd_class_fits =
            32'(metadata_active_classes) <= IPD_CLASS_SLOTS;
        ipd_capacity_fits = ipd_payload_bytes_comb <= SLOT_BYTES;
        fadc_capacity_fits = fadc_payload_bytes_comb <= SLOT_BYTES;

        decision_format = FORMAT_RAW;
        decision_reason = REASON_RAW_FADC_CAPACITY;
        selected_payload_bits_comb = RAW_PAYLOAD_BITS;

        if (metadata_overflow) begin
            decision_reason = REASON_RAW_METADATA_OVERFLOW;
        end else if (ipd_class_fits && ipd_capacity_fits) begin
            decision_format = FORMAT_IPD32W;
            decision_reason = REASON_IPD_FIT;
            selected_payload_bits_comb = ipd_payload_bytes_comb << 3;
        end else if (fadc_capacity_fits) begin
            decision_format = FORMAT_FADC24;
            decision_reason = ipd_class_fits ?
                REASON_FADC_IPD_CAPACITY : REASON_FADC_IPD_CLASS;
            selected_payload_bits_comb = fadc_payload_bytes_comb << 3;
        end

        decision_payload_bits = SIZE_W'(selected_payload_bits_comb);
        decision_word_count = WORD_COUNT_W'(
            (selected_payload_bits_comb + WORD_W - 1) / WORD_W);
        decision_ipd_payload_bytes = SIZE_W'(ipd_payload_bytes_comb);
        decision_fadc_payload_bytes = SIZE_W'(fadc_payload_bytes_comb);
    end

endmodule

`default_nettype wire
