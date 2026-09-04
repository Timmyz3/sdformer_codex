`timescale 1ns/1ps
`default_nettype none

module gatestack_typed_builder_frontend_assertions #(
    parameter int TAG_W          = 32,
    parameter int FORMAT_W       = 2,
    parameter int REASON_W       = 3,
    parameter int SIZE_W         = 16,
    parameter int WORD_COUNT_W   = 8,
    parameter int TERM_COUNT_W   = 8,
    parameter int EVENT_COUNT_W  = 13,
    parameter int SLOT_BYTES     = 832
) (
    input logic clk_core,
    input logic rst_core,
    input logic head_start_valid,
    input logic head_start_ready,
    input logic term_valid,
    input logic term_ready,
    input logic [7:0] term_destination_count,
    input logic head_end_valid,
    input logic head_end_ready,
    input logic head_end_builder_error,
    input logic decision_valid,
    input logic decision_ready,
    input logic [TAG_W-1:0] decision_tag,
    input logic [FORMAT_W-1:0] decision_format,
    input logic [REASON_W-1:0] decision_reason,
    input logic [SIZE_W-1:0] decision_payload_bits,
    input logic [WORD_COUNT_W-1:0] decision_word_count,
    input logic [SIZE_W-1:0] decision_ipd_payload_bytes,
    input logic [SIZE_W-1:0] decision_fadc_payload_bytes,
    input logic [TERM_COUNT_W-1:0] decision_term_count,
    input logic [EVENT_COUNT_W-1:0] decision_event_count,
    input logic [TERM_COUNT_W-1:0] decision_bitmap_term_count,
    input logic decision_metadata_overflow
);

    localparam logic [FORMAT_W-1:0] FORMAT_RAW = FORMAT_W'(0);
    localparam logic [FORMAT_W-1:0] FORMAT_IPD32W = FORMAT_W'(1);
    localparam logic [FORMAT_W-1:0] FORMAT_FADC24 = FORMAT_W'(2);

    property p_decision_stable_under_stall;
        @(posedge clk_core) disable iff (rst_core)
        decision_valid && !decision_ready |=> decision_valid &&
            $stable({decision_tag, decision_format, decision_reason,
                     decision_payload_bits, decision_word_count,
                     decision_ipd_payload_bytes,
                     decision_fadc_payload_bytes, decision_term_count,
                     decision_event_count, decision_bitmap_term_count,
                     decision_metadata_overflow});
    endproperty

    property p_no_zero_length_accepted_term;
        @(posedge clk_core) disable iff (rst_core)
        term_valid && term_ready |-> term_destination_count != 0;
    endproperty

    property p_no_simultaneous_start_end;
        @(posedge clk_core) disable iff (rst_core)
        !(head_start_valid && head_start_ready &&
          head_end_valid && head_end_ready);
    endproperty

    property p_builder_error_forces_raw_metadata;
        @(posedge clk_core) disable iff (rst_core)
        head_end_valid && head_end_ready && head_end_builder_error |=>
            decision_valid && decision_metadata_overflow;
    endproperty

    property p_legal_format;
        @(posedge clk_core) disable iff (rst_core)
        decision_valid |->
            decision_format == FORMAT_RAW ||
            decision_format == FORMAT_IPD32W ||
            decision_format == FORMAT_FADC24;
    endproperty

    property p_overflow_forces_raw;
        @(posedge clk_core) disable iff (rst_core)
        decision_valid && decision_metadata_overflow |->
            decision_format == FORMAT_RAW && decision_reason == REASON_W'(4);
    endproperty

    property p_ipd_fits_slot;
        @(posedge clk_core) disable iff (rst_core)
        decision_valid && decision_format == FORMAT_IPD32W |->
            32'(decision_ipd_payload_bytes) <= SLOT_BYTES;
    endproperty

    property p_fadc_fits_slot;
        @(posedge clk_core) disable iff (rst_core)
        decision_valid && decision_format == FORMAT_FADC24 |->
            32'(decision_fadc_payload_bytes) <= SLOT_BYTES;
    endproperty

    property p_word_count_matches_bits;
        @(posedge clk_core) disable iff (rst_core)
        decision_valid |->
            32'(decision_word_count) ==
            ((32'(decision_payload_bits) + 63) >> 6);
    endproperty

    assert property (p_decision_stable_under_stall);
    assert property (p_no_zero_length_accepted_term);
    assert property (p_no_simultaneous_start_end);
    assert property (p_builder_error_forces_raw_metadata);
    assert property (p_legal_format);
    assert property (p_overflow_forces_raw);
    assert property (p_ipd_fits_slot);
    assert property (p_fadc_fits_slot);
    assert property (p_word_count_matches_bits);

endmodule

`default_nettype wire
