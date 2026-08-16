`timescale 1ns/1ps
`default_nettype none

module gatestack_typed_payload_serializer_assertions #(
    parameter int TOKENS = 162,
    parameter int TAG_W = 32,
    parameter int FORMAT_W = 2,
    parameter int SIZE_W = 16,
    parameter int CONTEXT_ID_W = 1,
    parameter int HEAD_ID_W = 5,
    parameter int SLOT_CAPACITY_BITS = 6656
) (
    input logic clk_core,
    input logic rst_core,
    input logic commit_begin_valid,
    input logic commit_begin_ready,
    input logic [CONTEXT_ID_W-1:0] commit_context_id,
    input logic [HEAD_ID_W-1:0] commit_head_id,
    input logic [TAG_W-1:0] commit_tag,
    input logic commit_mode_is_csr,
    input logic [SIZE_W-1:0] commit_payload_bits,
    input logic commit_word_valid,
    input logic commit_word_ready,
    input logic [63:0] commit_word_data,
    input logic commit_word_last,
    input logic done_valid,
    input logic done_ready,
    input logic [TAG_W-1:0] done_tag,
    input logic [FORMAT_W-1:0] done_format,
    input logic done_error,
    input logic [7:0] done_word_count,
    input logic protocol_error,
    input logic destination_bitmap_valid,
    input logic destination_bitmap_ready,
    input logic [TOKENS-1:0] destination_bitmap
);

    property p_commit_begin_stable_under_stall;
        @(posedge clk_core) disable iff (rst_core)
        commit_begin_valid && !commit_begin_ready |=> commit_begin_valid &&
            $stable({commit_context_id, commit_head_id, commit_tag,
                     commit_mode_is_csr, commit_payload_bits});
    endproperty

    property p_commit_word_stable_under_stall;
        @(posedge clk_core) disable iff (rst_core)
        commit_word_valid && !commit_word_ready |=> commit_word_valid &&
            $stable({commit_word_data, commit_word_last});
    endproperty

    property p_done_stable_under_stall;
        @(posedge clk_core) disable iff (rst_core)
        done_valid && !done_ready |=> done_valid &&
            $stable({done_tag, done_format, done_error, done_word_count});
    endproperty

    property p_error_is_atomic;
        @(posedge clk_core) disable iff (rst_core)
        done_valid && done_error |->
            !commit_begin_valid && !commit_word_valid;
    endproperty

    property p_commit_payload_in_slot;
        @(posedge clk_core) disable iff (rst_core)
        commit_begin_valid |-> commit_payload_bits != 0 &&
            32'(commit_payload_bits) <= SLOT_CAPACITY_BITS;
    endproperty

    property p_last_requires_valid;
        @(posedge clk_core) disable iff (rst_core)
        commit_word_last |-> commit_word_valid;
    endproperty

    property p_success_has_words;
        @(posedge clk_core) disable iff (rst_core)
        done_valid && !done_error |-> done_word_count != 0;
    endproperty

    property p_protocol_error_sticky;
        @(posedge clk_core) disable iff (rst_core)
        protocol_error |=> protocol_error;
    endproperty

    property p_destination_bitmap_stable_under_stall;
        @(posedge clk_core) disable iff (rst_core)
        destination_bitmap_valid && !destination_bitmap_ready |=>
            destination_bitmap_valid && $stable(destination_bitmap);
    endproperty

    assert property (p_commit_begin_stable_under_stall);
    assert property (p_commit_word_stable_under_stall);
    assert property (p_done_stable_under_stall);
    assert property (p_error_is_atomic);
    assert property (p_commit_payload_in_slot);
    assert property (p_last_requires_valid);
    assert property (p_success_has_words);
    assert property (p_protocol_error_sticky);
    assert property (p_destination_bitmap_stable_under_stall);

endmodule

`default_nettype wire
