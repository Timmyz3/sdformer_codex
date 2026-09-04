`timescale 1ns/1ps
`default_nettype none

module gatestack_head_slot_sram_adapter_assertions #(
    parameter int CONTEXTS      = 2,
    parameter int HEADS         = 24,
    parameter int HEAD_BITS     = 6642,
    parameter int WORD_W        = 64,
    parameter int SLOT_CAPACITY_BITS =
        ((HEAD_BITS + WORD_W - 1) / WORD_W) * WORD_W,
    parameter int TAG_W         = 32,
    parameter int SIZE_W        = 16,
    parameter int FORMAT_W      = 2,
    parameter int COUNTER_W     = 32,
    parameter int CONTEXT_ID_W  = (CONTEXTS <= 1) ? 1 : $clog2(CONTEXTS),
    parameter int HEAD_ID_W     = (HEADS <= 1) ? 1 : $clog2(HEADS),
    parameter int WORD_INDEX_W  =
        (((SLOT_CAPACITY_BITS + WORD_W - 1) / WORD_W) <= 1) ? 1 :
        $clog2((SLOT_CAPACITY_BITS + WORD_W - 1) / WORD_W)
) (
    input logic clk_core,
    input logic rst_core,
    input logic commit_begin_valid,
    input logic commit_begin_ready,
    input logic [CONTEXT_ID_W-1:0] commit_context_id,
    input logic [HEAD_ID_W-1:0] commit_head_id,
    input logic [SIZE_W-1:0] commit_payload_bits,
    input logic commit_word_ready,
    input logic inspect_meta_valid,
    input logic inspect_meta_ready,
    input logic inspect_exists,
    input logic [TAG_W-1:0] inspect_tag,
    input logic inspect_mode_is_csr,
    input logic [FORMAT_W-1:0] inspect_format,
    input logic [SIZE_W-1:0] inspect_payload_bits,
    input logic [SIZE_W-1:0] inspect_word_count,
    input logic replay_begin_valid,
    input logic replay_begin_ready,
    input logic [CONTEXT_ID_W-1:0] replay_context_id,
    input logic [HEAD_ID_W-1:0] replay_head_id,
    input logic [WORD_INDEX_W-1:0] replay_start_word,
    input logic replay_word_valid,
    input logic replay_word_ready,
    input logic [WORD_W-1:0] replay_word_data,
    input logic [WORD_INDEX_W-1:0] replay_word_index,
    input logic replay_word_last,
    input logic [TAG_W-1:0] replay_tag,
    input logic replay_mode_is_csr,
    input logic [FORMAT_W-1:0] replay_format,
    input logic [SIZE_W-1:0] replay_payload_bits,
    input logic release_valid,
    input logic release_ready,
    input logic [CONTEXT_ID_W-1:0] release_context_id,
    input logic [HEAD_ID_W-1:0] release_head_id,
    input logic commit_session_active,
    input logic replay_session_active,
    input logic protocol_error,
    input logic [COUNTER_W-1:0] count_commit_heads,
    input logic [COUNTER_W-1:0] count_invalid_headers
);

    property p_replay_stable_under_stall;
        @(posedge clk_core) disable iff (rst_core)
        replay_word_valid && !replay_word_ready |=> replay_word_valid &&
            $stable({replay_word_data, replay_word_index, replay_word_last,
                     replay_tag, replay_mode_is_csr, replay_format,
                     replay_payload_bits});
    endproperty

    property p_commit_word_requires_session;
        @(posedge clk_core) disable iff (rst_core)
        commit_word_ready |-> commit_session_active;
    endproperty

    property p_inspect_stable_under_stall;
        @(posedge clk_core) disable iff (rst_core)
        inspect_meta_valid && !inspect_meta_ready |=> inspect_meta_valid &&
            $stable({inspect_exists, inspect_tag, inspect_mode_is_csr,
                     inspect_format, inspect_payload_bits,
                     inspect_word_count});
    endproperty

    property p_release_is_not_during_same_slot_session;
        @(posedge clk_core) disable iff (rst_core)
        release_valid && release_ready |->
            !((commit_session_active &&
               commit_context_id == release_context_id &&
               commit_head_id == release_head_id) ||
              (replay_session_active &&
               replay_context_id == release_context_id &&
               replay_head_id == release_head_id));
    endproperty

    property p_protocol_error_sticky;
        @(posedge clk_core) disable iff (rst_core)
        protocol_error |=> protocol_error;
    endproperty

    property p_commit_payload_is_bounded;
        @(posedge clk_core) disable iff (rst_core)
        commit_begin_valid && commit_begin_ready |->
            commit_payload_bits > 0 &&
            32'(commit_payload_bits) <= SLOT_CAPACITY_BITS;
    endproperty

    property p_replay_start_is_bounded;
        @(posedge clk_core) disable iff (rst_core)
        replay_begin_valid && replay_begin_ready |=>
            replay_session_active && 32'($past(replay_start_word)) <
            ((32'(replay_payload_bits) + WORD_W - 1) / WORD_W);
    endproperty

    property p_invalid_header_never_counts_as_commit;
        @(posedge clk_core) disable iff (rst_core)
        count_invalid_headers != $past(count_invalid_headers) |->
            !commit_session_active &&
            count_commit_heads == $past(count_commit_heads);
    endproperty

    assert property (p_replay_stable_under_stall);
    assert property (p_commit_word_requires_session);
    assert property (p_inspect_stable_under_stall);
    assert property (p_release_is_not_during_same_slot_session);
    assert property (p_protocol_error_sticky);
    assert property (p_commit_payload_is_bounded);
    assert property (p_replay_start_is_bounded);
    assert property (p_invalid_header_never_counts_as_commit);

endmodule

`default_nettype wire
