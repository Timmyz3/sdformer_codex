`timescale 1ns/1ps
`default_nettype none

module gatestack_onchip_typed_builder_c1_top_assertions #(
    parameter int TAG_W = 32,
    parameter int FORMAT_W = 2,
    parameter int SIZE_W = 16,
    parameter int COUNTER_W = 32
) (
    input logic clk_core,
    input logic rst_core,
    input logic head_begin_valid,
    input logic head_begin_ready,
    input logic token_valid,
    input logic token_ready,
    input logic token_last,
    input logic done_valid,
    input logic done_ready,
    input logic [TAG_W-1:0] done_tag,
    input logic [FORMAT_W-1:0] done_format,
    input logic done_error,
    input logic [7:0] done_word_count,
    input logic [2:0] selected_reason,
    input logic [SIZE_W-1:0] selected_payload_bits,
    input logic [COUNTER_W-1:0] done_sequence,
    input logic capture_active_q,
    input logic session_active_q,
    input logic session_abort_q,
    input logic emit_started_q,
    input logic emit_owner_q,
    input logic [1:0] ws_head_begin_valid,
    input logic [1:0] ws_metadata_ready,
    input logic [1:0] ws_emit_start_valid,
    input logic oldest_valid,
    input logic builder_begin_valid,
    input logic [COUNTER_W-1:0] next_capture_sequence_q,
    input logic [COUNTER_W-1:0] next_emit_sequence_q
);

    property p_done_stable_under_stall;
        @(posedge clk_core) disable iff (rst_core)
        done_valid && !done_ready |=> done_valid &&
            $stable({done_tag, done_format, done_error, done_word_count,
                     selected_reason, selected_payload_bits, done_sequence});
    endproperty

    property p_token_accept_requires_capture;
        @(posedge clk_core) disable iff (rst_core)
        token_valid && token_ready |-> capture_active_q;
    endproperty

    property p_begin_allocates_one_workspace;
        @(posedge clk_core) disable iff (rst_core)
        head_begin_valid && head_begin_ready |-> $onehot(ws_head_begin_valid);
    endproperty

    property p_control_is_onehot;
        @(posedge clk_core) disable iff (rst_core)
        $onehot0(ws_metadata_ready) && $onehot0(ws_emit_start_valid);
    endproperty

    property p_emit_requires_session;
        @(posedge clk_core) disable iff (rst_core)
        |ws_emit_start_valid |-> session_active_q;
    endproperty

    property p_builder_begin_is_ordered;
        @(posedge clk_core) disable iff (rst_core)
        builder_begin_valid |-> oldest_valid && !session_active_q;
    endproperty

    property p_last_token_releases_capture;
        @(posedge clk_core) disable iff (rst_core)
        token_valid && token_ready && token_last |=> !capture_active_q;
    endproperty

    property p_done_sequence_is_ordered;
        @(posedge clk_core) disable iff (rst_core)
        done_valid |-> done_sequence == next_emit_sequence_q;
    endproperty

    property p_sequence_window_is_bounded;
        @(posedge clk_core) disable iff (rst_core)
        next_capture_sequence_q >= next_emit_sequence_q &&
        next_capture_sequence_q - next_emit_sequence_q <= 2;
    endproperty

    property p_session_fields_stable;
        @(posedge clk_core) disable iff (rst_core)
        session_active_q && emit_started_q &&
        !(done_valid && done_ready) |=>
            !session_active_q ||
            $stable({session_abort_q, emit_started_q, emit_owner_q});
    endproperty

    assert property (p_done_stable_under_stall);
    assert property (p_token_accept_requires_capture);
    assert property (p_begin_allocates_one_workspace);
    assert property (p_control_is_onehot);
    assert property (p_emit_requires_session);
    assert property (p_builder_begin_is_ordered);
    assert property (p_last_token_releases_capture);
    assert property (p_done_sequence_is_ordered);
    assert property (p_sequence_window_is_bounded);
    assert property (p_session_fields_stable);

endmodule

`default_nettype wire
