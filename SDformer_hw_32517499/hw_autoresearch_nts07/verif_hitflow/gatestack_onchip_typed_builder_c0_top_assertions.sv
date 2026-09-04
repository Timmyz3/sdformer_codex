`timescale 1ns/1ps
`default_nettype none

module gatestack_onchip_typed_builder_c0_top_assertions #(
    parameter int TAG_W = 32,
    parameter int FORMAT_W = 2,
    parameter int SIZE_W = 16
) (
    input logic clk_core,
    input logic rst_core,
    input logic normal_active_q,
    input logic abort_active_q,
    input logic emit_started_q,
    input logic workspace_raw_capture_error,
    input logic builder_begin_valid,
    input logic workspace_emit_done_valid,
    input logic builder_done_valid,
    input logic done_valid,
    input logic done_ready,
    input logic [TAG_W-1:0] done_tag,
    input logic [FORMAT_W-1:0] done_format,
    input logic done_error,
    input logic [7:0] done_word_count,
    input logic [2:0] selected_reason,
    input logic [SIZE_W-1:0] selected_payload_bits
);

    property p_modes_are_exclusive;
        @(posedge clk_core) disable iff (rst_core)
        !(normal_active_q && abort_active_q);
    endproperty

    property p_emit_requires_active_head;
        @(posedge clk_core) disable iff (rst_core)
        emit_started_q |-> normal_active_q || abort_active_q;
    endproperty

    property p_bad_raw_never_starts_serializer;
        @(posedge clk_core) disable iff (rst_core)
        builder_begin_valid |-> !workspace_raw_capture_error;
    endproperty

    property p_normal_done_is_joined;
        @(posedge clk_core) disable iff (rst_core)
        done_valid && normal_active_q |->
            workspace_emit_done_valid && builder_done_valid;
    endproperty

    property p_abort_done_reports_error;
        @(posedge clk_core) disable iff (rst_core)
        done_valid && abort_active_q |-> done_error && done_word_count == 0;
    endproperty

    property p_done_stable_under_stall;
        @(posedge clk_core) disable iff (rst_core)
        done_valid && !done_ready |=> done_valid &&
            $stable({done_tag, done_format, done_error, done_word_count,
                     selected_reason, selected_payload_bits});
    endproperty

    property p_success_has_payload;
        @(posedge clk_core) disable iff (rst_core)
        done_valid && !done_error |-> done_word_count != 0 &&
            selected_payload_bits != 0;
    endproperty

    assert property (p_modes_are_exclusive);
    assert property (p_emit_requires_active_head);
    assert property (p_bad_raw_never_starts_serializer);
    assert property (p_normal_done_is_joined);
    assert property (p_abort_done_reports_error);
    assert property (p_done_stable_under_stall);
    assert property (p_success_has_payload);

endmodule

`default_nettype wire

