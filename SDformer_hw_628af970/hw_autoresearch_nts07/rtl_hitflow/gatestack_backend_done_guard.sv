`timescale 1ns/1ps
`default_nettype none

// Preserves scheduler-visible head identity while the projection backend runs.
module gatestack_backend_done_guard #(
    parameter int TAG_W = 32,
    parameter int HEAD_COUNT_W = 6
) (
    input  logic                         clk_core,
    input  logic                         rst_core,
    input  logic                         start_valid,
    output logic                         start_ready,
    input  logic [TAG_W-1:0]             start_execution_tag,
    input  logic [HEAD_COUNT_W-1:0]      start_head_index,
    input  logic                         start_last_head,
    input  logic                         backend_done_valid,
    output logic                         backend_done_ready,
    input  logic [TAG_W-1:0]             backend_done_execution_tag,
    input  logic [HEAD_COUNT_W-1:0]      backend_done_head_index,
    input  logic                         backend_done_last_head,
    input  logic                         backend_done_error,
    output logic                         checked_done_valid,
    input  logic                         checked_done_ready,
    output logic [TAG_W-1:0]             checked_done_execution_tag,
    output logic                         checked_done_error,
    output logic                         session_active,
    output logic                         protocol_error
);
    logic [TAG_W-1:0] execution_tag_q;
    logic [HEAD_COUNT_W-1:0] head_index_q;
    logic last_head_q;
    logic start_fire, done_fire, done_mismatch;

    assign start_ready = !session_active;
    assign start_fire = start_valid && start_ready;
    assign checked_done_valid = session_active && backend_done_valid;
    assign backend_done_ready = session_active && checked_done_ready;
    assign checked_done_execution_tag = backend_done_execution_tag;
    assign done_mismatch = backend_done_execution_tag != execution_tag_q ||
                           backend_done_head_index != head_index_q ||
                           backend_done_last_head != last_head_q;
    assign checked_done_error = backend_done_error || done_mismatch;
    assign done_fire = checked_done_valid && checked_done_ready;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            execution_tag_q <= '0;
            head_index_q <= '0;
            last_head_q <= 1'b0;
            session_active <= 1'b0;
            protocol_error <= 1'b0;
        end else begin
            if (start_valid && !start_ready)
                protocol_error <= 1'b1;
            if (start_fire) begin
                execution_tag_q <= start_execution_tag;
                head_index_q <= start_head_index;
                last_head_q <= start_last_head;
                session_active <= 1'b1;
            end
            if (done_fire) begin
                if (done_mismatch || backend_done_error)
                    protocol_error <= 1'b1;
                session_active <= 1'b0;
            end
        end
    end
endmodule

`default_nettype wire
