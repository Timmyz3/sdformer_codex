`timescale 1ns/1ps
`default_nettype none

// Converts one RAW41 direct event into the common ordered term/event protocol.
// The term command must handshake before its single destination event.
module gatestack_raw_issue_adapter #(
    parameter int EVENT_WAYS      = 4,
    parameter int TOKEN_ID_W      = 8,
    parameter int LANE_ID_W       = 5,
    parameter int ISSUE_SEQ_W     = 13,
    parameter int WAY_COUNT_W     = $clog2(EVENT_WAYS + 1),
    parameter int COUNTER_W       = 32
) (
    input  logic                               clk_core,
    input  logic                               rst_core,

    input  logic                               direct_valid,
    output logic                               direct_ready,
    input  logic [8:0]                         direct_gate_code,
    input  logic [LANE_ID_W-1:0]               direct_lane_id,
    input  logic [TOKEN_ID_W-1:0]              direct_token_id,
    input  logic                               direct_head_last,

    output logic                               term_valid,
    input  logic                               term_ready,
    output logic [8:0]                         term_gate_code,
    output logic [LANE_ID_W-1:0]               term_lane_id,
    output logic [7:0]                         term_destination_count,
    output logic [ISSUE_SEQ_W-1:0]             term_issue_seq,
    output logic                               term_head_last,

    output logic                               event_valid,
    input  logic                               event_ready,
    output logic [8:0]                         event_gate_code,
    output logic [LANE_ID_W-1:0]               event_lane_id,
    output logic [EVENT_WAYS-1:0]              event_token_valid,
    output logic [(EVENT_WAYS*TOKEN_ID_W)-1:0] event_token_ids,
    output logic [WAY_COUNT_W-1:0]             event_count,
    output logic [ISSUE_SEQ_W-1:0]             event_issue_seq,
    output logic                               event_term_first,
    output logic                               event_term_last,
    output logic                               event_head_last,

    output logic [COUNTER_W-1:0]               count_direct_inputs,
    output logic [COUNTER_W-1:0]               count_term_stall_cycles,
    output logic [COUNTER_W-1:0]               count_event_stall_cycles
);

    logic buffer_valid_q;
    logic term_accepted_q;
    logic [8:0] gate_q;
    logic [LANE_ID_W-1:0] lane_q;
    logic [TOKEN_ID_W-1:0] token_q;
    logic head_last_q;
    logic [ISSUE_SEQ_W-1:0] issue_seq_q;
    logic [ISSUE_SEQ_W-1:0] next_issue_seq_q;
    logic direct_fire;
    logic term_fire;
    logic event_fire;

    assign direct_ready = !buffer_valid_q;
    assign direct_fire = direct_valid && direct_ready;

    assign term_valid = buffer_valid_q && !term_accepted_q;
    assign term_gate_code = gate_q;
    assign term_lane_id = lane_q;
    assign term_destination_count = 8'd1;
    assign term_issue_seq = issue_seq_q;
    assign term_head_last = head_last_q;
    assign term_fire = term_valid && term_ready;

    assign event_valid = buffer_valid_q && term_accepted_q;
    assign event_gate_code = gate_q;
    assign event_lane_id = lane_q;
    assign event_token_valid = {{(EVENT_WAYS-1){1'b0}}, 1'b1};
    assign event_token_ids = {{((EVENT_WAYS-1)*TOKEN_ID_W){1'b0}}, token_q};
    assign event_count = WAY_COUNT_W'(1);
    assign event_issue_seq = issue_seq_q;
    assign event_term_first = 1'b1;
    assign event_term_last = 1'b1;
    assign event_head_last = head_last_q;
    assign event_fire = event_valid && event_ready;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            buffer_valid_q <= 1'b0;
            term_accepted_q <= 1'b0;
            gate_q <= '0;
            lane_q <= '0;
            token_q <= '0;
            head_last_q <= 1'b0;
            issue_seq_q <= '0;
            next_issue_seq_q <= '0;
            count_direct_inputs <= '0;
            count_term_stall_cycles <= '0;
            count_event_stall_cycles <= '0;
        end else begin
            if (direct_fire) begin
                buffer_valid_q <= 1'b1;
                term_accepted_q <= 1'b0;
                gate_q <= direct_gate_code;
                lane_q <= direct_lane_id;
                token_q <= direct_token_id;
                head_last_q <= direct_head_last;
                issue_seq_q <= next_issue_seq_q;
                next_issue_seq_q <= next_issue_seq_q + 1'b1;
                count_direct_inputs <= count_direct_inputs + 1'b1;
            end
            if (term_fire) begin
                term_accepted_q <= 1'b1;
            end
            if (event_fire) begin
                buffer_valid_q <= 1'b0;
                term_accepted_q <= 1'b0;
            end
            if (term_valid && !term_ready) begin
                count_term_stall_cycles <= count_term_stall_cycles + 1'b1;
            end
            if (event_valid && !event_ready) begin
                count_event_stall_cycles <= count_event_stall_cycles + 1'b1;
            end
        end
    end

endmodule

`default_nettype wire
