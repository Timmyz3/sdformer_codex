`timescale 1ns/1ps
`default_nettype none

module gatestack_raw_issue_adapter_assertions #(
    parameter int EVENT_WAYS = 4,
    parameter int TOKEN_ID_W = 8,
    parameter int LANE_ID_W = 5,
    parameter int ISSUE_SEQ_W = 13,
    parameter int WAY_COUNT_W = $clog2(EVENT_WAYS + 1)
) (
    input logic clk_core,
    input logic rst_core,
    input logic term_valid,
    input logic term_ready,
    input logic [8:0] term_gate_code,
    input logic [LANE_ID_W-1:0] term_lane_id,
    input logic [7:0] term_destination_count,
    input logic [ISSUE_SEQ_W-1:0] term_issue_seq,
    input logic term_head_last,
    input logic event_valid,
    input logic event_ready,
    input logic [8:0] event_gate_code,
    input logic [LANE_ID_W-1:0] event_lane_id,
    input logic [EVENT_WAYS-1:0] event_token_valid,
    input logic [(EVENT_WAYS*TOKEN_ID_W)-1:0] event_token_ids,
    input logic [WAY_COUNT_W-1:0] event_count,
    input logic [ISSUE_SEQ_W-1:0] event_issue_seq,
    input logic event_term_first,
    input logic event_term_last,
    input logic event_head_last
);
    property p_term_stable_under_stall;
        @(posedge clk_core) disable iff (rst_core)
        term_valid && !term_ready |=> term_valid &&
            $stable({term_gate_code, term_lane_id, term_destination_count,
                     term_issue_seq, term_head_last});
    endproperty
    property p_event_stable_under_stall;
        @(posedge clk_core) disable iff (rst_core)
        event_valid && !event_ready |=> event_valid &&
            $stable({event_gate_code, event_lane_id, event_token_valid,
                     event_token_ids, event_count, event_issue_seq,
                     event_term_first, event_term_last, event_head_last});
    endproperty
    property p_event_is_single_destination;
        @(posedge clk_core) disable iff (rst_core)
        event_valid |-> event_count == 1 &&
            event_token_valid == {{(EVENT_WAYS-1){1'b0}}, 1'b1} &&
            event_term_first && event_term_last;
    endproperty
    assert property (p_term_stable_under_stall);
    assert property (p_event_stable_under_stall);
    assert property (p_event_is_single_destination);
endmodule

`default_nettype wire
