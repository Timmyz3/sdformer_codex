`timescale 1ns/1ps
`default_nettype none

module gatestack_replay_mux_assertions #(
    parameter int SOURCES = 3,
    parameter int EVENT_WAYS = 4,
    parameter int TOKEN_ID_W = 8,
    parameter int LANE_ID_W = 5,
    parameter int ISSUE_SEQ_W = 13,
    parameter int TAG_W = 32,
    parameter int WAY_COUNT_W = $clog2(EVENT_WAYS + 1),
    parameter int ROUTE_W = (SOURCES <= 1) ? 1 : $clog2(SOURCES)
) (
    input logic clk_core,
    input logic rst_core,
    input logic route_active,
    input logic [ROUTE_W-1:0] route_active_select,
    input logic [SOURCES-1:0] source_term_ready,
    input logic [SOURCES-1:0] source_event_ready,
    input logic [SOURCES-1:0] source_done_ready,
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
    input logic event_head_last,
    input logic done_valid,
    input logic done_ready,
    input logic [TAG_W-1:0] done_tag,
    input logic done_error,
    input logic protocol_error
);
    property p_route_stable_while_active;
        @(posedge clk_core) disable iff (rst_core)
        route_active && $past(route_active) |-> $stable(route_active_select);
    endproperty
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
    property p_done_stable_under_stall;
        @(posedge clk_core) disable iff (rst_core)
        done_valid && !done_ready |=> done_valid &&
            $stable({done_tag, done_error});
    endproperty
    property p_at_most_one_term_ready;
        @(posedge clk_core) disable iff (rst_core)
        $onehot0(source_term_ready);
    endproperty
    property p_at_most_one_event_ready;
        @(posedge clk_core) disable iff (rst_core)
        $onehot0(source_event_ready);
    endproperty
    property p_at_most_one_done_ready;
        @(posedge clk_core) disable iff (rst_core)
        $onehot0(source_done_ready);
    endproperty
    property p_protocol_error_sticky;
        @(posedge clk_core) disable iff (rst_core)
        protocol_error |=> protocol_error;
    endproperty
    assert property (p_route_stable_while_active);
    assert property (p_term_stable_under_stall);
    assert property (p_event_stable_under_stall);
    assert property (p_done_stable_under_stall);
    assert property (p_at_most_one_term_ready);
    assert property (p_at_most_one_event_ready);
    assert property (p_at_most_one_done_ready);
    assert property (p_protocol_error_sticky);
endmodule

`default_nettype wire
