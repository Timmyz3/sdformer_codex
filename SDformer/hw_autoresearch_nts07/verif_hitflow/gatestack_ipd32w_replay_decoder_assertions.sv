`timescale 1ns/1ps
`default_nettype none

module gatestack_ipd32w_replay_decoder_assertions #(
    parameter int EVENT_WAYS = 4,
    parameter int TAG_W = 32,
    parameter int TOKEN_ID_W = 8,
    parameter int LANE_ID_W = 5,
    parameter int WAY_COUNT_W = $clog2(EVENT_WAYS + 1)
) (
    input logic clk_core,
    input logic rst_core,
    input logic term_valid,
    input logic term_ready,
    input logic [8:0] term_gate_code,
    input logic [LANE_ID_W-1:0] term_lane_id,
    input logic [7:0] term_destination_count,
    input logic term_head_last,
    input logic event_valid,
    input logic event_ready,
    input logic [8:0] event_gate_code,
    input logic [LANE_ID_W-1:0] event_lane_id,
    input logic [EVENT_WAYS-1:0] event_token_valid,
    input logic [(EVENT_WAYS*TOKEN_ID_W)-1:0] event_token_ids,
    input logic [WAY_COUNT_W-1:0] event_count,
    input logic event_term_first,
    input logic event_term_last,
    input logic event_head_last,
    input logic done_valid,
    input logic done_ready,
    input logic [TAG_W-1:0] done_tag,
    input logic done_error,
    input logic protocol_error
);

    property p_event_stable_under_stall;
        @(posedge clk_core) disable iff (rst_core)
        event_valid && !event_ready |=> event_valid &&
            $stable({event_gate_code, event_lane_id, event_token_valid,
                     event_token_ids, event_count, event_term_first,
                     event_term_last, event_head_last});
    endproperty

    property p_term_stable_under_stall;
        @(posedge clk_core) disable iff (rst_core)
        term_valid && !term_ready |=> term_valid &&
            $stable({term_gate_code, term_lane_id,
                     term_destination_count, term_head_last});
    endproperty

    property p_done_stable_under_stall;
        @(posedge clk_core) disable iff (rst_core)
        done_valid && !done_ready |=> done_valid &&
            $stable({done_tag, done_error});
    endproperty

    property p_valid_event_has_count;
        @(posedge clk_core) disable iff (rst_core)
        event_valid |-> event_count > 0;
    endproperty

    property p_head_last_implies_term_last;
        @(posedge clk_core) disable iff (rst_core)
        event_valid && event_head_last |-> event_term_last;
    endproperty

    property p_protocol_error_sticky;
        @(posedge clk_core) disable iff (rst_core)
        protocol_error |=> protocol_error;
    endproperty

    assert property (p_event_stable_under_stall);
    assert property (p_term_stable_under_stall);
    assert property (p_done_stable_under_stall);
    assert property (p_valid_event_has_count);
    assert property (p_head_last_implies_term_last);
    assert property (p_protocol_error_sticky);

    for (genvar way = 1; way < EVENT_WAYS; way = way + 1) begin : g_prefix
        property p_valid_mask_is_prefix;
            @(posedge clk_core) disable iff (rst_core)
            event_valid && event_token_valid[way] |->
                event_token_valid[way-1];
        endproperty
        assert property (p_valid_mask_is_prefix);
    end

endmodule

`default_nettype wire
