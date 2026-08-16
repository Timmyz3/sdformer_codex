`timescale 1ns/1ps
`default_nettype none

module gatestack_event_compactor_assertions #(
    parameter int WAYS = 4,
    parameter int TAG_W = 32,
    parameter int TOKEN_ID_W = 8,
    parameter int SLOT_ID_W = 2,
    parameter int LANE_ID_W = 5,
    parameter int COUNT_W = 3
) (
    input logic clk_core,
    input logic rst_core,
    input logic event_valid,
    input logic event_ready,
    input logic [TAG_W-1:0] event_tag,
    input logic [TOKEN_ID_W-1:0] event_token_id,
    input logic [SLOT_ID_W-1:0] event_slot_id,
    input logic [WAYS-1:0] event_lane_valid,
    input logic [(WAYS*LANE_ID_W)-1:0] event_lane_ids,
    input logic [COUNT_W-1:0] event_count,
    input logic event_last_for_token
);

    property p_event_stable_under_stall;
        @(posedge clk_core) disable iff (rst_core)
        event_valid && !event_ready |=> event_valid &&
            $stable({event_tag, event_token_id, event_slot_id,
                     event_lane_valid, event_lane_ids, event_count,
                     event_last_for_token});
    endproperty

    property p_count_in_range;
        @(posedge clk_core) disable iff (rst_core)
        event_valid |-> (event_count > 0) && (32'(event_count) <= WAYS);
    endproperty

    property p_valid_mask_matches_count;
        @(posedge clk_core) disable iff (rst_core)
        event_valid |-> ($countones(event_lane_valid) == event_count);
    endproperty

    assert property (p_event_stable_under_stall);
    assert property (p_count_in_range);
    assert property (p_valid_mask_matches_count);

endmodule

`default_nettype wire
