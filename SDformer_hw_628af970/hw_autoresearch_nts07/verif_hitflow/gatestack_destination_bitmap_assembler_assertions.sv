`timescale 1ns/1ps
`default_nettype none

module gatestack_destination_bitmap_assembler_assertions #(
    parameter int TOKENS = 162,
    parameter int LANE_ID_W = 5,
    parameter int ISSUE_SEQ_W = 13,
    parameter int TAG_W = 32
) (
    input logic clk_core,
    input logic rst_core,
    input logic bitmap_valid,
    input logic bitmap_ready,
    input logic [TAG_W-1:0] bitmap_tag,
    input logic [8:0] bitmap_gate_code,
    input logic [LANE_ID_W-1:0] bitmap_lane_id,
    input logic [ISSUE_SEQ_W-1:0] bitmap_issue_seq,
    input logic bitmap_head_last,
    input logic [TOKENS-1:0] bitmap_destinations,
    input logic protocol_error
);
    property p_bitmap_stable_under_stall;
        @(posedge clk_core) disable iff (rst_core)
        bitmap_valid && !bitmap_ready |=> bitmap_valid &&
            $stable({bitmap_tag, bitmap_gate_code, bitmap_lane_id,
                     bitmap_issue_seq,
                     bitmap_head_last, bitmap_destinations});
    endproperty
    property p_bitmap_nonempty;
        @(posedge clk_core) disable iff (rst_core)
        bitmap_valid |-> bitmap_destinations != 0;
    endproperty
    property p_protocol_error_sticky;
        @(posedge clk_core) disable iff (rst_core)
        protocol_error |=> protocol_error;
    endproperty
    assert property (p_bitmap_stable_under_stall);
    assert property (p_bitmap_nonempty);
    assert property (p_protocol_error_sticky);
endmodule

`default_nettype wire
