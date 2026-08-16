`timescale 1ns/1ps
`default_nettype none

module gatestack_term_fork_assertions #(
    parameter int TAG_W = 32,
    parameter int LANE_ID_W = 5,
    parameter int INPUT_CH_W = 10,
    parameter int OUTPUT_TILE_W = 8,
    parameter int ISSUE_SEQ_W = 13
) (
    input logic clk_core,
    input logic rst_core,
    input logic product_term_valid,
    input logic product_term_ready,
    input logic [TAG_W-1:0] product_term_tag,
    input logic [8:0] product_term_gate_code,
    input logic [INPUT_CH_W-1:0] product_term_input_channel,
    input logic [OUTPUT_TILE_W-1:0] product_term_output_tile,
    input logic [ISSUE_SEQ_W-1:0] product_term_issue_seq,
    input logic bitmap_term_valid,
    input logic bitmap_term_ready,
    input logic [TAG_W-1:0] bitmap_term_tag,
    input logic [8:0] bitmap_term_gate_code,
    input logic [LANE_ID_W-1:0] bitmap_term_lane_id,
    input logic [7:0] bitmap_term_destination_count,
    input logic [ISSUE_SEQ_W-1:0] bitmap_term_issue_seq,
    input logic bitmap_term_head_last
);
    property p_product_stable_under_stall;
        @(posedge clk_core) disable iff (rst_core)
        product_term_valid && !product_term_ready |=> product_term_valid &&
            $stable({product_term_tag, product_term_gate_code,
                     product_term_input_channel, product_term_output_tile,
                     product_term_issue_seq});
    endproperty
    property p_bitmap_stable_under_stall;
        @(posedge clk_core) disable iff (rst_core)
        bitmap_term_valid && !bitmap_term_ready |=> bitmap_term_valid &&
            $stable({bitmap_term_tag, bitmap_term_gate_code,
                     bitmap_term_lane_id, bitmap_term_destination_count,
                     bitmap_term_issue_seq, bitmap_term_head_last});
    endproperty
    assert property (p_product_stable_under_stall);
    assert property (p_bitmap_stable_under_stall);
endmodule

`default_nettype wire
