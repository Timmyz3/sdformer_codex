`timescale 1ns/1ps
`default_nettype none

module gatestack_product_bitmap_join_assertions #(
    parameter int TOKENS = 162,
    parameter int OUT_TILE = 8,
    parameter int PRODUCT_W = 17,
    parameter int INPUT_CH_W = 10,
    parameter int OUTPUT_TILE_W = 8,
    parameter int ISSUE_SEQ_W = 13,
    parameter int TAG_W = 32
) (
    input logic clk_core,
    input logic rst_core,
    input logic joined_valid,
    input logic joined_ready,
    input logic [TAG_W-1:0] joined_tag,
    input logic [INPUT_CH_W-1:0] joined_input_channel,
    input logic [OUTPUT_TILE_W-1:0] joined_output_tile,
    input logic [ISSUE_SEQ_W-1:0] joined_issue_seq,
    input logic [TOKENS-1:0] joined_destinations,
    input logic [(OUT_TILE*PRODUCT_W)-1:0] joined_values,
    input logic protocol_error
);
    property p_joined_stable_under_stall;
        @(posedge clk_core) disable iff (rst_core)
        joined_valid && !joined_ready |=> joined_valid &&
            $stable({joined_tag, joined_input_channel, joined_output_tile,
                     joined_issue_seq, joined_destinations, joined_values});
    endproperty
    property p_joined_destination_nonzero;
        @(posedge clk_core) disable iff (rst_core)
        joined_valid |-> joined_destinations != 0;
    endproperty
    property p_protocol_error_sticky;
        @(posedge clk_core) disable iff (rst_core)
        protocol_error |=> protocol_error;
    endproperty
    assert property (p_joined_stable_under_stall);
    assert property (p_joined_destination_nonzero);
    assert property (p_protocol_error_sticky);
endmodule

`default_nettype wire
