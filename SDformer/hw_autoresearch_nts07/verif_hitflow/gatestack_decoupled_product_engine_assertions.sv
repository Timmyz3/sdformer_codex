`timescale 1ns/1ps
`default_nettype none

module gatestack_decoupled_product_engine_assertions #(
    parameter int OUT_TILE = 8,
    parameter int PRODUCT_W = 17,
    parameter int INPUT_CH_W = 10,
    parameter int OUTPUT_TILE_W = 8,
    parameter int ISSUE_SEQ_W = 13,
    parameter int TAG_W = 32
) (
    input logic clk_core,
    input logic rst_core,
    input logic clear_error,
    input logic weight_req_valid,
    input logic weight_req_ready,
    input logic [TAG_W-1:0] weight_req_tag,
    input logic [INPUT_CH_W-1:0] weight_req_input_channel,
    input logic [OUTPUT_TILE_W-1:0] weight_req_output_tile,
    input logic product_valid,
    input logic product_ready,
    input logic [TAG_W-1:0] product_tag,
    input logic [INPUT_CH_W-1:0] product_input_channel,
    input logic [OUTPUT_TILE_W-1:0] product_output_tile,
    input logic [ISSUE_SEQ_W-1:0] product_issue_seq,
    input logic [(OUT_TILE*PRODUCT_W)-1:0] product_values,
    input logic protocol_error
);
    property p_request_stable_under_stall;
        @(posedge clk_core) disable iff (rst_core)
        weight_req_valid && !weight_req_ready |=> weight_req_valid &&
            $stable({weight_req_tag, weight_req_input_channel,
                     weight_req_output_tile});
    endproperty
    property p_product_stable_under_stall;
        @(posedge clk_core) disable iff (rst_core)
        product_valid && !product_ready |=> product_valid &&
            $stable({product_tag, product_input_channel,
                     product_output_tile, product_issue_seq, product_values});
    endproperty
    property p_protocol_error_sticky;
        @(posedge clk_core) disable iff (rst_core)
        protocol_error && !clear_error |=> clear_error || protocol_error;
    endproperty
    property p_clear_error_clears_sticky;
        @(posedge clk_core) disable iff (rst_core)
        clear_error |=> !protocol_error;
    endproperty
    assert property (p_request_stable_under_stall);
    assert property (p_product_stable_under_stall);
    assert property (p_protocol_error_sticky);
    assert property (p_clear_error_clears_sticky);
endmodule

`default_nettype wire
