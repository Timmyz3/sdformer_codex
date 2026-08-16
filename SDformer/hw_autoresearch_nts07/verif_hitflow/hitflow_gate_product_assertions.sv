`timescale 1ns/1ps
`default_nettype none

module hitflow_gate_product_assertions #(
    parameter int TOKENS = 162,
    parameter int PRODUCT_W = 17,
    parameter int OUT_TILE = 8,
    parameter int INPUT_CH_W = 10,
    parameter int OUTPUT_TILE_W = 8,
    parameter int TAG_W = 32
) (
    input logic                            clk_core,
    input logic                            rst_core,
    input logic                            protocol_error,
    input logic                            term_valid,
    input logic                            term_ready,
    input logic                            weight_req_valid,
    input logic                            weight_req_ready,
    input logic [TAG_W-1:0]                weight_req_tag,
    input logic [INPUT_CH_W-1:0]           weight_req_input_channel,
    input logic [OUTPUT_TILE_W-1:0]        weight_req_output_tile,
    input logic                            product_valid,
    input logic                            product_ready,
    input logic [TAG_W-1:0]                product_tag,
    input logic [INPUT_CH_W-1:0]           product_input_channel,
    input logic [OUTPUT_TILE_W-1:0]        product_output_tile,
    input logic [TOKENS-1:0]               product_destination_bitmap,
    input logic [(OUT_TILE*PRODUCT_W)-1:0] product_values
);

    property p_weight_request_stable;
        @(posedge clk_core) disable iff (rst_core)
            weight_req_valid && !weight_req_ready |=> weight_req_valid &&
            $stable(weight_req_tag) && $stable(weight_req_input_channel) &&
            $stable(weight_req_output_tile);
    endproperty

    property p_product_stable;
        @(posedge clk_core) disable iff (rst_core)
            product_valid && !product_ready |=> product_valid &&
            $stable(product_tag) && $stable(product_input_channel) &&
            $stable(product_output_tile) &&
            $stable(product_destination_bitmap) && $stable(product_values);
    endproperty

    property p_product_has_destination;
        @(posedge clk_core) disable iff (rst_core)
            product_valid |-> (product_destination_bitmap != '0);
    endproperty

    property p_invalid_term_is_rejected;
        @(posedge clk_core) disable iff (rst_core)
            protocol_error && term_valid |-> !term_ready;
    endproperty

    assert property (p_weight_request_stable);
    assert property (p_product_stable);
    assert property (p_product_has_destination);
    assert property (p_invalid_term_is_rejected);

endmodule

`default_nettype wire
