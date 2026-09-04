`timescale 1ns/1ps
`default_nettype none

bind hitflow_gate_product_engine hitflow_gate_product_assertions #(
    .TOKENS(TOKENS), .PRODUCT_W(PRODUCT_W),
    .OUT_TILE(OUT_TILE), .INPUT_CH_W(INPUT_CH_W),
    .OUTPUT_TILE_W(OUTPUT_TILE_W), .TAG_W(TAG_W)
) u_hitflow_gate_product_assertions (
    .clk_core(clk_core), .rst_core(rst_core), .protocol_error(protocol_error),
    .term_valid(term_valid), .term_ready(term_ready),
    .weight_req_valid(weight_req_valid), .weight_req_ready(weight_req_ready),
    .weight_req_tag(weight_req_tag),
    .weight_req_input_channel(weight_req_input_channel),
    .weight_req_output_tile(weight_req_output_tile),
    .product_valid(product_valid), .product_ready(product_ready),
    .product_tag(product_tag), .product_input_channel(product_input_channel),
    .product_output_tile(product_output_tile),
    .product_destination_bitmap(product_destination_bitmap),
    .product_values(product_values)
);

`default_nettype wire
