`timescale 1ns/1ps
`default_nettype none

bind hitflow_segmented_multicast hitflow_segmented_multicast_assertions #(
    .TOKENS(TOKENS), .BANKS(BANKS), .PRODUCT_W(PRODUCT_W),
    .OUT_TILE(OUT_TILE), .TAG_W(TAG_W), .TOKEN_ID_W(TOKEN_ID_W)
) u_hitflow_segmented_multicast_assertions (
    .clk_core(clk_core), .rst_core(rst_core), .protocol_error(protocol_error),
    .product_valid(product_valid), .product_ready(product_ready),
    .update_valid(update_valid), .update_ready(update_ready),
    .update_token_ids(update_token_ids), .update_tag(update_tag),
    .update_values(update_values), .product_done_valid(product_done_valid),
    .product_done_ready(product_done_ready), .product_done_tag(product_done_tag)
);

`default_nettype wire
