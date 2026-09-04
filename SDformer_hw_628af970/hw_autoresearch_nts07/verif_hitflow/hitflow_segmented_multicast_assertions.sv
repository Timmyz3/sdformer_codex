`timescale 1ns/1ps
`default_nettype none

module hitflow_segmented_multicast_assertions #(
    parameter int TOKENS       = 162,
    parameter int BANKS        = 4,
    parameter int PRODUCT_W    = 17,
    parameter int OUT_TILE     = 8,
    parameter int TAG_W        = 32,
    parameter int TOKEN_ID_W   = (TOKENS <= 1) ? 1 : $clog2(TOKENS)
) (
    input logic                            clk_core,
    input logic                            rst_core,
    input logic                            protocol_error,
    input logic                            product_valid,
    input logic                            product_ready,
    input logic [BANKS-1:0]                update_valid,
    input logic [BANKS-1:0]                update_ready,
    input logic [(BANKS*TOKEN_ID_W)-1:0]   update_token_ids,
    input logic [TAG_W-1:0]                update_tag,
    input logic [(OUT_TILE*PRODUCT_W)-1:0] update_values,
    input logic                            product_done_valid,
    input logic                            product_done_ready,
    input logic [TAG_W-1:0]                product_done_tag
);

    for (genvar bank = 0; bank < BANKS; bank = bank + 1) begin : g_bank_assertions
        property p_stalled_update_is_stable;
            @(posedge clk_core) disable iff (rst_core)
                update_valid[bank] && !update_ready[bank] |=>
                update_valid[bank] && $stable(update_tag) &&
                $stable(update_values) &&
                $stable(update_token_ids[(bank*TOKEN_ID_W) +: TOKEN_ID_W]);
        endproperty

        property p_update_matches_bank;
            @(posedge clk_core) disable iff (rst_core)
                update_valid[bank] |->
                (32'(update_token_ids[(bank*TOKEN_ID_W) +: TOKEN_ID_W]) %
                 BANKS) == bank;
        endproperty

        assert property (p_stalled_update_is_stable);
        assert property (p_update_matches_bank);
    end

    property p_done_is_stable;
        @(posedge clk_core) disable iff (rst_core)
            product_done_valid && !product_done_ready |=>
            product_done_valid && $stable(product_done_tag);
    endproperty

    property p_invalid_product_is_rejected;
        @(posedge clk_core) disable iff (rst_core)
            protocol_error && product_valid |-> !product_ready;
    endproperty

    assert property (p_done_is_stable);
    assert property (p_invalid_product_is_rejected);

endmodule

`default_nettype wire
