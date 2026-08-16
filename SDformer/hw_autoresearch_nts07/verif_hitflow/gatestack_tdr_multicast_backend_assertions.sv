`timescale 1ns/1ps
`default_nettype none

module gatestack_tdr_multicast_backend_assertions #(
    parameter int TAG_W = 32,
    parameter int BANKS = 2,
    parameter int TOKEN_ID_W = 8,
    parameter int OUT_TILE = 8,
    parameter int PRODUCT_W = 17,
    parameter int OUTSTANDING_W = 14
) (
    input logic clk_core,
    input logic rst_core,
    input logic session_start_valid,
    input logic session_start_ready,
    input logic term_valid,
    input logic term_ready,
    input logic source_done_valid,
    input logic source_done_ready,
    input logic [BANKS-1:0] update_valid,
    input logic [BANKS-1:0] update_ready,
    input logic [(BANKS*TOKEN_ID_W)-1:0] update_token_ids,
    input logic [TAG_W-1:0] update_tag,
    input logic [(OUT_TILE*PRODUCT_W)-1:0] update_values,
    input logic backend_done_valid,
    input logic backend_done_ready,
    input logic [TAG_W-1:0] backend_done_tag,
    input logic [OUTSTANDING_W-1:0] outstanding_terms
);
    assert property (@(posedge clk_core) disable iff (rst_core)
        session_start_valid && !session_start_ready |=>
        $stable(session_start_valid));
    assert property (@(posedge clk_core) disable iff (rst_core)
        term_valid && !term_ready |=> $stable(term_valid));
    assert property (@(posedge clk_core) disable iff (rst_core)
        source_done_valid && !source_done_ready |=>
        $stable(source_done_valid));
    assert property (@(posedge clk_core) disable iff (rst_core)
        backend_done_valid |-> outstanding_terms == 0);
    assert property (@(posedge clk_core) disable iff (rst_core)
        backend_done_valid && !backend_done_ready |=>
        backend_done_valid && $stable(backend_done_tag));
    for (genvar bank = 0; bank < BANKS; bank = bank + 1) begin : g_bank_stall
        assert property (@(posedge clk_core) disable iff (rst_core)
            update_valid[bank] && !update_ready[bank] |=>
            $stable(update_tag) && $stable(update_values) &&
            $stable(update_token_ids[(bank*TOKEN_ID_W) +: TOKEN_ID_W]));
    end
endmodule

`default_nettype wire
