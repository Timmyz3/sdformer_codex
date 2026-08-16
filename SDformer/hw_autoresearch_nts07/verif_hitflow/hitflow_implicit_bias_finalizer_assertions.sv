`timescale 1ns/1ps
`default_nettype none

module hitflow_implicit_bias_finalizer_assertions #(
    parameter int BANKS = 2,
    parameter int TOKEN_ID_W = 8,
    parameter int OUT_TILE = 8,
    parameter int ACC_W = 32,
    parameter int TAG_W = 32
) (
    input logic clk_core,
    input logic rst_core,
    input logic flush,
    input logic [BANKS-1:0] final_valid,
    input logic [BANKS-1:0] final_ready,
    input logic [(BANKS*TOKEN_ID_W)-1:0] final_token_ids,
    input logic [TAG_W-1:0] final_tag,
    input logic [(BANKS*OUT_TILE*ACC_W)-1:0] final_values,
    input logic finalize_done_valid,
    input logic finalize_done_ready,
    input logic [TAG_W-1:0] finalize_done_tag
);
    for (genvar bank = 0; bank < BANKS; bank = bank + 1) begin : g_stable
        assert property (@(posedge clk_core)
            disable iff (rst_core || flush)
            final_valid[bank] && !final_ready[bank] |=>
            final_valid[bank] &&
            $stable(final_token_ids[(bank*TOKEN_ID_W) +: TOKEN_ID_W]) &&
            $stable(final_values[(bank*OUT_TILE*ACC_W) +:
                                 (OUT_TILE*ACC_W)]) &&
            $stable(final_tag));
    end
    assert property (@(posedge clk_core)
        disable iff (rst_core || flush)
        finalize_done_valid && !finalize_done_ready |=>
        finalize_done_valid && $stable(finalize_done_tag));
endmodule

`default_nettype wire
