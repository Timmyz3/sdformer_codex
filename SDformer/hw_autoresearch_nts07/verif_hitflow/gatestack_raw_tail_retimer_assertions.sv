`timescale 1ns/1ps
`default_nettype none

module gatestack_raw_tail_retimer_assertions #(
    parameter int TAG_W = 32,
    parameter int LANE_ID_W = 5,
    parameter int TOKEN_ID_W = 8
) (
    input logic clk_core,
    input logic rst_core,
    input logic output_valid,
    input logic output_ready,
    input logic [8:0] output_gate_code,
    input logic [LANE_ID_W-1:0] output_lane_id,
    input logic [TOKEN_ID_W-1:0] output_token_id,
    input logic output_head_last,
    input logic output_done_valid,
    input logic output_done_ready,
    input logic [TAG_W-1:0] output_done_tag,
    input logic output_done_error
);
    assert property (@(posedge clk_core) disable iff (rst_core)
        output_valid && !output_ready |=> output_valid &&
        $stable(output_gate_code) && $stable(output_lane_id) &&
        $stable(output_token_id) && $stable(output_head_last));
    assert property (@(posedge clk_core) disable iff (rst_core)
        output_done_valid && !output_done_ready |=> output_done_valid &&
        $stable(output_done_tag) && $stable(output_done_error));
    assert property (@(posedge clk_core) disable iff (rst_core)
        output_head_last |-> output_valid);
endmodule

`default_nettype wire
