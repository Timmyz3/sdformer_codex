`timescale 1ns/1ps
`default_nettype none

module h67_tare_score_pair_assertions #(
    parameter int HEAD_DIM = 32,
    parameter int RESIDUAL_W = 16,
    parameter int TAG_W = 8,
    parameter int SCORE_W = 16
) (
    input logic clk_core,
    input logic rst_core,
    input logic window_start,
    input logic in_valid,
    input logic in_enable,
    input logic in_ready,
    input logic [TAG_W-1:0] in_tag,
    input logic [2*HEAD_DIM-1:0] in_q_pair,
    input logic [2*HEAD_DIM-1:0] in_k_pair,
    input logic out_valid,
    input logic out_ready,
    input logic [TAG_W-1:0] out_tag,
    input logic signed [SCORE_W-1:0] out_score0_q7,
    input logic signed [SCORE_W-1:0] out_score1_q7,
    input logic [1:0] out_k_active,
    input logic [5:0] out_update_count,
    input logic out_dense_fallback,
    input logic signed [12:0] out_delta_raw16,
    input logic protocol_error
);
    assert property (@(posedge clk_core) disable iff (rst_core || window_start)
        out_valid && !out_ready |=>
        out_valid && $stable(out_tag) && $stable(out_score0_q7)
        && $stable(out_score1_q7) && $stable(out_k_active)
        && $stable(out_update_count) && $stable(out_dense_fallback)
        && $stable(out_delta_raw16));

    assert property (@(posedge clk_core) disable iff (rst_core || window_start)
        in_valid && !in_ready |=>
        in_valid && $stable(in_tag) && $stable(in_q_pair) && $stable(in_k_pair));

    assert property (@(posedge clk_core) disable iff (rst_core || window_start)
        !in_enable |-> !in_ready);

    assert property (@(posedge clk_core) disable iff (rst_core || window_start)
        out_valid |-> out_score0_q7 >= 0 && out_score0_q7 <= 162
                  && out_score1_q7 >= 0 && out_score1_q7 <= 162
                  && out_update_count <= 32);

    assert property (@(posedge clk_core) disable iff (rst_core || window_start)
        out_valid && out_dense_fallback |-> out_update_count > RESIDUAL_W);

    assert property (@(posedge clk_core) disable iff (rst_core || window_start)
        out_valid && !out_dense_fallback |-> out_update_count <= RESIDUAL_W);

    assert property (@(posedge clk_core) disable iff (rst_core || window_start)
        in_valid && in_ready && (out_update_count > RESIDUAL_W) |=>
        out_valid && out_dense_fallback);

    assert property (@(posedge clk_core) disable iff (rst_core || window_start)
        !protocol_error);
endmodule

bind h67_tare_score_pair h67_tare_score_pair_assertions #(
    .HEAD_DIM(HEAD_DIM),
    .RESIDUAL_W(RESIDUAL_W),
    .TAG_W(TAG_W),
    .SCORE_W(SCORE_W)
) u_h67_tare_score_pair_assertions (.*);

`default_nettype wire
