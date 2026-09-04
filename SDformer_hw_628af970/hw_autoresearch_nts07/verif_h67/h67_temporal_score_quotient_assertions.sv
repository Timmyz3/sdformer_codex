`timescale 1ns/1ps
`default_nettype none

module h67_temporal_score_quotient_assertions #(
    parameter int SCORE_W = 16,
    parameter int PAIR_ID_W = 9
) (
    input logic clk_core,
    input logic rst_core,
    input logic out_valid,
    input logic out_ready,
    input logic [PAIR_ID_W-1:0] out_pair_id,
    input logic signed [SCORE_W-1:0] out_score_q7,
    input logic [1:0] out_temporal_mask,
    input logic [1:0] out_active_mask,
    input logic out_last
);
    assert property (@(posedge clk_core) disable iff (rst_core)
        out_valid && !out_ready
        |=> $stable({out_pair_id, out_score_q7, out_temporal_mask, out_active_mask, out_last})
    );
    assert property (@(posedge clk_core) disable iff (rst_core)
        out_valid |-> out_temporal_mask inside {2'b01, 2'b10, 2'b11}
    );
    assert property (@(posedge clk_core) disable iff (rst_core)
        out_valid |-> (out_active_mask & ~out_temporal_mask) == 0
    );
    assert property (@(posedge clk_core) disable iff (rst_core)
        out_valid |-> out_last == (out_temporal_mask != 2'b01)
    );
endmodule

bind h67_temporal_score_quotient
    h67_temporal_score_quotient_assertions #(
        .SCORE_W(SCORE_W),
        .PAIR_ID_W(PAIR_ID_W)
    ) u_h67_temporal_score_quotient_assertions (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .out_valid(out_valid),
        .out_ready(out_ready),
        .out_pair_id(out_pair_id),
        .out_score_q7(out_score_q7),
        .out_temporal_mask(out_temporal_mask),
        .out_active_mask(out_active_mask),
        .out_last(out_last)
    );

`default_nettype wire
