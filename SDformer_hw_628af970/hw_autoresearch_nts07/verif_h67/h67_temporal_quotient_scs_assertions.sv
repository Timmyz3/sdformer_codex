`timescale 1ns/1ps
`default_nettype none

module h67_temporal_quotient_scs_assertions #(
    parameter int SCORE_W = 16,
    parameter int PAIR_ID_W = 9,
    parameter int COUNT_W = 9,
    parameter int CLASS_W = 8
) (
    input logic clk_core,
    input logic rst_core,
    input logic class_valid,
    input logic class_ready,
    input logic [CLASS_W-1:0] class_score,
    input logic [COUNT_W-1:0] class_multiplicity,
    input logic class_last,
    input logic active_valid,
    input logic active_ready,
    input logic [PAIR_ID_W-1:0] active_pair_id,
    input logic signed [SCORE_W-1:0] active_score_q7,
    input logic [1:0] active_temporal_mask,
    input logic [1:0] active_k_mask,
    input logic active_last,
    input logic protocol_error
);
    assert property (@(posedge clk_core) disable iff (rst_core)
        class_valid && !class_ready
        |=> $stable({class_score, class_multiplicity, class_last})
    );
    assert property (@(posedge clk_core) disable iff (rst_core)
        active_valid && !active_ready
        |=> $stable({active_pair_id, active_score_q7,
                     active_temporal_mask, active_k_mask, active_last})
    );
    assert property (@(posedge clk_core) disable iff (rst_core)
        class_valid |-> class_multiplicity != 0
    );
    assert property (@(posedge clk_core) disable iff (rst_core)
        active_valid |-> active_k_mask != 0
    );
    assert property (@(posedge clk_core) disable iff (rst_core)
        active_valid |-> (active_k_mask & ~active_temporal_mask) == 0
    );
    assert property (@(posedge clk_core) disable iff (rst_core)
        !protocol_error
    );
endmodule

bind h67_temporal_quotient_scs_frontend
    h67_temporal_quotient_scs_assertions #(
        .SCORE_W(SCORE_W),
        .PAIR_ID_W(PAIR_ID_W),
        .COUNT_W(COUNT_W),
        .CLASS_W(CLASS_W)
    ) u_h67_temporal_quotient_scs_assertions (.*);

`default_nettype wire
