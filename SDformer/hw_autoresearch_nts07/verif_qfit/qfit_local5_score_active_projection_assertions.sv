`timescale 1ns/1ps
`default_nettype none

module qfit_local5_score_active_projection_assertions #(
    parameter int HEAD_DIM = 32,
    parameter int PLANE_W = 1,
    parameter int Y_W = 4,
    parameter int X_W = 4
) (
    input logic clk_core,
    input logic rst_core,
    input logic row_valid,
    input logic row_ready,
    input logic [PLANE_W-1:0] row_plane,
    input logic [Y_W-1:0] row_destination_y,
    input logic [X_W-1:0] row_destination_x,
    input logic [HEAD_DIM-1:0] row_q,
    input logic [5*HEAD_DIM-1:0] row_candidate_k,
    input logic [4:0] row_candidate_valid,
    input logic relation_seal,
    input logic relation_seal_ready,
    input logic [1:0] meta_count_q,
    input logic score_out_valid,
    input logic score_out_ready,
    input logic backend_relation_ready,
    input logic protocol_error
);
    property p_row_holds_under_backpressure;
        @(posedge clk_core) disable iff (rst_core)
        row_valid && !row_ready |=> row_valid
            && $stable({row_plane, row_destination_y, row_destination_x,
                        row_q, row_candidate_k, row_candidate_valid});
    endproperty
    assert property (p_row_holds_under_backpressure);

    property p_seal_only_when_drained;
        @(posedge clk_core) disable iff (rst_core)
        relation_seal |-> relation_seal_ready
            && (meta_count_q == 2'd0) && !score_out_valid;
    endproperty
    assert property (p_seal_only_when_drained);

    property p_score_backpressure_matches_projection;
        @(posedge clk_core) disable iff (rst_core)
        score_out_valid && (meta_count_q != 2'd0)
        |-> (score_out_ready == backend_relation_ready);
    endproperty
    assert property (p_score_backpressure_matches_projection);

    property p_score_has_metadata;
        @(posedge clk_core) disable iff (rst_core)
        score_out_valid |-> (meta_count_q != 2'd0);
    endproperty
    assert property (p_score_has_metadata);

    property p_metadata_count_bounded;
        @(posedge clk_core) disable iff (rst_core)
        meta_count_q <= 2'd2;
    endproperty
    assert property (p_metadata_count_bounded);

    property p_no_error;
        @(posedge clk_core) disable iff (rst_core) !protocol_error;
    endproperty
    assert property (p_no_error);
endmodule

bind qfit_local5_score_active_projection_tile
    qfit_local5_score_active_projection_assertions #(
        .HEAD_DIM(HEAD_DIM),
        .PLANE_W(PLANE_W),
        .Y_W(Y_W),
        .X_W(X_W)
    ) u_qfit_local5_score_active_projection_assertions (.*);

`default_nettype wire
