`timescale 1ns/1ps
`default_nettype none

module qfit_local5_1rw_active_projection_assertions #(
    parameter int MODE = 1,
    parameter int GEOMETRY_SYNC_MODE = 1,
    parameter int SOURCE_ID_W = 9,
    parameter int PLANE_W = 1,
    parameter int Y_W = 4,
    parameter int X_W = 4,
    parameter int HEAD_DIM = 32,
    parameter int GATE_W = 9
) (
    input logic clk_core,
    input logic rst_core,
    input logic projection_start,
    input logic projection_close,
    input logic projection_close_ready,
    input logic descriptor_valid,
    input logic frontier_descriptor_ready,
    input logic builder_descriptor_ready,
    input logic backend_geometry_ready,
    input logic [SOURCE_ID_W-1:0] descriptor_source_id,
    input logic [PLANE_W-1:0] descriptor_plane,
    input logic [Y_W-1:0] descriptor_y,
    input logic [X_W-1:0] descriptor_x,
    input logic [HEAD_DIM-1:0] descriptor_k,
    input logic [5*GATE_W-1:0] descriptor_gates,
    input logic [4:0] descriptor_mask,
    input logic descriptor_last,
    input logic term_valid,
    input logic term_ready,
    input logic [SOURCE_ID_W-1:0] term_source_id,
    input logic builder_idle,
    input logic relation_done,
    input logic backend_close_ready,
    input logic backend_geometry_fire,
    input logic issue_geometry_valid,
    input logic issue_geometry_ready,
    input logic [SOURCE_ID_W-1:0] issue_geometry_source_id,
    input logic [PLANE_W-1:0] issue_geometry_plane,
    input logic [Y_W-1:0] issue_geometry_y,
    input logic [X_W-1:0] issue_geometry_x,
    input logic issue_geometry_last,
    input logic backend_current_source_valid,
    input logic [SOURCE_ID_W-1:0] backend_current_source_id,
    input logic protocol_error
);
    property p_descriptor_stable_until_atomic_accept;
        @(posedge clk_core) disable iff (rst_core || projection_start)
        descriptor_valid && !frontier_descriptor_ready
        |=> descriptor_valid && $stable({
            descriptor_source_id, descriptor_plane, descriptor_y,
            descriptor_x, descriptor_k, descriptor_gates,
            descriptor_mask, descriptor_last
        });
    endproperty

    property p_descriptor_commit_is_atomic;
        @(posedge clk_core) disable iff (rst_core || projection_start)
        GEOMETRY_SYNC_MODE != 0
        && descriptor_valid && frontier_descriptor_ready
        |-> builder_descriptor_ready && backend_geometry_ready
            && backend_geometry_fire;
    endproperty

    property p_no_partial_descriptor_commit;
        @(posedge clk_core) disable iff (rst_core || projection_start)
        GEOMETRY_SYNC_MODE != 0 && backend_geometry_fire
        |-> descriptor_valid && builder_descriptor_ready
            && frontier_descriptor_ready;
    endproperty

    property p_issue_geometry_stable_until_prepare;
        @(posedge clk_core) disable iff (rst_core || projection_start)
        GEOMETRY_SYNC_MODE == 0
        && issue_geometry_valid && !issue_geometry_ready
        |=> issue_geometry_valid && $stable({
            issue_geometry_source_id, issue_geometry_plane,
            issue_geometry_y, issue_geometry_x, issue_geometry_last
        });
    endproperty

    property p_gasr_term_matches_active_source;
        @(posedge clk_core) disable iff (rst_core || projection_start)
        MODE != 0 && term_valid && term_ready
        |-> backend_current_source_valid
            && term_source_id == backend_current_source_id;
    endproperty

    property p_close_only_after_pipeline_drain;
        @(posedge clk_core) disable iff (rst_core || projection_start)
        projection_close && projection_close_ready
        |-> relation_done && builder_idle && backend_close_ready;
    endproperty

    assert property (p_descriptor_stable_until_atomic_accept);
    assert property (p_descriptor_commit_is_atomic);
    assert property (p_no_partial_descriptor_commit);
    assert property (p_issue_geometry_stable_until_prepare);
    assert property (p_gasr_term_matches_active_source);
    assert property (p_close_only_after_pipeline_drain);
    assert property (@(posedge clk_core) disable iff (rst_core)
        !protocol_error
    );
endmodule

bind qfit_local5_1rw_active_projection_tile
    qfit_local5_1rw_active_projection_assertions #(
        .MODE(MODE), .GEOMETRY_SYNC_MODE(GEOMETRY_SYNC_MODE),
        .SOURCE_ID_W(SOURCE_ID_W), .PLANE_W(PLANE_W),
        .Y_W(Y_W), .X_W(X_W), .HEAD_DIM(HEAD_DIM), .GATE_W(GATE_W)
    ) u_qfit_local5_1rw_active_projection_assertions (
        .clk_core(clk_core), .rst_core(rst_core),
        .projection_start(projection_start),
        .projection_close(projection_close),
        .projection_close_ready(projection_close_ready),
        .descriptor_valid(descriptor_valid),
        .frontier_descriptor_ready(frontier_descriptor_ready),
        .builder_descriptor_ready(builder_descriptor_ready),
        .backend_geometry_ready(backend_geometry_ready),
        .descriptor_source_id(descriptor_source_id),
        .descriptor_plane(descriptor_plane), .descriptor_y(descriptor_y),
        .descriptor_x(descriptor_x), .descriptor_k(descriptor_k),
        .descriptor_gates(descriptor_gates), .descriptor_mask(descriptor_mask),
        .descriptor_last(descriptor_last), .term_valid(term_valid),
        .term_ready(term_ready), .term_source_id(term_source_id),
        .builder_idle(builder_idle), .relation_done(relation_done),
        .backend_close_ready(backend_close_ready),
        .backend_geometry_fire(u_backend.geometry_fire),
        .issue_geometry_valid(geometry_valid),
        .issue_geometry_ready(frontier_geometry_ready),
        .issue_geometry_source_id(geometry_source_id),
        .issue_geometry_plane(geometry_plane),
        .issue_geometry_y(geometry_y), .issue_geometry_x(geometry_x),
        .issue_geometry_last(geometry_last),
        .backend_current_source_valid(u_backend.current_source_valid_q),
        .backend_current_source_id(u_backend.current_source_id_q),
        .protocol_error(protocol_error)
    );

`default_nettype wire
