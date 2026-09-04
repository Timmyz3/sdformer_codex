`timescale 1ns/1ps
`default_nettype none

// Minimal C1+C2 integration: exact XBF-DBDR score fabric feeding FCSR-RX.
module qfit_local5_tile #(
    parameter int HEIGHT = 15,
    parameter int WIDTH = 15,
    parameter int TIME_PLANES = 2,
    parameter int TAG_W = 16,
    parameter int SCORE_W = 16,
    parameter int GATE_W = 9,
    parameter int Y_W = (HEIGHT <= 1) ? 1 : $clog2(HEIGHT),
    parameter int X_W = (WIDTH <= 1) ? 1 : $clog2(WIDTH),
    parameter int PLANE_W =
        (TIME_PLANES <= 1) ? 1 : $clog2(TIME_PLANES),
    parameter int SOURCE_ID_W =
        (HEIGHT * WIDTH * TIME_PLANES <= 1)
        ? 1 : $clog2(HEIGHT * WIDTH * TIME_PLANES)
) (
    input  logic                       clk_core,
    input  logic                       rst_core,
    input  logic                       plane_start,
    input  logic [PLANE_W-1:0]         plane_id,
    output logic                       plane_start_ready,
    input  logic                       in_valid,
    output logic                       in_ready,
    input  logic [Y_W-1:0]             in_y,
    input  logic [X_W-1:0]             in_x,
    input  logic [31:0]                in_q,
    input  logic [5*32-1:0]            in_k,
    input  logic [4:0]                 in_valid_mask,
    output logic                       descriptor_valid,
    input  logic                       descriptor_ready,
    output logic [SOURCE_ID_W-1:0]     descriptor_source_id,
    output logic [Y_W-1:0]             descriptor_y,
    output logic [X_W-1:0]             descriptor_x,
    output logic [31:0]                descriptor_k,
    output logic [5*GATE_W-1:0]        descriptor_incoming_gates,
    output logic [4:0]                 descriptor_valid_mask,
    output logic [15:0]                perf_score_service_cycles,
    output logic [3:0]                 perf_score_direct_mask,
    output logic [31:0]                perf_relation_stalls,
    output logic [2:0]                 perf_relation_max_pending
);
    localparam int META_W = PLANE_W + Y_W + X_W;

    logic score_in_ready;
    logic score_out_valid;
    logic score_out_ready;
    logic [TAG_W-1:0] score_out_tag;
    logic [5*SCORE_W-1:0] score_out_q7;
    logic [5*GATE_W-1:0] score_out_gate;
    logic [31:0] score_out_k_self;
    logic [4:0] score_out_valid_mask;
    logic [TAG_W-1:0] score_in_tag;
    logic [PLANE_W-1:0] score_plane;
    logic [Y_W-1:0] score_y;
    logic [X_W-1:0] score_x;
    logic relation_plane_idle;
    logic plane_start_fire;

    initial begin
        if (TAG_W < META_W)
            $error("TAG_W must cover plane/y/x metadata");
    end

    always_comb begin
        score_in_tag = '0;
        score_in_tag[META_W-1:0] = {plane_id, in_y, in_x};
        {score_plane, score_y, score_x} =
            score_out_tag[META_W-1:0];
    end

    assign in_ready = score_in_ready && !plane_start;
    assign plane_start_ready = score_in_ready
                            && !score_out_valid
                            && relation_plane_idle
                            && 32'(plane_id) < TIME_PLANES;
    assign plane_start_fire = plane_start && plane_start_ready;

    qfit_local5_score_leaf #(
        .ARCH_QFSA(1'b1),
        .PIPE_COMPACTOR(1'b1),
        .XBF_BANKED(1'b1),
        .USE_THRESHOLD_ROUTE(1'b1),
        .ROUTE_THRESHOLD(8),
        .USE_BANK_PRESSURE_ROUTE(1'b1),
        .BANK_PRESSURE_THRESHOLD(2),
        .TAG_W(TAG_W),
        .SCORE_W(SCORE_W),
        .GATE_W(GATE_W)
    ) u_score (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .in_valid(in_valid && !plane_start),
        .in_ready(score_in_ready),
        .in_tag(score_in_tag),
        .in_q(in_q),
        .in_k(in_k),
        .in_valid_mask(in_valid_mask),
        .out_valid(score_out_valid),
        .out_ready(score_out_ready),
        .out_tag(score_out_tag),
        .out_score_q7(score_out_q7),
        .out_gate_q17(score_out_gate),
        .out_k_self(score_out_k_self),
        .out_valid_mask(score_out_valid_mask),
        .perf_service_cycles(perf_score_service_cycles),
        .perf_route_direct_mask(perf_score_direct_mask)
    );

    qfit_relation_transpose_leaf #(
        .SCHED_MODE(0),
        .HEIGHT(HEIGHT),
        .WIDTH(WIDTH),
        .TIME_PLANES(TIME_PLANES),
        .K_W(32),
        .GATE_W(GATE_W)
    ) u_relation (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .plane_start(plane_start_fire),
        .plane_id(plane_id),
        .in_valid(score_out_valid),
        .in_ready(score_out_ready),
        .in_y(score_y),
        .in_x(score_x),
        .in_candidate_valid(score_out_valid_mask),
        .in_k_self(score_out_k_self),
        .in_direction_gates(score_out_gate),
        .descriptor_valid(descriptor_valid),
        .descriptor_ready(descriptor_ready),
        .descriptor_source_id(descriptor_source_id),
        .descriptor_y(descriptor_y),
        .descriptor_x(descriptor_x),
        .descriptor_k(descriptor_k),
        .descriptor_incoming_gates(descriptor_incoming_gates),
        .descriptor_valid_mask(descriptor_valid_mask),
        .plane_idle(relation_plane_idle),
        .perf_producer_stalls(perf_relation_stalls),
        .perf_max_pending(perf_relation_max_pending),
        .debug_read_pending(),
        .debug_k_read_data_valid()
    );

    logic unused_score_plane;
    logic [5*SCORE_W-1:0] unused_score_q7;
    assign unused_score_plane = ^score_plane;
    assign unused_score_q7 = score_out_q7;
endmodule

`default_nettype wire
