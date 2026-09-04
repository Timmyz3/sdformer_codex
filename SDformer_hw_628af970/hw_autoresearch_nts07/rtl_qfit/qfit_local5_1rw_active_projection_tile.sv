`timescale 1ns/1ps
`default_nettype none

// Real online Local5 front end plus fair direct/GASR five-bank backend.
module qfit_local5_1rw_active_projection_tile #(
    parameter int MODE = 1,
    // 0: prepare at active-index issue; 1: prepare at descriptor commit.
    parameter int GEOMETRY_SYNC_MODE = 1,
    parameter int HEIGHT = 15,
    parameter int WIDTH = 15,
    parameter int TIME_PLANES = 2,
    parameter int HEAD_DIM = 32,
    parameter int OUT_DIM = 2,
    parameter int GATE_W = 9,
    parameter int W_W = 8,
    parameter int ACC_W = 32,
    parameter int RELATION_READ_LATENCY = 1,
    parameter int RELATION_MEMORY_IMPL = 0,
    parameter int ACC_MEMORY_IMPL = 0,
    parameter int Y_W = $clog2(HEIGHT),
    parameter int X_W = $clog2(WIDTH),
    parameter int PLANE_W = (TIME_PLANES <= 1) ? 1 : $clog2(TIME_PLANES),
    parameter int LANE_W = $clog2(HEAD_DIM),
    parameter int OUT_W = (OUT_DIM <= 1) ? 1 : $clog2(OUT_DIM),
    parameter int SOURCE_ID_W = $clog2(HEIGHT * WIDTH * TIME_PLANES)
) (
    input  logic                       clk_core,
    input  logic                       rst_core,
    input  logic                       weight_valid,
    output logic                       weight_ready,
    input  logic [LANE_W-1:0]          weight_lane,
    input  logic [OUT_W-1:0]           weight_out,
    input  logic signed [W_W-1:0]      weight_data,
    input  logic                       weight_last,
    input  logic                       projection_start,
    input  logic                       projection_close,
    output logic                       projection_close_ready,
    output logic                       projection_busy,
    output logic                       projection_done,
    input  logic                       relation_start,
    input  logic                       relation_seal,
    output logic                       relation_active,
    output logic                       relation_done,
    input  logic                       relation_valid,
    output logic                       relation_ready,
    input  logic [PLANE_W-1:0]         relation_plane,
    input  logic [Y_W-1:0]             relation_destination_y,
    input  logic [X_W-1:0]             relation_destination_x,
    input  logic [4:0]                 relation_candidate_valid,
    input  logic [4:0]                 relation_active_candidate_mask,
    input  logic [HEAD_DIM-1:0]        relation_k_self,
    input  logic [5*GATE_W-1:0]        relation_direction_gates,
    input  logic                       read_valid,
    output logic                       read_ready,
    input  logic [PLANE_W-1:0]         read_plane,
    input  logic [Y_W-1:0]             read_y,
    input  logic [X_W-1:0]             read_x,
    input  logic [OUT_W-1:0]           read_out,
    output logic                       read_data_valid,
    output logic signed [ACC_W-1:0]    read_data,
    output logic                       protocol_error,
    output logic [31:0]                perf_relation_writes,
    output logic [31:0]                perf_active_source_reads,
    output logic [31:0]                perf_dense_reads_avoided,
    output logic [31:0]                perf_memory_wait_cycles,
    output logic [31:0]                perf_descriptors,
    output logic [31:0]                perf_product_terms,
    output logic [31:0]                perf_destination_updates,
    output logic [31:0]                perf_term_stall_cycles,
    output logic [31:0]                perf_sram_reads,
    output logic [31:0]                perf_sram_writes
);
    logic descriptor_valid;
    logic frontier_descriptor_ready;
    logic builder_descriptor_ready;
    logic [SOURCE_ID_W-1:0] descriptor_source_id;
    logic [PLANE_W-1:0] descriptor_plane;
    logic [Y_W-1:0] descriptor_y;
    logic [X_W-1:0] descriptor_x;
    logic [HEAD_DIM-1:0] descriptor_k;
    logic [5*GATE_W-1:0] descriptor_gates;
    logic [4:0] descriptor_mask;
    logic descriptor_last;
    logic geometry_valid;
    logic [SOURCE_ID_W-1:0] geometry_source_id;
    logic [PLANE_W-1:0] geometry_plane;
    logic [Y_W-1:0] geometry_y;
    logic [X_W-1:0] geometry_x;
    logic geometry_last;
    logic backend_geometry_ready;
    logic frontier_geometry_ready;
    logic builder_descriptor_valid;
    logic backend_geometry_valid;
    logic [SOURCE_ID_W-1:0] backend_geometry_source_id;
    logic [PLANE_W-1:0] backend_geometry_plane;
    logic [Y_W-1:0] backend_geometry_y;
    logic [X_W-1:0] backend_geometry_x;
    logic [4:0] backend_geometry_role_mask;
    logic backend_geometry_last;
    logic term_valid;
    logic term_ready;
    logic [SOURCE_ID_W-1:0] term_source_id;
    logic [PLANE_W-1:0] term_source_plane;
    logic [Y_W-1:0] term_source_y;
    logic [X_W-1:0] term_source_x;
    logic [LANE_W-1:0] term_lane;
    logic [GATE_W-1:0] term_gate;
    logic [4:0] term_destination_mask;
    logic term_last;
    logic term_source_last;
    logic builder_idle;
    logic frontier_protocol_error;
    logic builder_protocol_error;
    logic backend_protocol_error;
    logic close_protocol_error_q;
    logic backend_close_ready;
    logic [31:0] builder_descriptors;
    logic [31:0] builder_terms;
    logic [31:0] builder_updates;
    logic [31:0] descriptor_base_q;

    qfit_dual_color_relation_frontier_sync #(
        .HEIGHT(HEIGHT), .WIDTH(WIDTH), .TIME_PLANES(TIME_PLANES),
        .K_W(HEAD_DIM), .GATE_W(GATE_W),
        .READ_LATENCY(RELATION_READ_LATENCY),
        .RELATION_MEMORY_IMPL(RELATION_MEMORY_IMPL)
    ) u_relation_frontier (
        .clk_core(clk_core), .rst_core(rst_core),
        .build_start(relation_start), .build_seal(relation_seal),
        .build_active(relation_active), .build_done(relation_done),
        .in_valid(relation_valid), .in_ready(relation_ready),
        .in_plane(relation_plane),
        .in_destination_y(relation_destination_y),
        .in_destination_x(relation_destination_x),
        .in_candidate_valid(relation_candidate_valid),
        .in_active_candidate_mask(relation_active_candidate_mask),
        .in_k_self(relation_k_self),
        .in_direction_gates(relation_direction_gates),
        .descriptor_valid(descriptor_valid),
        .descriptor_ready(frontier_descriptor_ready),
        .descriptor_source_id(descriptor_source_id),
        .descriptor_plane(descriptor_plane),
        .descriptor_y(descriptor_y), .descriptor_x(descriptor_x),
        .descriptor_k(descriptor_k),
        .descriptor_incoming_gates(descriptor_gates),
        .descriptor_valid_mask(descriptor_mask),
        .descriptor_last(descriptor_last),
        .geometry_valid(geometry_valid),
        .geometry_ready(frontier_geometry_ready),
        .geometry_source_id(geometry_source_id),
        .geometry_plane(geometry_plane), .geometry_y(geometry_y),
        .geometry_x(geometry_x), .geometry_last(geometry_last),
        .protocol_error(frontier_protocol_error),
        .perf_relation_writes(perf_relation_writes),
        .perf_source_reads(perf_active_source_reads),
        .perf_dense_reads_avoided(perf_dense_reads_avoided),
        .perf_memory_wait_cycles(perf_memory_wait_cycles)
    );

    qfit_source_multicast_term_builder_fifo2 #(
        .HEAD_DIM(HEAD_DIM), .GATE_W(GATE_W),
        .SOURCE_ID_W(SOURCE_ID_W), .PLANE_W(PLANE_W),
        .Y_W(Y_W), .X_W(X_W)
    ) u_builder_fifo (
        .clk_core(clk_core), .rst_core(rst_core),
        .descriptor_valid(builder_descriptor_valid),
        .descriptor_ready(builder_descriptor_ready),
        .descriptor_source_id(descriptor_source_id),
        .descriptor_plane(descriptor_plane),
        .descriptor_y(descriptor_y), .descriptor_x(descriptor_x),
        .descriptor_k(descriptor_k),
        .descriptor_incoming_gates(descriptor_gates),
        .descriptor_valid_mask(descriptor_mask),
        .descriptor_last(descriptor_last),
        .term_valid(term_valid), .term_ready(term_ready),
        .term_source_id(term_source_id),
        .term_source_plane(term_source_plane),
        .term_source_y(term_source_y), .term_source_x(term_source_x),
        .term_lane(term_lane), .term_gate(term_gate),
        .term_destination_mask(term_destination_mask),
        .term_last(term_last), .term_source_last(term_source_last),
        .pipeline_idle(builder_idle), .protocol_error(builder_protocol_error),
        .perf_descriptors(builder_descriptors), .perf_terms(builder_terms),
        .perf_destination_updates(builder_updates)
    );

    qfit_local5_1rw_projection_backend #(
        .MODE(MODE), .HEIGHT(HEIGHT), .WIDTH(WIDTH),
        .TIME_PLANES(TIME_PLANES), .HEAD_DIM(HEAD_DIM),
        .OUT_DIM(OUT_DIM), .GATE_W(GATE_W), .W_W(W_W), .ACC_W(ACC_W),
        .ACC_MEMORY_IMPL(ACC_MEMORY_IMPL),
        .SOURCE_ID_W(SOURCE_ID_W), .Y_W(Y_W), .X_W(X_W),
        .PLANE_W(PLANE_W), .LANE_W(LANE_W), .OUT_W(OUT_W)
    ) u_backend (
        .clk_core(clk_core), .rst_core(rst_core),
        .weight_valid(weight_valid), .weight_ready(weight_ready),
        .weight_lane(weight_lane), .weight_out(weight_out),
        .weight_data(weight_data), .weight_last(weight_last),
        .run_start(projection_start), .run_busy(projection_busy),
        .run_done(projection_done),
        .geometry_valid(backend_geometry_valid),
        .geometry_ready(backend_geometry_ready),
        .geometry_source_id(backend_geometry_source_id),
        .geometry_plane(backend_geometry_plane),
        .geometry_y(backend_geometry_y), .geometry_x(backend_geometry_x),
        .geometry_role_mask(backend_geometry_role_mask),
        .geometry_last(backend_geometry_last),
        .term_valid(term_valid), .term_ready(term_ready),
        .term_source_id(term_source_id),
        .term_source_plane(term_source_plane),
        .term_source_y(term_source_y), .term_source_x(term_source_x),
        .term_lane(term_lane), .term_gate(term_gate),
        .term_destination_mask(term_destination_mask),
        .term_last(term_last), .term_source_last(term_source_last),
        .window_close(projection_close && projection_close_ready),
        .window_close_ready(backend_close_ready),
        .read_valid(read_valid), .read_ready(read_ready),
        .read_plane(read_plane), .read_y(read_y), .read_x(read_x),
        .read_out(read_out), .read_data_valid(read_data_valid),
        .read_data(read_data), .protocol_error(backend_protocol_error),
        .perf_product_terms(perf_product_terms),
        .perf_destination_updates(perf_destination_updates),
        .perf_term_stall_cycles(perf_term_stall_cycles),
        .perf_sram_reads(perf_sram_reads),
        .perf_sram_writes(perf_sram_writes)
    );

    assign projection_close_ready = relation_done && builder_idle
                                  && backend_close_ready;
    always_comb begin
        backend_geometry_role_mask = 5'b11111;
        if (geometry_y == 0)
            backend_geometry_role_mask[2] = 1'b0;
        if (32'(geometry_y) == HEIGHT - 1)
            backend_geometry_role_mask[1] = 1'b0;
        if (geometry_x == 0)
            backend_geometry_role_mask[4] = 1'b0;
        if (32'(geometry_x) == WIDTH - 1)
            backend_geometry_role_mask[3] = 1'b0;
        if (GEOMETRY_SYNC_MODE != 0)
            backend_geometry_role_mask = descriptor_mask;
    end

    assign frontier_descriptor_ready = GEOMETRY_SYNC_MODE == 0
        ? builder_descriptor_ready
        : builder_descriptor_ready && backend_geometry_ready;
    assign frontier_geometry_ready = GEOMETRY_SYNC_MODE == 0
        ? backend_geometry_ready : 1'b1;
    assign builder_descriptor_valid = GEOMETRY_SYNC_MODE == 0
        ? descriptor_valid : descriptor_valid && backend_geometry_ready;
    assign backend_geometry_valid = GEOMETRY_SYNC_MODE == 0
        ? geometry_valid : descriptor_valid && builder_descriptor_ready;
    assign backend_geometry_source_id = GEOMETRY_SYNC_MODE == 0
        ? geometry_source_id : descriptor_source_id;
    assign backend_geometry_plane = GEOMETRY_SYNC_MODE == 0
        ? geometry_plane : descriptor_plane;
    assign backend_geometry_y = GEOMETRY_SYNC_MODE == 0
        ? geometry_y : descriptor_y;
    assign backend_geometry_x = GEOMETRY_SYNC_MODE == 0
        ? geometry_x : descriptor_x;
    assign backend_geometry_last = GEOMETRY_SYNC_MODE == 0
        ? geometry_last : descriptor_last;
    assign perf_descriptors = builder_descriptors - descriptor_base_q;
    assign protocol_error = frontier_protocol_error
                         || builder_protocol_error
                         || backend_protocol_error
                         || close_protocol_error_q;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            close_protocol_error_q <= 1'b0;
            descriptor_base_q <= '0;
        end else begin
            if (projection_start) begin
                close_protocol_error_q <= 1'b0;
                descriptor_base_q <= builder_descriptors;
            end
            if (projection_close && !projection_close_ready)
                close_protocol_error_q <= 1'b1;
        end
    end

    logic [31:0] unused_builder_terms;
    logic [31:0] unused_builder_updates;
    assign unused_builder_terms = builder_terms;
    assign unused_builder_updates = builder_updates;
endmodule

`default_nettype wire
