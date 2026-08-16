`timescale 1ns/1ps
`default_nettype none

`ifndef QFIT_ROLLING_SCHED_MODE
`define QFIT_ROLLING_SCHED_MODE 0
`endif

// Sidecar implementation of the production frontier interface using the
// three-row closed-form relation-transpose leaf. Compile this file instead of
// rtl_qfit/qfit_dual_color_relation_frontier_sync.sv for rolling experiments.
module qfit_dual_color_relation_frontier_sync #(
    parameter int HEIGHT = 15,
    parameter int WIDTH = 15,
    parameter int TIME_PLANES = 2,
    parameter int K_W = 32,
    parameter int GATE_W = 9,
    parameter int READ_LATENCY = 1,
    parameter int RELATION_MEMORY_IMPL = 0,
    parameter int ROLLING_SCHED_MODE = `QFIT_ROLLING_SCHED_MODE,
    parameter int ROLLING_STRIPE_RING_ROWS = 4,
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
    input  logic                       build_start,
    input  logic                       build_seal,
    output logic                       build_active,
    output logic                       build_done,
    input  logic                       in_valid,
    output logic                       in_ready,
    input  logic [PLANE_W-1:0]         in_plane,
    input  logic [Y_W-1:0]             in_destination_y,
    input  logic [X_W-1:0]             in_destination_x,
    input  logic [4:0]                 in_candidate_valid,
    input  logic [4:0]                 in_active_candidate_mask,
    input  logic [K_W-1:0]             in_k_self,
    input  logic [5*GATE_W-1:0]        in_direction_gates,
    output logic                       descriptor_valid,
    input  logic                       descriptor_ready,
    output logic [SOURCE_ID_W-1:0]     descriptor_source_id,
    output logic [PLANE_W-1:0]         descriptor_plane,
    output logic [Y_W-1:0]             descriptor_y,
    output logic [X_W-1:0]             descriptor_x,
    output logic [K_W-1:0]             descriptor_k,
    output logic [5*GATE_W-1:0]        descriptor_incoming_gates,
    output logic [4:0]                 descriptor_valid_mask,
    output logic                       descriptor_last,
    output logic                       geometry_valid,
    input  logic                       geometry_ready,
    output logic [SOURCE_ID_W-1:0]     geometry_source_id,
    output logic [PLANE_W-1:0]         geometry_plane,
    output logic [Y_W-1:0]             geometry_y,
    output logic [X_W-1:0]             geometry_x,
    output logic                       geometry_last,
    output logic                       protocol_error,
    output logic [31:0]                perf_relation_writes,
    output logic [31:0]                perf_source_reads,
    output logic [31:0]                perf_dense_reads_avoided,
    output logic [31:0]                perf_memory_wait_cycles
);
    localparam int TOKENS_PER_PLANE = HEIGHT * WIDTH;
    localparam int TOTAL_SOURCES = TOKENS_PER_PLANE * TIME_PLANES;
    localparam int TOKEN_COUNT_W =
        (TOKENS_PER_PLANE <= 1) ? 1 : $clog2(TOKENS_PER_PLANE + 1);

    typedef enum logic [2:0] {
        ST_IDLE,
        ST_START_PLANE,
        ST_FEED_PLANE,
        ST_WAIT_PLANE,
        ST_DONE
    } state_t;

    state_t state_q;
    logic [PLANE_W-1:0] plane_q;
    logic [TOKEN_COUNT_W-1:0] plane_tokens_q;
    logic seal_seen_q;
    logic protocol_error_q;
    logic [31:0] relation_writes_q;
    logic [31:0] source_reads_q;
    logic [31:0] memory_wait_cycles_q;

    logic leaf_plane_start;
    logic leaf_in_valid;
    logic leaf_in_ready;
    logic leaf_descriptor_valid;
    logic leaf_descriptor_ready;
    logic [SOURCE_ID_W-1:0] leaf_descriptor_source_id;
    logic [Y_W-1:0] leaf_descriptor_y;
    logic [X_W-1:0] leaf_descriptor_x;
    logic [K_W-1:0] leaf_descriptor_k;
    logic [5*GATE_W-1:0] leaf_descriptor_gates;
    logic [4:0] leaf_descriptor_mask;
    logic leaf_plane_idle;
    logic [31:0] unused_producer_stalls;
    logic [2:0] unused_max_pending;
    logic [4:0] leaf_candidate_valid;
    logic input_fire;
    logic active_descriptor;

    assign leaf_plane_start = state_q == ST_START_PLANE && leaf_plane_idle;
    assign leaf_in_valid = state_q == ST_FEED_PLANE && in_valid;
    assign in_ready = state_q == ST_FEED_PLANE && leaf_in_ready;
    assign input_fire = in_valid && in_ready;
    // Dynamic consumer counting is a stronger exact baseline when inactive K
    // sources never allocate counter events. FCSR derives its three closed-form
    // release events from the resident K ring, while Stripe still needs the
    // geometric topology mask for row release.
    assign leaf_candidate_valid = (
        ROLLING_SCHED_MODE == 1 || ROLLING_SCHED_MODE == 3
    ) ? in_active_candidate_mask : in_candidate_valid;

    // K==0 sources have no projection term. The dense retirement event still
    // drains the three-row ring, but it is not materialized downstream.
    assign active_descriptor = leaf_descriptor_k != '0;
    assign leaf_descriptor_ready = leaf_descriptor_valid
        && !active_descriptor ? 1'b1 : descriptor_ready;
    assign descriptor_valid = leaf_descriptor_valid && active_descriptor;
    assign descriptor_source_id = leaf_descriptor_source_id;
    always_comb begin
        descriptor_plane = '0;
        for (integer plane = 1; plane < TIME_PLANES; plane = plane + 1)
            if (32'(leaf_descriptor_source_id) >= plane * TOKENS_PER_PLANE)
                descriptor_plane = PLANE_W'(plane);
    end
    assign descriptor_y = leaf_descriptor_y;
    assign descriptor_x = leaf_descriptor_x;
    assign descriptor_k = leaf_descriptor_k;
    assign descriptor_incoming_gates = leaf_descriptor_gates;
    assign descriptor_valid_mask = leaf_descriptor_mask;
    assign descriptor_last = descriptor_valid
        && descriptor_source_id == SOURCE_ID_W'(TOTAL_SOURCES - 1);

    assign build_active = state_q != ST_IDLE && state_q != ST_DONE;
    assign build_done = state_q == ST_DONE;
    assign protocol_error = protocol_error_q;
    assign perf_relation_writes = relation_writes_q;
    assign perf_source_reads = source_reads_q;
    assign perf_dense_reads_avoided = build_done
        ? 32'(TOTAL_SOURCES) - source_reads_q : '0;
    assign perf_memory_wait_cycles = memory_wait_cycles_q;

    // Geometry look-ahead is not used by qfit_local5_active_projection_tile.
    assign geometry_valid = 1'b0;
    assign geometry_source_id = '0;
    assign geometry_plane = '0;
    assign geometry_y = '0;
    assign geometry_x = '0;
    assign geometry_last = 1'b0;

    qfit_relation_transpose_leaf #(
        .SCHED_MODE(ROLLING_SCHED_MODE),
        .HEIGHT(HEIGHT),
        .WIDTH(WIDTH),
        .TIME_PLANES(TIME_PLANES),
        .K_W(K_W),
        .GATE_W(GATE_W),
        .SKIP_ZERO_K(1'b1),
        .STRIPE_RING_ROWS(ROLLING_STRIPE_RING_ROWS)
    ) u_rolling_leaf (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .plane_start(leaf_plane_start),
        .plane_id(plane_q),
        .in_valid(leaf_in_valid),
        .in_ready(leaf_in_ready),
        .in_y(in_destination_y),
        .in_x(in_destination_x),
        .in_candidate_valid(leaf_candidate_valid),
        .in_k_self(in_k_self),
        .in_direction_gates(in_direction_gates),
        .descriptor_valid(leaf_descriptor_valid),
        .descriptor_ready(leaf_descriptor_ready),
        .descriptor_source_id(leaf_descriptor_source_id),
        .descriptor_y(leaf_descriptor_y),
        .descriptor_x(leaf_descriptor_x),
        .descriptor_k(leaf_descriptor_k),
        .descriptor_incoming_gates(leaf_descriptor_gates),
        .descriptor_valid_mask(leaf_descriptor_mask),
        .plane_idle(leaf_plane_idle),
        .perf_producer_stalls(unused_producer_stalls),
        .perf_max_pending(unused_max_pending),
        .debug_read_pending(read_pending_q),
        .debug_k_read_data_valid(k_read_data_valid)
    );

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            state_q <= ST_IDLE;
            plane_q <= '0;
            plane_tokens_q <= '0;
            seal_seen_q <= 1'b0;
            protocol_error_q <= 1'b0;
            relation_writes_q <= '0;
            source_reads_q <= '0;
            memory_wait_cycles_q <= '0;
        end else begin
            if (build_start) begin
                if (state_q != ST_IDLE && state_q != ST_DONE)
                    protocol_error_q <= 1'b1;
                else begin
                    state_q <= ST_START_PLANE;
                    plane_q <= '0;
                    plane_tokens_q <= '0;
                    seal_seen_q <= 1'b0;
                    protocol_error_q <= 1'b0;
                    relation_writes_q <= '0;
                    source_reads_q <= '0;
                    memory_wait_cycles_q <= '0;
                end
            end else begin
                if (build_seal) begin
                    seal_seen_q <= 1'b1;
                    if (relation_writes_q != 32'(TOTAL_SOURCES))
                        protocol_error_q <= 1'b1;
                end
                if (in_valid && !in_ready)
                    memory_wait_cycles_q <= memory_wait_cycles_q + 1'b1;
                if (input_fire) begin
                    relation_writes_q <= relation_writes_q + 1'b1;
                    if (
                        in_plane != plane_q
                        || 32'(in_destination_y) != 32'(plane_tokens_q) / WIDTH
                        || 32'(in_destination_x) != 32'(plane_tokens_q) % WIDTH
                    )
                        protocol_error_q <= 1'b1;
                    if (plane_tokens_q == TOKEN_COUNT_W'(TOKENS_PER_PLANE - 1)) begin
                        plane_tokens_q <= '0;
                        state_q <= ST_WAIT_PLANE;
                    end else begin
                        plane_tokens_q <= plane_tokens_q + 1'b1;
                    end
                end
                if (
                    leaf_descriptor_valid
                    && leaf_descriptor_ready
                    && active_descriptor
                )
                    source_reads_q <= source_reads_q + 1'b1;

                case (state_q)
                    ST_START_PLANE: begin
                        if (leaf_plane_idle)
                            state_q <= ST_FEED_PLANE;
                    end
                    ST_WAIT_PLANE: begin
                        if (leaf_plane_idle) begin
                            if (plane_q == PLANE_W'(TIME_PLANES - 1)) begin
                                if (seal_seen_q || build_seal)
                                    state_q <= ST_DONE;
                            end else begin
                                plane_q <= plane_q + 1'b1;
                                state_q <= ST_START_PLANE;
                            end
                        end
                    end
                    default: begin end
                endcase
            end
        end
    end

    logic unused_geometry_ready;
    logic read_pending_q;
    logic k_read_data_valid;
    assign unused_geometry_ready = geometry_ready;
    initial begin
        if (READ_LATENCY != 1)
            $fatal(1, "rolling frontier requires READ_LATENCY=1");
        if (RELATION_MEMORY_IMPL != 0)
            $fatal(1, "rolling frontier fakeram binding is not implemented");
    end
endmodule

`default_nettype wire
