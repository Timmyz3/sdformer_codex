`timescale 1ns/1ps
`default_nettype none

// Local5 post-score architecture slice. A destination-major relation stream
// is converted into active-source descriptors, gate-equivalent product terms,
// and finally accumulated by either the topology-colored or linear baseline.
module qfit_local5_active_projection_tile #(
    parameter int HEIGHT = 15,
    parameter int WIDTH = 15,
    parameter int TIME_PLANES = 2,
    parameter int HEAD_DIM = 32,
    parameter int OUT_DIM = 4,
    parameter int GATE_W = 9,
    parameter int W_W = 8,
    parameter int ACC_W = 32,
    parameter int RELATION_READ_LATENCY = 1,
    parameter int RELATION_MEMORY_IMPL = 0,
    parameter int RELATION_SCHED_MODE = 0,
    // 0: TCFM-5; 1: equal-capacity Linear-5 baseline.
    parameter int BACKEND_KIND = 0,
    parameter int ACC_BACKEND_KIND = 0,
    parameter int ACC_MEMORY_IMPL = 0,
    parameter bit GROUP_EQUAL_GATES = 1'b1,
    // 0 keeps the sealed direct TCFM5 path; >=2 inserts the existing LRU leaf.
    parameter int PRODUCT_CACHE_WAYS = 0,
    parameter int Y_W = (HEIGHT <= 1) ? 1 : $clog2(HEIGHT),
    parameter int X_W = (WIDTH <= 1) ? 1 : $clog2(WIDTH),
    parameter int PLANE_W =
        (TIME_PLANES <= 1) ? 1 : $clog2(TIME_PLANES),
    parameter int LANE_W =
        (HEAD_DIM <= 1) ? 1 : $clog2(HEAD_DIM),
    parameter int OUT_W =
        (OUT_DIM <= 1) ? 1 : $clog2(OUT_DIM),
    parameter int SOURCE_ID_W =
        (HEIGHT * WIDTH * TIME_PLANES <= 1)
        ? 1 : $clog2(HEIGHT * WIDTH * TIME_PLANES)
) (
    input  logic                       clk_core,
    input  logic                       rst_core,

    input  logic                       weight_valid,
    output logic                       weight_ready,
    input  logic [LANE_W-1:0]          weight_lane,
    input  logic [OUT_W-1:0]           weight_out,
    input  logic signed [W_W-1:0]      weight_data,
    input  logic                       weight_last,
    input  logic                       weight_context_release,
    output logic                       weight_context_release_ready,

    input  logic                       projection_start,
    input  logic                       projection_accumulate,
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
    output logic [31:0]                perf_cache_hits,
    output logic [31:0]                perf_cache_misses,
    output logic [31:0]                perf_tag_compares,
    output logic [31:0]                perf_lru_writes,
    output logic [31:0]                perf_product_reads,
    output logic [31:0]                perf_product_writes,
    output logic [31:0]                perf_product_starts,
    output logic [31:0]                perf_weight_reads
);
    localparam int TOKENS_PER_PLANE = HEIGHT * WIDTH;

    logic descriptor_valid;
    logic descriptor_ready;
    logic [SOURCE_ID_W-1:0] descriptor_source_id;
    logic [PLANE_W-1:0] descriptor_plane;
    logic [Y_W-1:0] descriptor_y;
    logic [X_W-1:0] descriptor_x;
    logic [HEAD_DIM-1:0] descriptor_k;
    logic [5*GATE_W-1:0] descriptor_gates;
    logic [4:0] descriptor_mask;
    logic descriptor_last;
    logic unused_geometry_valid;
    logic [SOURCE_ID_W-1:0] unused_geometry_source_id;
    logic [PLANE_W-1:0] unused_geometry_plane;
    logic [Y_W-1:0] unused_geometry_y;
    logic [X_W-1:0] unused_geometry_x;
    logic unused_geometry_last;
    logic frontier_protocol_error;

    logic term_valid;
    logic term_ready;
    logic [SOURCE_ID_W-1:0] term_source_id;
    logic [Y_W-1:0] term_source_y;
    logic [X_W-1:0] term_source_x;
    logic [LANE_W-1:0] term_lane;
    logic [GATE_W-1:0] term_gate;
    logic [4:0] term_destination_mask;
    logic term_last;
    logic [PLANE_W-1:0] term_source_plane;
    logic [31:0] builder_terms;
    logic [31:0] builder_updates;
    logic [31:0] builder_descriptors;
    logic [31:0] descriptor_base_q;

    logic backend_close_ready;
    logic backend_protocol_error;
    logic [31:0] backend_terms;
    logic [31:0] backend_updates;
    logic close_protocol_error_q;

    initial begin
        if (HEAD_DIM != 32)
            $error("qfit_local5_active_projection_tile requires HEAD_DIM=32");
        if (BACKEND_KIND < 0 || BACKEND_KIND > 1)
            $error("unsupported BACKEND_KIND");
        if (PRODUCT_CACHE_WAYS == 1)
            $error("PRODUCT_CACHE_WAYS must be 0 or >=2");
        if (BACKEND_KIND != 0 && PRODUCT_CACHE_WAYS != 0)
            $error("product-cache ablation requires TCFM5 backend");
    end

    qfit_dual_color_relation_frontier_sync #(
        .HEIGHT(HEIGHT),
        .WIDTH(WIDTH),
        .TIME_PLANES(TIME_PLANES),
        .K_W(HEAD_DIM),
        .GATE_W(GATE_W),
        .READ_LATENCY(RELATION_READ_LATENCY),
        .RELATION_MEMORY_IMPL(RELATION_MEMORY_IMPL),
        .ROLLING_SCHED_MODE(RELATION_SCHED_MODE)
    ) u_relation_frontier (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .build_start(relation_start),
        .build_seal(relation_seal),
        .build_active(relation_active),
        .build_done(relation_done),
        .in_valid(relation_valid),
        .in_ready(relation_ready),
        .in_plane(relation_plane),
        .in_destination_y(relation_destination_y),
        .in_destination_x(relation_destination_x),
        .in_candidate_valid(relation_candidate_valid),
        .in_active_candidate_mask(relation_active_candidate_mask),
        .in_k_self(relation_k_self),
        .in_direction_gates(relation_direction_gates),
        .descriptor_valid(descriptor_valid),
        .descriptor_ready(descriptor_ready),
        .descriptor_source_id(descriptor_source_id),
        .descriptor_plane(descriptor_plane),
        .descriptor_y(descriptor_y),
        .descriptor_x(descriptor_x),
        .descriptor_k(descriptor_k),
        .descriptor_incoming_gates(descriptor_gates),
        .descriptor_valid_mask(descriptor_mask),
        .descriptor_last(descriptor_last),
        .geometry_valid(unused_geometry_valid),
        .geometry_ready(1'b1),
        .geometry_source_id(unused_geometry_source_id),
        .geometry_plane(unused_geometry_plane),
        .geometry_y(unused_geometry_y),
        .geometry_x(unused_geometry_x),
        .geometry_last(unused_geometry_last),
        .protocol_error(frontier_protocol_error),
        .perf_relation_writes(perf_relation_writes),
        .perf_source_reads(perf_active_source_reads),
        .perf_dense_reads_avoided(perf_dense_reads_avoided),
        .perf_memory_wait_cycles(perf_memory_wait_cycles)
    );

    // source_id contains the plane in the same raster order as the frontier.
    always_comb begin
        term_source_plane = '0;
        for (integer plane = 1; plane < TIME_PLANES; plane = plane + 1)
            if (32'(term_source_id) >= plane * TOKENS_PER_PLANE)
                term_source_plane = PLANE_W'(plane);
    end

    qfit_source_multicast_term_builder #(
        .HEAD_DIM(HEAD_DIM),
        .GATE_W(GATE_W),
        .GROUP_EQUAL_GATES(GROUP_EQUAL_GATES),
        .SOURCE_ID_W(SOURCE_ID_W),
        .Y_W(Y_W),
        .X_W(X_W)
    ) u_term_builder (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .descriptor_valid(descriptor_valid),
        .descriptor_ready(descriptor_ready),
        .descriptor_source_id(descriptor_source_id),
        .descriptor_y(descriptor_y),
        .descriptor_x(descriptor_x),
        .descriptor_k(descriptor_k),
        .descriptor_incoming_gates(descriptor_gates),
        .descriptor_valid_mask(descriptor_mask),
        .term_valid(term_valid),
        .term_ready(term_ready),
        .term_source_id(term_source_id),
        .term_source_y(term_source_y),
        .term_source_x(term_source_x),
        .term_lane(term_lane),
        .term_gate(term_gate),
        .term_destination_mask(term_destination_mask),
        .term_last(term_last),
        .perf_descriptors(builder_descriptors),
        .perf_terms(builder_terms),
        .perf_destination_updates(builder_updates)
    );

    generate
        if (BACKEND_KIND == 0) begin : g_tcfm5
            if (PRODUCT_CACHE_WAYS == 0) begin : g_direct
                qfit_tcfm5_projection_top #(
                    .HEIGHT(HEIGHT),
                    .WIDTH(WIDTH),
                    .TIME_PLANES(TIME_PLANES),
                    .HEAD_DIM(HEAD_DIM),
                    .OUT_DIM(OUT_DIM),
                    .GATE_W(GATE_W),
                    .W_W(W_W),
                    .ACC_W(ACC_W),
                    .ACC_BACKEND_KIND(ACC_BACKEND_KIND),
                    .ACC_MEMORY_IMPL(ACC_MEMORY_IMPL)
                ) u_backend (
                    .clk_core(clk_core), .rst_core(rst_core),
                    .weight_valid(weight_valid), .weight_ready(weight_ready),
                    .weight_lane(weight_lane), .weight_out(weight_out),
                    .weight_data(weight_data), .weight_last(weight_last),
                    .weight_context_release(weight_context_release),
                    .weight_context_release_ready(
                        weight_context_release_ready
                    ),
                    .run_start(projection_start),
                    .run_accumulate(projection_accumulate),
                    .run_busy(projection_busy),
                    .run_done(projection_done), .term_valid(term_valid),
                    .term_ready(term_ready),
                    .term_source_plane(term_source_plane),
                    .term_source_y(term_source_y),
                    .term_source_x(term_source_x),
                    .term_lane(term_lane), .term_gate(term_gate),
                    .term_destination_mask(term_destination_mask),
                    .term_product('0), .term_window_last(1'b0),
                    .window_close(projection_close && projection_close_ready),
                    .window_close_ready(backend_close_ready),
                    .read_valid(read_valid), .read_ready(read_ready),
                    .read_plane(read_plane), .read_y(read_y), .read_x(read_x),
                    .read_out(read_out), .read_data_valid(read_data_valid),
                    .read_data(read_data),
                    .protocol_error(backend_protocol_error),
                    .vector_read_valid(1'b0), .vector_read_ready(),
                    .vector_read_plane('0), .vector_read_y('0),
                    .vector_read_x('0), .vector_read_data_valid(),
                    .vector_read_data(),
                    .perf_product_terms(backend_terms),
                    .perf_destination_updates(backend_updates)
                );
                assign perf_cache_hits = '0;
                assign perf_cache_misses = '0;
                assign perf_tag_compares = '0;
                assign perf_lru_writes = '0;
                assign perf_product_reads = '0;
                assign perf_product_writes = '0;
                assign perf_product_starts = backend_terms;
                assign perf_weight_reads = backend_terms * OUT_DIM;
            end else begin : g_cached
                assign weight_context_release_ready = 1'b0;
                qfit_cached_tcfm5_projection_top #(
                    .HEIGHT(HEIGHT), .WIDTH(WIDTH),
                    .TIME_PLANES(TIME_PLANES), .HEAD_DIM(HEAD_DIM),
                    .OUT_DIM(OUT_DIM), .GATE_W(GATE_W), .W_W(W_W),
                    .ACC_W(ACC_W), .CACHE_WAYS(PRODUCT_CACHE_WAYS),
                    .ACC_BACKEND_KIND(ACC_BACKEND_KIND),
                    .ACC_MEMORY_IMPL(ACC_MEMORY_IMPL)
                ) u_backend (
                    .clk_core(clk_core), .rst_core(rst_core),
                    .weight_valid(weight_valid), .weight_ready(weight_ready),
                    .weight_lane(weight_lane), .weight_out(weight_out),
                    .weight_data(weight_data), .weight_last(weight_last),
                    .run_start(projection_start), .run_busy(projection_busy),
                    .run_done(projection_done), .term_valid(term_valid),
                    .term_ready(term_ready),
                    .term_source_plane(term_source_plane),
                    .term_source_y(term_source_y),
                    .term_source_x(term_source_x),
                    .term_lane(term_lane), .term_gate(term_gate),
                    .term_destination_mask(term_destination_mask),
                    .term_window_last(1'b0),
                    .window_close(projection_close && projection_close_ready),
                    .window_close_ready(backend_close_ready),
                    .read_valid(read_valid), .read_ready(read_ready),
                    .read_plane(read_plane), .read_y(read_y), .read_x(read_x),
                    .read_out(read_out), .read_data_valid(read_data_valid),
                    .read_data(read_data),
                    .protocol_error(backend_protocol_error),
                    .perf_product_terms(backend_terms),
                    .perf_destination_updates(backend_updates),
                    .perf_cache_hits(perf_cache_hits),
                    .perf_cache_misses(perf_cache_misses),
                    .perf_tag_compares(perf_tag_compares),
                    .perf_lru_writes(perf_lru_writes),
                    .perf_product_reads(perf_product_reads),
                    .perf_product_writes(perf_product_writes),
                    .perf_product_starts(perf_product_starts),
                    .perf_weight_reads(perf_weight_reads)
                );
            end
        end else begin : g_linear5
            assign weight_context_release_ready = 1'b0;
            qfit_linear5_projection_top #(
                .HEIGHT(HEIGHT),
                .WIDTH(WIDTH),
                .TIME_PLANES(TIME_PLANES),
                .HEAD_DIM(HEAD_DIM),
                .OUT_DIM(OUT_DIM),
                .GATE_W(GATE_W),
                .W_W(W_W),
                .ACC_W(ACC_W)
            ) u_backend (
                .clk_core(clk_core), .rst_core(rst_core),
                .weight_valid(weight_valid), .weight_ready(weight_ready),
                .weight_lane(weight_lane), .weight_out(weight_out),
                .weight_data(weight_data), .weight_last(weight_last),
                .run_start(projection_start), .run_busy(projection_busy),
                .run_done(projection_done), .term_valid(term_valid),
                .term_ready(term_ready), .term_source_plane(term_source_plane),
                .term_source_y(term_source_y), .term_source_x(term_source_x),
                .term_lane(term_lane), .term_gate(term_gate),
                .term_destination_mask(term_destination_mask),
                .term_window_last(1'b0),
                .window_close(projection_close && projection_close_ready),
                .window_close_ready(backend_close_ready),
                .read_valid(read_valid), .read_ready(read_ready),
                .read_plane(read_plane), .read_y(read_y), .read_x(read_x),
                .read_out(read_out), .read_data_valid(read_data_valid),
                .read_data(read_data), .protocol_error(backend_protocol_error),
                .perf_product_terms(backend_terms),
                .perf_destination_updates(backend_updates)
            );
            assign perf_cache_hits = '0;
            assign perf_cache_misses = '0;
            assign perf_tag_compares = '0;
            assign perf_lru_writes = '0;
            assign perf_product_reads = '0;
            assign perf_product_writes = '0;
            assign perf_product_starts = backend_terms;
            assign perf_weight_reads = backend_terms * OUT_DIM;
        end
    endgenerate

    assign projection_close_ready = relation_done
                                  && descriptor_ready
                                  && !descriptor_valid
                                  && !term_valid
                                  && backend_close_ready;
    assign perf_product_terms = backend_terms;
    assign perf_destination_updates = backend_updates;
    assign perf_descriptors = builder_descriptors - descriptor_base_q;
    assign protocol_error = frontier_protocol_error
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
            if (
                (BACKEND_KIND != 0 || PRODUCT_CACHE_WAYS != 0)
                && (weight_context_release || projection_accumulate)
            )
                close_protocol_error_q <= 1'b1;
        end
    end

    logic unused_descriptor_last;
    logic unused_term_last;
    logic [31:0] unused_builder_terms;
    logic [31:0] unused_builder_updates;
    assign unused_descriptor_last = descriptor_last;
    assign unused_term_last = term_last;
    assign unused_builder_terms = builder_terms;
    assign unused_builder_updates = builder_updates;
endmodule

`default_nettype wire
